"""Background classification and vehicle foreground segmentation."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def classify_background(
    image: Image.Image,
    ai_hint: str | None = None,
) -> str:
    """Classify white/transparent/environment backgrounds with an AI soft hint."""
    if image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    ):
        alpha = np.asarray(image.convert("RGBA"))[:, :, 3]
        if np.count_nonzero(alpha < 250) >= alpha.size * 0.02:
            return "transparent"

    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    border = max(2, round(min(width, height) * 0.03))
    border_pixels = np.concatenate(
        (
            rgb[:border].reshape(-1, 3),
            rgb[-border:].reshape(-1, 3),
            rgb[:, :border].reshape(-1, 3),
            rgb[:, -border:].reshape(-1, 3),
        ),
        axis=0,
    )
    white_fraction = float(np.mean(np.all(border_pixels >= 240, axis=1)))
    neutral_fraction = float(np.mean(
        np.max(border_pixels, axis=1) - np.min(border_pixels, axis=1) <= 12
    ))
    if white_fraction >= 0.78 and neutral_fraction >= 0.90:
        return "white"
    if ai_hint == "environment":
        return "environment"
    # Qwen is a soft hint only: do not let white vehicle paint in an outdoor
    # image switch the segmentation path unless the border pixels also support
    # a studio background.
    if (
        ai_hint == "white"
        and white_fraction >= 0.55
        and neutral_fraction >= 0.80
    ):
        return "white"
    return "environment"


def _environment_foreground_mask(
    image: Image.Image,
    max_analysis_dimension: int = 1200,
) -> np.ndarray:
    """Segment a side-view vehicle without learning trees, road, or buildings."""
    full_rgb = np.asarray(image.convert("RGB"))
    full_height, full_width = full_rgb.shape[:2]
    scale = min(
        1.0,
        max_analysis_dimension / max(full_width, full_height),
    )
    if scale < 1.0:
        width = max(2, round(full_width * scale))
        height = max(2, round(full_height * scale))
        rgb = cv2.resize(
            full_rgb,
            (width, height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        rgb = full_rgb
        height, width = full_height, full_width

    mask = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)
    # With normalized front-right side views these rectangles lie inside the
    # painted bed, cab, hood, and rocker even when the body style changes.
    seed_boxes = (
        (0.07, 0.30, 0.30, 0.58),
        (0.37, 0.66, 0.22, 0.58),
        (0.72, 0.94, 0.32, 0.58),
        (0.34, 0.74, 0.56, 0.70),
        (0.13, 0.29, 0.64, 0.82),
        (0.75, 0.91, 0.64, 0.82),
    )
    seed_mask = np.zeros_like(mask)
    for left, right, top, bottom in seed_boxes:
        x1, x2 = round(width * left), round(width * right)
        y1, y2 = round(height * top), round(height * bottom)
        mask[y1:y2, x1:x2] = cv2.GC_FGD
        seed_mask[y1:y2, x1:x2] = 255

    # The top outer corners cannot contain the cab on a normalized pickup.
    top_band = max(1, round(height * 0.08))
    mask[:top_band, :round(width * 0.28)] = cv2.GC_BGD
    mask[:top_band, round(width * 0.78):] = cv2.GC_BGD

    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            mask,
            None,
            background_model,
            foreground_model,
            3,
            cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error as exc:
        raise RuntimeError(
            "Could not isolate the vehicle from its environment"
        ) from exc

    foreground = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    kernel_size = max(3, round(min(width, height) * 0.006))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, 8)
    selected = np.zeros_like(foreground)
    for label in range(1, count):
        component = labels == label
        overlap = int(np.count_nonzero(component & (seed_mask > 0)))
        area = int(stats[label, cv2.CC_STAT_AREA])
        if overlap >= max(10, round(area * 0.01)):
            selected[component] = 255
    if not np.any(selected):
        raise RuntimeError(
            "Could not connect environmental foreground to the vehicle body"
        )

    if scale < 1.0:
        selected = cv2.resize(
            selected,
            (full_width, full_height),
            interpolation=cv2.INTER_NEAREST,
        )
    return selected


def _foreground_mask(
    image: Image.Image,
    background_type: str | None = None,
) -> np.ndarray:
    background_type = background_type or classify_background(image)
    if background_type == "environment":
        return _environment_foreground_mask(image)

    if background_type == "transparent":
        alpha = np.asarray(image.convert("RGBA"))[:, :, 3]
        mask = np.where(alpha >= 16, 255, 0).astype(np.uint8)
        if np.any(mask):
            return mask

    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    distance_from_white = np.linalg.norm(255.0 - rgb.astype(np.float32), axis=2)
    mask = np.where(distance_from_white > 22.0, 255, 0).astype(np.uint8)

    kernel_size = max(3, round(min(width, height) * 0.006))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # Remove the broad photographic ground shadow while retaining both tyres.
    lower_start = round(height * 0.87)
    keep = np.zeros_like(mask)
    keep[:lower_start, :] = 255
    keep[lower_start:, round(width * 0.08):round(width * 0.40)] = 255
    keep[lower_start:, round(width * 0.60):round(width * 0.92)] = 255
    mask = cv2.bitwise_and(mask, keep)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        raise RuntimeError("Could not isolate a vehicle silhouette for annotation")
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)
