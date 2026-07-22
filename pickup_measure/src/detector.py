from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np
from PIL import Image

from .geometry import Bounds


class VehicleDetector(ABC):
    """Extension point for a future YOLO/SAM automatic detector."""

    @abstractmethod
    def detect(self, image: Image.Image) -> Bounds:
        """Return vehicle bounds in source-image pixels or raise on failure."""
        raise NotImplementedError


class OpenCVVehicleDetector(VehicleDetector):
    """Foreground-based detector intended for clean side-profile product images."""

    def __init__(self, max_working_size: int = 1400, iterations: int = 5):
        self.max_working_size = max_working_size
        self.iterations = iterations

    def detect(self, image: Image.Image) -> Bounds:
        rgb = np.asarray(image.convert("RGB"))
        original_height, original_width = rgb.shape[:2]
        scale = min(1.0, self.max_working_size / max(original_width, original_height))
        if scale < 1.0:
            working = cv2.resize(
                rgb,
                (round(original_width * scale), round(original_height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        else:
            working = rgb

        height, width = working.shape[:2]
        if width < 40 or height < 40:
            raise RuntimeError("Image is too small for automatic vehicle detection")

        bgr = cv2.cvtColor(working, cv2.COLOR_RGB2BGR)
        mask = np.full((height, width), cv2.GC_PR_BGD, dtype=np.uint8)
        border = max(2, round(min(width, height) * 0.02))
        mask[:border, :] = cv2.GC_BGD
        mask[-border:, :] = cv2.GC_BGD
        mask[:, :border] = cv2.GC_BGD
        mask[:, -border:] = cv2.GC_BGD

        # Border colour provides a strong prior for white/solid studio backgrounds.
        border_pixels = np.concatenate(
            (working[:border].reshape(-1, 3), working[-border:].reshape(-1, 3),
             working[:, :border].reshape(-1, 3), working[:, -border:].reshape(-1, 3)),
            axis=0,
        )
        background_colour = np.median(border_pixels, axis=0)
        colour_distance = np.linalg.norm(working.astype(np.float32) - background_colour, axis=2)
        mask[colour_distance > 32.0] = cv2.GC_PR_FGD

        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(
                bgr,
                mask,
                None,
                bg_model,
                fg_model,
                self.iterations,
                cv2.GC_INIT_WITH_MASK,
            )
        except cv2.error as exc:
            raise RuntimeError("OpenCV foreground segmentation failed") from exc

        foreground = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)
        kernel_size = max(3, round(min(width, height) * 0.008))
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=2)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel, iterations=1)

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(foreground, 8)
        candidates: list[tuple[float, int]] = []
        image_area = width * height
        for label in range(1, count):
            x, y, component_width, component_height, area = stats[label]
            if area < image_area * 0.01 or component_width < width * 0.2:
                continue
            aspect = component_width / max(component_height, 1)
            center_x, center_y = centroids[label]
            centrality = 1.0 - min(
                0.8,
                abs(center_x / width - 0.5) + 0.5 * abs(center_y / height - 0.55),
            )
            wide_bonus = min(aspect, 4.0) / 2.0
            candidates.append((area * centrality * wide_bonus, label))

        if not candidates:
            raise RuntimeError("No plausible vehicle foreground was detected")
        _, best_label = max(candidates)
        selected = np.where(labels == best_label, 255, 0).astype(np.uint8)
        x, y, box_width, box_height = cv2.boundingRect(selected)

        # Ground shadows often connect to the tyres and can span almost the full canvas.
        # Preserve the vertical result, but infer left/right from the upper 88% of the
        # detected object where bumpers and body panels live.
        body_band_bottom = y + max(1, round(box_height * 0.88))
        column_counts = np.count_nonzero(selected[y:body_band_bottom, :], axis=0)
        minimum_column_pixels = max(3, round(box_height * 0.012))
        active_columns = np.flatnonzero(column_counts >= minimum_column_pixels)
        if active_columns.size >= 2:
            projected_left = int(active_columns[0])
            projected_right = int(active_columns[-1] + 1)
            if projected_right - projected_left >= width * 0.2:
                x = projected_left
                box_width = projected_right - projected_left

        # Place the crop's ground edge on the lowest tyre contact pixel. The two
        # side zones cover the usual rear/front wheel positions while excluding
        # most of a broad centre shadow.
        grayscale = cv2.cvtColor(working, cv2.COLOR_RGB2GRAY)
        wheel_zones = (
            (x + round(box_width * 0.08), x + round(box_width * 0.40)),
            (x + round(box_width * 0.60), x + round(box_width * 0.92)),
        )
        tyre_search_top = y + round(box_height * 0.55)
        minimum_tyre_pixels = max(5, round(box_width * 0.008))
        tyre_bottom_rows: list[int] = []
        for zone_left, zone_right in wheel_zones:
            dark_counts = np.count_nonzero(
                grayscale[tyre_search_top:y + box_height, zone_left:zone_right] < 100,
                axis=1,
            )
            contact_rows = np.flatnonzero(dark_counts >= minimum_tyre_pixels)
            if contact_rows.size:
                tyre_bottom_rows.append(tyre_search_top + int(contact_rows[-1]) + 1)
        if tyre_bottom_rows:
            tyre_ground = max(tyre_bottom_rows)
            maximum_safe_trim = round(box_height * 0.06)
            if 0 <= y + box_height - tyre_ground <= maximum_safe_trim:
                box_height = tyre_ground - y

        fill_ratio = stats[best_label, cv2.CC_STAT_AREA] / (box_width * box_height)
        box_area_ratio = box_width * box_height / image_area
        aspect = box_width / max(box_height, 1)
        if box_area_ratio > 0.92 or fill_ratio > 0.92 or aspect < 1.2:
            raise RuntimeError(
                "Automatic detection confidence is too low; use --manual for this image"
            )

        inverse_scale = 1.0 / scale
        bounds = Bounds(
            left=max(0, int(np.floor(x * inverse_scale))),
            right=min(original_width, int(np.ceil((x + box_width) * inverse_scale))),
            roof=max(0, int(np.floor(y * inverse_scale))),
            ground=min(original_height, int(np.ceil((y + box_height) * inverse_scale))),
        )
        bounds.validate(original_width, original_height)
        return bounds
