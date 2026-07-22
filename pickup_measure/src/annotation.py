from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image

from .scaler import ScaleMapping


@dataclass(frozen=True)
class AnnotationGeometry:
    outline_segments_mm: list[list[tuple[float, float]]]
    cab_start_x_mm: float
    roof_end_x_mm: float
    neck_x_mm: float
    door_seam_x_mm: float
    door_to_front_mm: float
    bed_top_y_mm: float
    cab_roof_y_mm: float
    neck_y_mm: float
    hood_front_y_mm: float
    chassis_y_mm: float
    ground_y_mm: float
    bed_height_mm: float
    cab_height_mm: float
    neck_height_mm: float
    hood_height_mm: float

    def points_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["outline_segments_mm"] = [
            [[round(x, 3), round(y, 3)] for x, y in segment]
            for segment in self.outline_segments_mm
        ]
        return payload


def _foreground_mask(image: Image.Image) -> np.ndarray:
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


def _envelope(mask: np.ndarray, x_start: int, x_end: int, top: bool) -> np.ndarray:
    values: list[int] = []
    for x in range(max(0, x_start), min(mask.shape[1], x_end)):
        rows = np.flatnonzero(mask[:, x])
        if rows.size:
            values.append(int(rows[0] if top else rows[-1]))
    if not values:
        raise RuntimeError("Could not determine annotation landmark envelope")
    return np.asarray(values)


def measure_and_trace(
    image: Image.Image,
    mapping: ScaleMapping,
    height_mm: float,
) -> AnnotationGeometry:
    mask = _foreground_mask(image)
    height, width = mask.shape
    bed_values = _envelope(mask, round(width * 0.07), round(width * 0.31), top=True)
    cab_values = _envelope(mask, round(width * 0.37), round(width * 0.59), top=True)
    neck_values = _envelope(mask, round(width * 0.73), round(width * 0.78), top=True)
    hood_values = _envelope(mask, round(width * 0.92), round(width * 0.98), top=True)
    chassis_values = _envelope(mask, round(width * 0.43), round(width * 0.68), top=False)

    bed_y_px = float(np.median(bed_values))
    cab_y_px = float(np.percentile(cab_values, 10))
    neck_y_px = float(np.median(neck_values))
    hood_y_px = float(np.median(hood_values))
    chassis_y_px = float(np.median(chassis_values))

    # Denoise the upper envelope to locate the bed/cab transition and roof slope.
    profile_x: list[int] = []
    profile_y: list[int] = []
    chassis_limit = max(1, round(chassis_y_px))
    for x in range(width):
        rows = np.flatnonzero(mask[:chassis_limit, x])
        if rows.size:
            profile_x.append(x)
            profile_y.append(int(rows[0]))
    if len(profile_x) < width * 0.4:
        raise RuntimeError("Could not derive the upper vehicle profile")
    profile_array = np.asarray(profile_y, dtype=np.float32)
    smooth_window = max(5, round(width * 0.015))
    if smooth_window % 2 == 0:
        smooth_window += 1
    padding = smooth_window // 2
    padded_profile = np.pad(profile_array, (padding, padding), mode="edge")
    profile_array = np.median(
        np.lib.stride_tricks.sliding_window_view(padded_profile, smooth_window), axis=1
    )
    transition_start = round(width * 0.24)
    transition_end = round(width * 0.46)
    dense_profile = np.interp(
        np.arange(width), np.asarray(profile_x), profile_array
    )
    transition_span = max(5, round(width * 0.006))
    profile_drop = dense_profile[transition_span:] - dense_profile[:-transition_span]
    # The cab begins at the left edge of the sustained bed-to-roof rise. Using
    # the maximum negative step avoids shortening the cab to the roof plateau.
    cab_start_x_px = transition_start + int(
        np.argmin(profile_drop[transition_start:transition_end])
    )
    roof_end_x_px = round(width * 0.64)
    neck_x_px = round(width * 0.77)
    front_x_px = width - 1
    bed_y_mm = bed_y_px * mapping.scale_y
    cab_y_mm = cab_y_px * mapping.scale_y
    neck_y_mm = neck_y_px * mapping.scale_y
    hood_y_mm = hood_y_px * mapping.scale_y
    chassis_y_mm = chassis_y_px * mapping.scale_y
    cab_start_x_mm = cab_start_x_px * mapping.scale_x
    roof_end_x_mm = roof_end_x_px * mapping.scale_x
    neck_x_mm = neck_x_px * mapping.scale_x
    front_x_mm = front_x_px * mapping.scale_x

    # Detect the full-height B-pillar/door seam using vertical gradient energy in
    # the lower door panels. Handles and mirrors do not persist through this ROI.
    grayscale = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    vertical_edges = np.abs(cv2.Sobel(grayscale, cv2.CV_32F, 1, 0, ksize=3))
    cabin_span = neck_x_px - cab_start_x_px
    door_search_left = cab_start_x_px + round(cabin_span * 0.16)
    door_search_right = neck_x_px - round(cabin_span * 0.16)
    door_y_start = round(height * 0.38)
    door_y_end = min(height, round(chassis_y_px * 0.98))
    door_scores = vertical_edges[door_y_start:door_y_end].sum(axis=0)
    door_scores = cv2.GaussianBlur(door_scores.reshape(1, -1), (0, 0), sigmaX=3).reshape(-1)
    door_seam_x_px = door_search_left + int(
        np.argmax(door_scores[door_search_left:door_search_right])
    )
    door_seam_x_mm = door_seam_x_px * mapping.scale_x
    door_to_front_mm = max(0.0, front_x_mm - door_seam_x_mm)

    outline_segments_mm = [
        [(0.0, chassis_y_mm), (0.0, bed_y_mm), (cab_start_x_mm, bed_y_mm), (cab_start_x_mm, chassis_y_mm), (0.0, chassis_y_mm)],
        [(cab_start_x_mm, chassis_y_mm), (cab_start_x_mm, cab_y_mm), (roof_end_x_mm, cab_y_mm), (neck_x_mm, neck_y_mm), (neck_x_mm, chassis_y_mm), (cab_start_x_mm, chassis_y_mm)],
        [(neck_x_mm, chassis_y_mm), (neck_x_mm, neck_y_mm), (front_x_mm, hood_y_mm), (front_x_mm, chassis_y_mm), (neck_x_mm, chassis_y_mm)],
    ]

    return AnnotationGeometry(
        outline_segments_mm=outline_segments_mm,
        cab_start_x_mm=cab_start_x_mm,
        roof_end_x_mm=roof_end_x_mm,
        neck_x_mm=neck_x_mm,
        door_seam_x_mm=door_seam_x_mm,
        door_to_front_mm=door_to_front_mm,
        bed_top_y_mm=bed_y_mm,
        cab_roof_y_mm=cab_y_mm,
        neck_y_mm=neck_y_mm,
        hood_front_y_mm=hood_y_mm,
        chassis_y_mm=chassis_y_mm,
        ground_y_mm=height_mm,
        bed_height_mm=max(0.0, chassis_y_mm - bed_y_mm),
        cab_height_mm=max(0.0, chassis_y_mm - cab_y_mm),
        neck_height_mm=max(0.0, chassis_y_mm - neck_y_mm),
        hood_height_mm=max(0.0, chassis_y_mm - hood_y_mm),
    )
