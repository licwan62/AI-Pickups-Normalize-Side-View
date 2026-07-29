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
    bed_top_y_mm: float
    cab_roof_y_mm: float
    neck_y_mm: float
    hood_front_y_mm: float
    chassis_y_mm: float
    chassis_left_y_mm: float
    chassis_right_y_mm: float
    ground_y_mm: float
    bed_height_mm: float
    cab_height_mm: float
    neck_height_mm: float
    hood_height_mm: float

    def chassis_y_at(self, x_mm: float) -> float:
        if not self.outline_segments_mm:
            return self.chassis_y_mm
        vehicle_width_mm = self.outline_segments_mm[-1][2][0]
        if vehicle_width_mm <= 0:
            return self.chassis_y_mm
        fraction = min(1.0, max(0.0, x_mm / vehicle_width_mm))
        return (
            self.chassis_left_y_mm
            + (self.chassis_right_y_mm - self.chassis_left_y_mm) * fraction
        )

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


def _chassis_line(image: Image.Image, fallback_y: float) -> tuple[float, float]:
    """Detect the lower body edge and extrapolate it across the vehicle width."""
    grayscale = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    height, width = grayscale.shape
    edges = cv2.Canny(grayscale, 30, 120)

    roi = np.zeros_like(edges)
    x_start, x_end = round(width * 0.25), round(width * 0.75)
    y_start = round(height * 0.62)
    y_end = min(height, round(height * 0.91))
    roi[y_start:y_end, x_start:x_end] = edges[y_start:y_end, x_start:x_end]
    lines = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi / 720,
        threshold=max(12, round(width * 0.035)),
        minLineLength=max(20, round(width * 0.08)),
        maxLineGap=max(4, round(width * 0.04)),
    )
    if lines is None:
        return fallback_y, fallback_y

    maximum_y = fallback_y - max(3.0, height * 0.035)
    candidates: list[tuple[float, float, float, float]] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if x1 == x2:
            continue
        if x2 < x1:
            x1, x2, y1, y2 = x2, x1, y2, y1
        length = x2 - x1
        slope = (y2 - y1) / length
        midpoint_y = (y1 + y2) / 2
        if abs(slope) > 0.12:
            continue
        if not height * 0.68 <= midpoint_y <= maximum_y:
            continue
        midpoint_x = (x1 + x2) / 2
        # Prefer the lowest sustained body edge, with a modest length bonus.
        score = midpoint_y + length * 0.03
        candidates.append((score, slope, midpoint_x, midpoint_y))
    if not candidates:
        return fallback_y, fallback_y

    _, local_slope, midpoint_x, midpoint_y = max(
        candidates, key=lambda item: item[0]
    )

    # Estimate vehicle pitch from all long near-horizontal edges. This keeps a
    # short diagonal wheel-arch or bumper edge from tilting the whole chassis.
    pitch_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 720,
        threshold=max(20, round(width * 0.05)),
        minLineLength=max(30, round(width * 0.15)),
        maxLineGap=max(4, round(width * 0.04)),
    )
    weighted_slopes: list[tuple[float, float]] = []
    if pitch_lines is not None:
        for x1, y1, x2, y2 in pitch_lines.reshape(-1, 4):
            if x1 == x2:
                continue
            length = abs(x2 - x1)
            slope = (y2 - y1) / (x2 - x1)
            line_midpoint_y = (y1 + y2) / 2
            if abs(slope) <= 0.10 and height * 0.04 < line_midpoint_y < height * 0.88:
                weighted_slopes.append((slope, float(length)))
    if not weighted_slopes:
        return fallback_y, fallback_y
    weighted_slopes.sort(key=lambda item: item[0])
    total_weight = sum(weight for _, weight in weighted_slopes)
    half_weight = total_weight / 2
    accumulated = 0.0
    slope = local_slope
    for candidate_slope, weight in weighted_slopes:
        accumulated += weight
        if accumulated >= half_weight:
            slope = candidate_slope
            break
    coherent_weight = sum(
        weight
        for candidate_slope, weight in weighted_slopes
        if abs(candidate_slope - slope) <= 0.01
    )
    pitch_is_coherent = coherent_weight / total_weight >= 0.35
    # On level vehicles, a nearly horizontal edge spanning most of the image is
    # a stronger chassis cue than the foreground envelope, which can absorb a
    # short central ground shadow.
    if abs(slope) < 0.012 or not pitch_is_coherent:
        flat_edges: list[tuple[float, float]] = []
        if pitch_lines is not None:
            for x1, y1, x2, y2 in pitch_lines.reshape(-1, 4):
                if x1 == x2:
                    continue
                length = abs(x2 - x1)
                candidate_slope = (y2 - y1) / (x2 - x1)
                candidate_y = (y1 + y2) / 2
                if (
                    length >= width * 0.70
                    and abs(candidate_slope) <= 0.012
                    and fallback_y - height * 0.25 <= candidate_y < fallback_y
                ):
                    flat_edges.append((candidate_y, float(length)))
        if flat_edges:
            chassis_y = max(flat_edges, key=lambda item: (item[0], item[1]))[0]
            return chassis_y, chassis_y
        return fallback_y, fallback_y

    intercept = midpoint_y - slope * midpoint_x
    left_y = float(np.clip(intercept, height * 0.55, fallback_y))
    right_y = float(np.clip(intercept + slope * (width - 1), height * 0.55, fallback_y))
    return left_y, right_y


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
    chassis_left_y_px, chassis_right_y_px = _chassis_line(image, chassis_y_px)

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
    chassis_left_y_mm = chassis_left_y_px * mapping.scale_y
    chassis_right_y_mm = chassis_right_y_px * mapping.scale_y
    chassis_y_mm = (chassis_left_y_mm + chassis_right_y_mm) / 2
    cab_start_x_mm = cab_start_x_px * mapping.scale_x
    roof_end_x_mm = roof_end_x_px * mapping.scale_x
    neck_x_mm = neck_x_px * mapping.scale_x
    front_x_mm = front_x_px * mapping.scale_x

    def chassis_at(x_mm: float) -> float:
        fraction = x_mm / max(front_x_mm, 1.0)
        return chassis_left_y_mm + (chassis_right_y_mm - chassis_left_y_mm) * fraction

    bed_chassis_y_mm = chassis_at(0.0)
    cab_chassis_y_mm = chassis_at(cab_start_x_mm)
    neck_chassis_y_mm = chassis_at(neck_x_mm)
    hood_chassis_y_mm = chassis_at(front_x_mm)
    outline_segments_mm = [
        [(0.0, bed_chassis_y_mm), (0.0, bed_y_mm), (cab_start_x_mm, bed_y_mm), (cab_start_x_mm, cab_chassis_y_mm), (0.0, bed_chassis_y_mm)],
        [(cab_start_x_mm, cab_chassis_y_mm), (cab_start_x_mm, cab_y_mm), (roof_end_x_mm, cab_y_mm), (neck_x_mm, neck_y_mm), (neck_x_mm, neck_chassis_y_mm), (cab_start_x_mm, cab_chassis_y_mm)],
        [(neck_x_mm, neck_chassis_y_mm), (neck_x_mm, neck_y_mm), (front_x_mm, hood_y_mm), (front_x_mm, hood_chassis_y_mm), (neck_x_mm, neck_chassis_y_mm)],
    ]

    return AnnotationGeometry(
        outline_segments_mm=outline_segments_mm,
        cab_start_x_mm=cab_start_x_mm,
        roof_end_x_mm=roof_end_x_mm,
        neck_x_mm=neck_x_mm,
        bed_top_y_mm=bed_y_mm,
        cab_roof_y_mm=cab_y_mm,
        neck_y_mm=neck_y_mm,
        hood_front_y_mm=hood_y_mm,
        chassis_y_mm=chassis_y_mm,
        chassis_left_y_mm=chassis_left_y_mm,
        chassis_right_y_mm=chassis_right_y_mm,
        ground_y_mm=height_mm,
        bed_height_mm=max(0.0, bed_chassis_y_mm - bed_y_mm),
        cab_height_mm=max(0.0, cab_chassis_y_mm - cab_y_mm),
        neck_height_mm=max(0.0, neck_chassis_y_mm - neck_y_mm),
        hood_height_mm=max(0.0, hood_chassis_y_mm - hood_y_mm),
    )
