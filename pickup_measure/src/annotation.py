from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np
from PIL import Image

from .scaler import ScaleMapping


@dataclass(frozen=True)
class AnnotationGeometry:
    outline_segments_mm: list[list[tuple[float, float]]]
    is_pickup: bool
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
    dimension_definitions: dict[str, object] = field(default_factory=dict)
    region_types: dict[str, str] = field(default_factory=dict)

    def chassis_y_at(self, x_mm: float) -> float:
        if not self.outline_segments_mm:
            return self.chassis_y_mm
        vehicle_width_mm = max(
            point[0]
            for segment in self.outline_segments_mm
            for point in segment
        )
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


def _geometry_from_api_polylines_legacy(
    structure: dict[str, object],
    mapping: ScaleMapping,
    image_width_px: int,
    height_mm: float,
) -> AnnotationGeometry:
    """Generate red regions from semantic points and real structure polylines."""
    vehicle_type = str(structure.get("vehicle_type", ""))
    if vehicle_type not in {"pickup", "non_pickup"}:
        raise RuntimeError("API structure has an invalid vehicle_type")
    raw_keypoints = structure.get("keypoints")
    raw_polylines = structure.get("polylines")
    if not isinstance(raw_keypoints, dict) or not isinstance(raw_polylines, dict):
        raise RuntimeError("API structure is missing keypoints or polylines")

    def point(name: str) -> tuple[float, float]:
        raw = raw_keypoints.get(name)
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise RuntimeError(f"API structure is missing keypoint {name}")
        return float(raw[0]), float(raw[1])

    def line(name: str, required: bool = True) -> list[tuple[float, float]] | None:
        raw = raw_polylines.get(name)
        if raw is None and not required:
            return None
        if not isinstance(raw, list) or len(raw) < 2:
            raise RuntimeError(f"API structure is missing polyline {name}")
        return [(float(raw_point[0]), float(raw_point[1])) for raw_point in raw]

    lower = line("vehicle_lower_body_line")
    cab_roof = line("cab_roof_line")
    cab_rear = line("cab_rear_line")
    windshield = line("windshield_line")
    hood_top = line("hood_top_line")
    front_profile = line("front_profile_line")
    rocker = line("rocker_line")
    assert lower and cab_roof and cab_rear and windshield
    assert hood_top and front_profile and rocker

    def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def oriented(
        points: list[tuple[float, float]],
        start: tuple[float, float],
    ) -> list[tuple[float, float]]:
        if distance(points[-1], start) < distance(points[0], start):
            return list(reversed(points))
        return list(points)

    def at_x(points: list[tuple[float, float]], x: float) -> tuple[float, float]:
        ordered = sorted(points, key=lambda item: item[0])
        for first, second in zip(ordered, ordered[1:]):
            if first[0] <= x <= second[0]:
                span = second[0] - first[0]
                if abs(span) < 1e-6:
                    return x, (first[1] + second[1]) / 2
                fraction = (x - first[0]) / span
                return x, first[1] + (second[1] - first[1]) * fraction
        endpoint = min(ordered, key=lambda item: abs(item[0] - x))
        return x, endpoint[1]

    def between_x(
        points: list[tuple[float, float]],
        start_x: float,
        end_x: float,
    ) -> list[tuple[float, float]]:
        low, high = sorted((start_x, end_x))
        selected = [at_x(points, low)]
        selected.extend(
            item for item in sorted(points, key=lambda item: item[0])
            if low < item[0] < high
        )
        selected.append(at_x(points, high))
        return selected if start_x <= end_x else list(reversed(selected))

    def joined(*paths: list[tuple[float, float]]) -> list[tuple[float, float]]:
        result: list[tuple[float, float]] = []
        for path in paths:
            for item in path:
                if not result or distance(result[-1], item) > 0.5:
                    result.append(item)
        if result and distance(result[0], result[-1]) > 0.5:
            result.append(result[0])
        return result

    rocker_rear = point("rocker_rear")
    rocker_front = point("rocker_front")
    cab_rear_roof = point("cab_rear_roof")
    cab_roof_front = point("cab_roof_front")
    windshield_base = point("windshield_base")
    hood_rear = point("hood_rear_top")
    hood_front = point("hood_front_top")
    rear_bumper = point("rear_bumper_outermost")
    front_bumper = point("front_bumper_outermost")

    cab_segment_px = joined(
        oriented(cab_rear, rocker_rear),
        oriented(cab_roof, cab_rear_roof),
        oriented(windshield, cab_roof_front),
        [windshield_base, rocker_front],
        list(reversed(oriented(rocker, rocker_rear))),
    )
    hood_lower = between_x(lower, rocker_front[0], front_bumper[0])
    hood_segment_px = joined(
        oriented(hood_top, hood_rear),
        oriented(front_profile, hood_front),
        list(reversed(hood_lower)),
        [rocker_front, hood_rear],
    )

    segments_px: list[list[tuple[float, float]]] = []
    bed_top_y_px = at_x(lower, rear_bumper[0])[1]
    bed_height_px = 0.0
    if vehicle_type == "pickup":
        bed_top = line("bed_top_line")
        assert bed_top
        gap_top = point("bed_cab_gap_top")
        gap_bottom = point("bed_cab_gap_bottom")
        tailgate_top = point("tailgate_top_rear")
        bed_lower = between_x(lower, rear_bumper[0], gap_bottom[0])
        bed_segment_px = joined(
            [at_x(lower, rear_bumper[0]), rear_bumper, tailgate_top],
            oriented(bed_top, point("bed_rail_rear")),
            [gap_top, gap_bottom],
            list(reversed(bed_lower)),
        )
        segments_px.append(bed_segment_px)
        bed_top_y_px = float(np.mean([item[1] for item in bed_top]))
        bed_mid_x = (point("bed_rail_rear")[0] + point("bed_rail_front")[0]) / 2
        bed_height_px = max(0.0, at_x(lower, bed_mid_x)[1] - bed_top_y_px)

    segments_px.extend([cab_segment_px, hood_segment_px])
    outline_segments_mm = [
        [(x * mapping.scale_x, y * mapping.scale_y) for x, y in segment]
        for segment in segments_px
    ]

    roof_highest = point("roof_highest")
    cab_base = at_x(rocker, roof_highest[0])
    hood_base = at_x(lower, hood_front[0])
    neck_base = at_x(lower, windshield_base[0])
    lower_left = lower[0]
    lower_right = lower[-1]
    chassis_left_y_mm = lower_left[1] * mapping.scale_y
    chassis_right_y_mm = lower_right[1] * mapping.scale_y

    def mm(raw_point: tuple[float, float]) -> list[float]:
        return [
            round(raw_point[0] * mapping.scale_x, 3),
            round(raw_point[1] * mapping.scale_y, 3),
        ]

    definitions: dict[str, object] = {
        "overall_height": {
            "start_point": mm(point("rear_tire_contact")),
            "end_point": mm(roof_highest),
            "baseline": "ground_line",
        },
        "cab_height": {
            "start_point": mm(cab_base),
            "end_point": mm(roof_highest),
            "baseline": "rocker_line",
        },
        "neck_height": {
            "start_point": mm(neck_base),
            "end_point": mm(windshield_base),
            "baseline": "vehicle_lower_body_line",
        },
        "hood_height": {
            "start_point": mm(hood_base),
            "end_point": mm(hood_front),
            "baseline": "vehicle_lower_body_line",
        },
        "bed_height": None,
    }
    if vehicle_type == "pickup":
        bed_mid_x = (point("bed_rail_rear")[0] + point("bed_rail_front")[0]) / 2
        bed_base = at_x(lower, bed_mid_x)
        bed_top_point = at_x(line("bed_top_line") or [], bed_mid_x)
        definitions["bed_height"] = {
            "start_point": mm(bed_base),
            "end_point": mm(bed_top_point),
            "baseline": "vehicle_lower_body_line",
        }

    return AnnotationGeometry(
        outline_segments_mm=outline_segments_mm,
        is_pickup=vehicle_type == "pickup",
        cab_start_x_mm=rocker_rear[0] * mapping.scale_x,
        roof_end_x_mm=windshield_base[0] * mapping.scale_x,
        neck_x_mm=hood_rear[0] * mapping.scale_x,
        bed_top_y_mm=bed_top_y_px * mapping.scale_y,
        cab_roof_y_mm=roof_highest[1] * mapping.scale_y,
        neck_y_mm=windshield_base[1] * mapping.scale_y,
        hood_front_y_mm=hood_front[1] * mapping.scale_y,
        chassis_y_mm=(chassis_left_y_mm + chassis_right_y_mm) / 2,
        chassis_left_y_mm=chassis_left_y_mm,
        chassis_right_y_mm=chassis_right_y_mm,
        ground_y_mm=height_mm,
        bed_height_mm=bed_height_px * mapping.scale_y,
        cab_height_mm=max(0.0, (cab_base[1] - roof_highest[1]) * mapping.scale_y),
        neck_height_mm=max(
            0.0,
            (neck_base[1] - windshield_base[1]) * mapping.scale_y,
        ),
        hood_height_mm=max(0.0, (hood_base[1] - hood_front[1]) * mapping.scale_y),
        dimension_definitions=definitions,
    )


def geometry_from_api_structure(
    structure: dict[str, object],
    mapping: ScaleMapping,
    image_width_px: int,
    height_mm: float,
) -> AnnotationGeometry:
    """Build one rectangle and two trapezoids on one Python chassis line."""
    vehicle_type = str(structure.get("vehicle_type", ""))
    if vehicle_type not in {"pickup", "non_pickup"}:
        raise RuntimeError("API structure has an invalid vehicle_type")
    raw_keypoints = structure.get("keypoints")
    raw_polylines = structure.get("polylines")
    if not isinstance(raw_keypoints, dict) or not isinstance(raw_polylines, dict):
        raise RuntimeError("API structure is missing keypoints or polylines")

    def point(name: str) -> tuple[float, float]:
        raw = raw_keypoints.get(name)
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            raise RuntimeError(f"API structure is missing keypoint {name}")
        return float(raw[0]), float(raw[1])

    raw_chassis = structure.get("python_chassis_line")
    if not isinstance(raw_chassis, list) or len(raw_chassis) != 2:
        lower = raw_polylines.get("vehicle_lower_body_line")
        if not isinstance(lower, list) or len(lower) < 2:
            raise RuntimeError("API structure has no usable chassis line")
        raw_chassis = [lower[0], lower[-1]]
    chassis = sorted(
        [
            (float(raw_chassis[0][0]), float(raw_chassis[0][1])),
            (float(raw_chassis[1][0]), float(raw_chassis[1][1])),
        ],
        key=lambda item: item[0],
    )
    if chassis[1][0] <= chassis[0][0]:
        raise RuntimeError("Python chassis line has no horizontal span")
    chassis_slope = (
        (chassis[1][1] - chassis[0][1])
        / (chassis[1][0] - chassis[0][0])
    )

    def chassis_at(x: float) -> tuple[float, float]:
        return (
            x,
            chassis[0][1] + chassis_slope * (x - chassis[0][0]),
        )

    def closed(
        bottom_left: tuple[float, float],
        top_left: tuple[float, float],
        top_right: tuple[float, float],
        bottom_right: tuple[float, float],
    ) -> list[tuple[float, float]]:
        return [bottom_left, top_left, top_right, bottom_right, bottom_left]

    gap_bottom = (
        point("bed_cab_gap_bottom")
        if vehicle_type == "pickup"
        else point("rocker_rear")
    )
    rocker_front = point("rocker_front")
    cab_rear_roof = point("cab_rear_roof")
    cab_roof_front = point("cab_roof_front")
    roof_highest = point("roof_highest")
    hood_rear = point("hood_rear_top")
    hood_front = point("hood_front_top")
    front_bumper = point("front_bumper_outermost")

    cab_segment_px = closed(
        chassis_at(gap_bottom[0]),
        cab_rear_roof,
        cab_roof_front,
        chassis_at(rocker_front[0]),
    )
    hood_segment_px = closed(
        chassis_at(rocker_front[0]),
        hood_rear,
        hood_front,
        chassis_at(front_bumper[0]),
    )

    segments_px: list[list[tuple[float, float]]] = []
    bed_height_px = 0.0
    bed_top_y_px = chassis_at(gap_bottom[0])[1]
    if vehicle_type == "pickup":
        bed_rear = point("bed_rail_rear")
        bed_front = point("bed_rail_front")
        raw_bed_line = raw_polylines.get("bed_top_line")
        if not isinstance(raw_bed_line, list) or len(raw_bed_line) < 2:
            raise RuntimeError("API structure is missing bed_top_line")
        observed_bed_top_y = float(
            np.mean([float(raw_point[1]) for raw_point in raw_bed_line])
        )
        bed_mid_x = (bed_rear[0] + bed_front[0]) / 2
        bed_height_px = max(
            1.0,
            chassis_at(bed_mid_x)[1] - observed_bed_top_y,
        )
        bed_left_x = bed_rear[0]
        bed_right_x = gap_bottom[0]
        bed_bottom_left = chassis_at(bed_left_x)
        bed_bottom_right = chassis_at(bed_right_x)
        bed_top_left = (
            bed_left_x,
            bed_bottom_left[1] - bed_height_px,
        )
        bed_top_right = (
            bed_right_x,
            bed_bottom_right[1] - bed_height_px,
        )
        segments_px.append(
            closed(
                bed_bottom_left,
                bed_top_left,
                bed_top_right,
                bed_bottom_right,
            )
        )
        bed_top_y_px = (bed_top_left[1] + bed_top_right[1]) / 2

    segments_px.extend([cab_segment_px, hood_segment_px])
    outline_segments_mm = [
        [(x * mapping.scale_x, y * mapping.scale_y) for x, y in segment]
        for segment in segments_px
    ]

    cab_base = chassis_at(roof_highest[0])
    neck_top = point("windshield_base")
    neck_base = chassis_at(neck_top[0])
    hood_base = chassis_at(hood_front[0])
    chassis_left_y_mm = chassis_at(0.0)[1] * mapping.scale_y
    chassis_right_y_mm = chassis_at(image_width_px - 1)[1] * mapping.scale_y

    def mm(raw_point: tuple[float, float]) -> list[float]:
        return [
            round(raw_point[0] * mapping.scale_x, 3),
            round(raw_point[1] * mapping.scale_y, 3),
        ]

    definitions: dict[str, object] = {
        "overall_height": {
            "start_point": mm(chassis_at(roof_highest[0])),
            "end_point": mm(roof_highest),
            "baseline": "python_chassis_line",
        },
        "cab_height": {
            "start_point": mm(cab_base),
            "end_point": mm(roof_highest),
            "baseline": "python_chassis_line",
        },
        "neck_height": {
            "start_point": mm(neck_base),
            "end_point": mm(neck_top),
            "baseline": "python_chassis_line",
        },
        "hood_height": {
            "start_point": mm(hood_base),
            "end_point": mm(hood_front),
            "baseline": "python_chassis_line",
        },
        "bed_height": None,
    }
    if vehicle_type == "pickup":
        bed_mid_x = (
            point("bed_rail_rear")[0] + point("bed_rail_front")[0]
        ) / 2
        definitions["bed_height"] = {
            "start_point": mm(chassis_at(bed_mid_x)),
            "end_point": mm(
                (bed_mid_x, chassis_at(bed_mid_x)[1] - bed_height_px)
            ),
            "baseline": "python_chassis_line",
        }

    return AnnotationGeometry(
        outline_segments_mm=outline_segments_mm,
        is_pickup=vehicle_type == "pickup",
        cab_start_x_mm=gap_bottom[0] * mapping.scale_x,
        roof_end_x_mm=rocker_front[0] * mapping.scale_x,
        neck_x_mm=hood_rear[0] * mapping.scale_x,
        bed_top_y_mm=bed_top_y_px * mapping.scale_y,
        cab_roof_y_mm=roof_highest[1] * mapping.scale_y,
        neck_y_mm=neck_top[1] * mapping.scale_y,
        hood_front_y_mm=hood_front[1] * mapping.scale_y,
        chassis_y_mm=(chassis_left_y_mm + chassis_right_y_mm) / 2,
        chassis_left_y_mm=chassis_left_y_mm,
        chassis_right_y_mm=chassis_right_y_mm,
        ground_y_mm=height_mm,
        bed_height_mm=bed_height_px * mapping.scale_y,
        cab_height_mm=max(
            0.0,
            (cab_base[1] - roof_highest[1]) * mapping.scale_y,
        ),
        neck_height_mm=max(
            0.0,
            (neck_base[1] - neck_top[1]) * mapping.scale_y,
        ),
        hood_height_mm=max(
            0.0,
            (hood_base[1] - hood_front[1]) * mapping.scale_y,
        ),
        dimension_definitions=definitions,
        region_types={
            "bed": "rectangle" if vehicle_type == "pickup" else "absent",
            "cab": "trapezoid",
            "hood": "trapezoid",
        },
    )


def geometry_from_api_parts(
    parts: dict[str, object],
    mapping: ScaleMapping,
    image_width_px: int,
    height_mm: float,
) -> AnnotationGeometry:
    """Build red-frame geometry from Qwen landmarks, without pixel recognition."""
    if parts.get("schema") == "semantic_structure_v1":
        return geometry_from_api_structure(
            parts,
            mapping,
            image_width_px,
            height_mm,
        )
    vehicle_type = str(parts.get("vehicle_type", ""))
    if vehicle_type not in {"pickup", "non_pickup"}:
        raise RuntimeError("API parts have an invalid vehicle_type")

    def ordered_quad(name: str) -> list[tuple[float, float]]:
        raw = parts.get(name)
        if not isinstance(raw, list) or len(raw) != 4:
            raise RuntimeError(f"API parts are missing {name}")
        points = [(float(point[0]), float(point[1])) for point in raw]
        by_y = sorted(points, key=lambda point: (point[1], point[0]))
        top = sorted(by_y[:2], key=lambda point: point[0])
        bottom = sorted(by_y[2:], key=lambda point: point[0])
        return [top[0], top[1], bottom[1], bottom[0]]

    raw_chassis = parts.get("chassis_line")
    if not isinstance(raw_chassis, list) or len(raw_chassis) != 2:
        raise RuntimeError("API parts are missing chassis_line")
    chassis_points = sorted(
        [(float(point[0]), float(point[1])) for point in raw_chassis],
        key=lambda point: point[0],
    )
    (chassis_x1, chassis_y1), (chassis_x2, chassis_y2) = chassis_points
    if chassis_x2 <= chassis_x1:
        raise RuntimeError("API chassis line has no horizontal span")
    chassis_slope = (chassis_y2 - chassis_y1) / (chassis_x2 - chassis_x1)

    def chassis_at_px(x_px: float) -> float:
        return chassis_y1 + chassis_slope * (x_px - chassis_x1)

    def polygon_segment(
        name: str,
        force_parallel_top: bool = False,
    ) -> tuple[list[tuple[float, float]], float]:
        top_left, top_right, bottom_right, bottom_left = ordered_quad(name)
        if force_parallel_top:
            left_x = min(top_left[0], bottom_left[0])
            right_x = max(top_right[0], bottom_right[0])
            height_left = chassis_at_px(left_x) - top_left[1]
            height_right = chassis_at_px(right_x) - top_right[1]
            part_height = max(1.0, (height_left + height_right) / 2)
            top_left_y = chassis_at_px(left_x) - part_height
            top_right_y = chassis_at_px(right_x) - part_height
            top_left_x = left_x
            top_right_x = right_x
            bottom_left_x = left_x
            bottom_right_x = right_x
        else:
            top_left_x, top_left_y = top_left
            top_right_x, top_right_y = top_right
            bottom_left_x = bottom_left[0]
            bottom_right_x = bottom_right[0]
        pixel_points = [
            (bottom_left_x, chassis_at_px(bottom_left_x)),
            (top_left_x, top_left_y),
            (top_right_x, top_right_y),
            (bottom_right_x, chassis_at_px(bottom_right_x)),
            (bottom_left_x, chassis_at_px(bottom_left_x)),
        ]
        millimetre_points = [
            (x * mapping.scale_x, y * mapping.scale_y)
            for x, y in pixel_points
        ]
        top_y_px = (top_left_y + top_right_y) / 2
        return millimetre_points, top_y_px

    cab_segment, cab_top_y_px = polygon_segment("cab_quad")
    hood_segment, hood_top_y_px = polygon_segment("hood_quad")
    bed_segment = None
    bed_top_y_px = chassis_at_px(0.0)
    if vehicle_type == "pickup":
        bed_segment, bed_top_y_px = polygon_segment(
            "bed_quad",
            force_parallel_top=True,
        )

    segments = [cab_segment, hood_segment]
    if bed_segment is not None:
        segments.append(bed_segment)
    segments.sort(key=lambda segment: min(point[0] for point in segment))

    cab_start_x_mm = min(point[0] for point in cab_segment)
    roof_end_x_mm = max(point[0] for point in cab_segment)
    neck_x_mm = min(point[0] for point in hood_segment)
    front_x_mm = max(point[0] for point in hood_segment)
    cab_y_mm = cab_top_y_px * mapping.scale_y
    bed_y_mm = bed_top_y_px * mapping.scale_y
    hood_y_mm = hood_top_y_px * mapping.scale_y
    hood_quad = ordered_quad("hood_quad")
    neck_y_mm = hood_quad[0][1] * mapping.scale_y

    chassis_left_y_mm = chassis_at_px(0.0) * mapping.scale_y
    chassis_right_y_mm = chassis_at_px(image_width_px - 1) * mapping.scale_y
    chassis_y_mm = (chassis_left_y_mm + chassis_right_y_mm) / 2

    def chassis_at_mm(x_mm: float) -> float:
        fraction = x_mm / max((image_width_px - 1) * mapping.scale_x, 1.0)
        return (
            chassis_left_y_mm
            + (chassis_right_y_mm - chassis_left_y_mm) * fraction
        )

    bed_reference_x = (
        min(point[0] for point in bed_segment)
        if bed_segment is not None
        else 0.0
    )
    return AnnotationGeometry(
        outline_segments_mm=segments,
        is_pickup=vehicle_type == "pickup",
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
        bed_height_mm=(
            max(0.0, chassis_at_mm(bed_reference_x) - bed_y_mm)
            if bed_segment is not None
            else 0.0
        ),
        cab_height_mm=max(0.0, chassis_at_mm(cab_start_x_mm) - cab_y_mm),
        neck_height_mm=max(0.0, chassis_at_mm(neck_x_mm) - neck_y_mm),
        hood_height_mm=max(0.0, chassis_at_mm(front_x_mm) - hood_y_mm),
    )


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
    """Detect the lower fixed-body edge without selecting tyres or trim lines."""
    grayscale = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    height, width = grayscale.shape
    edges = cv2.Canny(grayscale, 30, 120)

    roi = np.zeros_like(edges)
    x_start, x_end = round(width * 0.25), round(width * 0.75)
    y_start = round(height * 0.58)
    y_end = min(height, round(height * 0.89))
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

    # A tight roof-to-tyre crop puts the rocker/lower fixed-body edge close to
    # this band. Fresh Qwen responses provide a semantic chassis hint instead.
    target_y = height * 0.79
    candidates: list[tuple[float, float, float, float, float]] = []
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
        if not height * 0.62 <= midpoint_y <= height * 0.88:
            continue
        midpoint_x = (x1 + x2) / 2
        distance = abs(midpoint_y - target_y) / max(height * 0.08, 1.0)
        # Long tyre/shadow tangents occur lower, while decorative door creases
        # occur higher. Select sustained evidence nearest the rocker band.
        score = length / width - distance * 0.70
        candidates.append(
            (score, slope, midpoint_x, midpoint_y, float(length))
        )
    if not candidates:
        return fallback_y, fallback_y

    _, local_slope, midpoint_x, midpoint_y, _ = max(
        candidates, key=lambda item: item[0]
    )
    nearby = [
        item
        for item in candidates
        if abs(item[3] - midpoint_y) <= height * 0.025
        and abs(item[1] - local_slope) <= 0.035
    ]
    total_weight = sum(item[4] for item in nearby)
    slope = sum(item[1] * item[4] for item in nearby) / total_weight
    slope = float(np.clip(slope, -0.04, 0.04))
    intercept = sum(
        (item[3] - slope * item[2]) * item[4]
        for item in nearby
    ) / total_weight
    left_y = float(np.clip(intercept, height * 0.58, height * 0.88))
    right_y = float(np.clip(
        intercept + slope * (width - 1),
        height * 0.58,
        height * 0.88,
    ))
    return left_y, right_y


def _windshield_edge(
    image: Image.Image,
    cab_roof_y: float,
    maximum_bottom_y: float | None = None,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Fit the real A-pillar/windshield edge on a front-right side view."""
    grayscale = cv2.cvtColor(
        np.asarray(image.convert("RGB")),
        cv2.COLOR_RGB2GRAY,
    )
    height, width = grayscale.shape
    edges = cv2.Canny(grayscale, 40, 130)
    roi = np.zeros_like(edges)
    x_start, x_end = round(width * 0.55), round(width * 0.86)
    y_start, y_end = round(height * 0.02), round(height * 0.62)
    roi[y_start:y_end, x_start:x_end] = edges[
        y_start:y_end,
        x_start:x_end,
    ]
    lines = cv2.HoughLinesP(
        roi,
        rho=1,
        theta=np.pi / 720,
        threshold=max(20, round(width * 0.025)),
        minLineLength=max(25, round(width * 0.04)),
        maxLineGap=max(8, round(width * 0.025)),
    )
    if lines is None:
        return None

    candidates: list[
        tuple[float, tuple[float, float], tuple[float, float]]
    ] = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if x2 < x1:
            x1, x2, y1, y2 = x2, x1, y2, y1
        horizontal_span = x2 - x1
        if horizontal_span <= 0:
            continue
        slope = (y2 - y1) / horizontal_span
        length = float(np.hypot(horizontal_span, y2 - y1))
        if not 0.55 <= slope <= 1.60:
            continue
        if not width * 0.58 <= x1 <= width * 0.75:
            continue
        if not width * 0.72 <= x2 <= width * 0.85:
            continue
        if not y1 <= height * 0.28 or not height * 0.18 <= y2 <= height * 0.60:
            continue
        if maximum_bottom_y is not None and y2 > maximum_bottom_y:
            continue
        if not width * 0.05 <= horizontal_span <= width * 0.19:
            continue
        endpoint_penalty = abs(x2 / width - 0.80) * width * 1.4
        roof_penalty = abs(y1 - cab_roof_y) * 0.35
        score = length - endpoint_penalty - roof_penalty
        candidates.append(
            (
                score,
                (float(x1), float(y1)),
                (float(x2), float(y2)),
            )
        )
    if not candidates:
        return None
    _, top_point, bottom_point = max(candidates, key=lambda item: item[0])
    return top_point, bottom_point


def detect_body_chassis_line(
    image: Image.Image,
    ai_hint_line: object = None,
    pitch_hint_line: object = None,
) -> list[tuple[float, float]]:
    """Select one real lower-body edge with Python, using AI only as a soft hint."""
    grayscale = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    height, width = grayscale.shape
    hint_points: list[tuple[float, float]] = []
    if isinstance(ai_hint_line, list):
        for raw_point in ai_hint_line:
            if isinstance(raw_point, (list, tuple)) and len(raw_point) == 2:
                hint_points.append((float(raw_point[0]), float(raw_point[1])))
    target_y = (
        float(np.median([point[1] for point in hint_points]))
        if len(hint_points) >= 2
        else height * 0.72
    )
    pitch_points: list[tuple[float, float]] = []
    if isinstance(pitch_hint_line, list):
        for raw_point in pitch_hint_line:
            if isinstance(raw_point, (list, tuple)) and len(raw_point) == 2:
                pitch_points.append((float(raw_point[0]), float(raw_point[1])))
    pitch_slope: float | None = None
    if len(pitch_points) >= 2:
        ordered_pitch = sorted(pitch_points, key=lambda point: point[0])
        pitch_span = ordered_pitch[-1][0] - ordered_pitch[0][0]
        if pitch_span > width * 0.20:
            pitch_slope = float(np.clip(
                (ordered_pitch[-1][1] - ordered_pitch[0][1]) / pitch_span,
                -0.015,
                0.015,
            ))

    edges = cv2.Canny(grayscale, 30, 120)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 720,
        threshold=max(20, round(width * 0.04)),
        minLineLength=max(30, round(width * 0.10)),
        maxLineGap=max(4, round(width * 0.035)),
    )
    candidates: list[tuple[float, float, float, float, float]] = []
    if lines is not None:
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            if x1 == x2:
                continue
            if x2 < x1:
                x1, x2, y1, y2 = x2, x1, y2, y1
            horizontal_span = float(x2 - x1)
            slope = float((y2 - y1) / horizontal_span)
            midpoint_x = float((x1 + x2) / 2)
            midpoint_y = float((y1 + y2) / 2)
            if abs(slope) > 0.08:
                continue
            if not height * 0.60 <= midpoint_y <= height * 0.82:
                continue
            distance_penalty = abs(midpoint_y - target_y) / max(
                height * 0.10,
                1.0,
            )
            if len(hint_points) >= 2:
                # The AI hint resolves which of several genuine horizontal
                # edges is the semantic body base. A longer door moulding must
                # not beat a slightly shorter rocker/step edge at the hinted
                # height.
                score = (
                    horizontal_span / width * 0.30
                    - distance_penalty * 1.50
                )
            else:
                score = horizontal_span / width - distance_penalty * 0.65
            candidates.append(
                (score, slope, midpoint_x, midpoint_y, horizontal_span)
            )

    if candidates:
        selected_candidates = candidates
        if len(hint_points) >= 2:
            # Qwen identifies the correct semantic structure band. Within
            # that narrow band choose its lower sustained edge (for example
            # the bottom, not the top, of a dark rocker/running-board strip).
            # This prevents a much longer door moulding above the hint from
            # winning only because of its span.
            hint_band = [
                item
                for item in candidates
                if target_y - height * 0.04
                <= item[3]
                <= target_y + height * 0.06
                and item[4] >= width * 0.28
            ]
            if hint_band:
                selected_candidates = hint_band
                _, local_slope, midpoint_x, midpoint_y, _ = max(
                    selected_candidates,
                    key=lambda item: (item[3], item[4]),
                )
            else:
                _, local_slope, midpoint_x, midpoint_y, _ = max(
                    selected_candidates,
                    key=lambda item: item[0],
                )
        else:
            _, local_slope, midpoint_x, midpoint_y, _ = max(
                selected_candidates,
                key=lambda item: item[0],
            )
        nearby = [
            item
            for item in candidates
            if abs(item[3] - midpoint_y) <= height * 0.018
            and abs(item[1] - local_slope) <= 0.025
        ]
        if nearby:
            total_weight = sum(item[4] for item in nearby)
            detected_slope = (
                sum(item[1] * item[4] for item in nearby) / total_weight
            )
            slope = pitch_slope if pitch_slope is not None else float(
                np.clip(detected_slope, -0.015, 0.015)
            )
            intercept = sum(
                (item[3] - slope * item[2]) * item[4]
                for item in nearby
            ) / total_weight
        else:
            slope = pitch_slope if pitch_slope is not None else float(
                np.clip(local_slope, -0.015, 0.015)
            )
            intercept = midpoint_y - slope * midpoint_x
        left_y = float(np.clip(intercept, height * 0.58, height * 0.84))
        right_y = float(
            np.clip(intercept + slope * (width - 1), height * 0.58, height * 0.84)
        )
        return [(0.0, left_y), (float(width - 1), right_y)]

    if len(hint_points) >= 2:
        ordered = sorted(hint_points, key=lambda point: point[0])
        first, last = ordered[0], ordered[-1]
        span = max(last[0] - first[0], 1.0)
        slope = (
            pitch_slope
            if pitch_slope is not None
            else float(np.clip((last[1] - first[1]) / span, -0.015, 0.015))
        )
        intercept = first[1] - slope * first[0]
        return [
            (0.0, intercept),
            (float(width - 1), intercept + slope * (width - 1)),
        ]
    fallback = height * 0.72
    return [(0.0, fallback), (float(width - 1), fallback)]


def measure_and_trace(
    image: Image.Image,
    mapping: ScaleMapping,
    height_mm: float,
    background_type: str | None = None,
    chassis_hint_line: list[tuple[float, float]] | None = None,
) -> AnnotationGeometry:
    mask = _foreground_mask(image, background_type=background_type)
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
    if chassis_hint_line is not None:
        detected_chassis = detect_body_chassis_line(
            image,
            ai_hint_line=chassis_hint_line,
            # The hint has already passed through perspective correction, so
            # its residual slope is the desired body pitch. This prevents one
            # short rocker fragment from tilting the full-width red baseline.
            pitch_hint_line=chassis_hint_line,
        )
        chassis_left_y_px, chassis_right_y_px = (
            detected_chassis[0][1],
            detected_chassis[-1][1],
        )
    else:
        chassis_left_y_px, chassis_right_y_px = _chassis_line(
            image,
            chassis_y_px,
        )

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
    transition_start = round(width * 0.20)
    transition_end = round(width * 0.58)
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
    windshield_search_start = round(width * 0.55)
    # Stop before the grille/front-profile drop, which is often steeper than
    # the windshield and would otherwise be mistaken for the A-pillar.
    windshield_search_end = round(width * 0.80)
    roof_end_x_px = windshield_search_start + int(
        np.argmax(
            profile_drop[windshield_search_start:windshield_search_end]
        )
    )
    # Windshield base/hood rear follows the start of the sustained A-pillar
    # descent. Keep a small span for the sloped glass while remaining within
    # the normal engine-bay transition zone.
    neck_x_px = max(
        round(width * 0.77),
        min(round(width * 0.82), roof_end_x_px + round(width * 0.03)),
    )
    windshield_edge = _windshield_edge(image, cab_y_px)
    if (
        windshield_edge is not None
        and windshield_edge[1][1] > hood_y_px - max(2.0, height * 0.02)
    ):
        # A windshield base below the hood top makes NECK-H < HOOD-H,
        # which is mechanically implausible for this three-part profile. The
        # first Hough pass often joined the A-pillar to a longer door/hood
        # diagonal, so retry inside the valid topology band.
        windshield_edge = _windshield_edge(
            image,
            cab_y_px,
            maximum_bottom_y=hood_y_px - max(2.0, height * 0.02),
        )
    if windshield_edge is not None:
        windshield_top, windshield_bottom = windshield_edge
        roof_end_x_px = round(windshield_top[0])
        neck_x_px = round(windshield_bottom[0])
        neck_y_px = windshield_bottom[1]
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
    neck_height_mm = max(0.0, neck_chassis_y_mm - neck_y_mm)
    hood_height_mm = max(0.0, hood_chassis_y_mm - hood_y_mm)
    if neck_height_mm < hood_height_mm:
        # Heavy-duty trucks (F-350 DRW, etc.) can have a hood nearly as tall
        # as the windshield base. When the gap is small, clamp NECK-H to
        # HOOD-H instead of failing the whole vehicle.
        shortfall_mm = hood_height_mm - neck_height_mm
        tolerance_mm = max(30.0, height_mm * 0.02)
        if shortfall_mm <= tolerance_mm:
            neck_y_mm = neck_chassis_y_mm - hood_height_mm
            neck_height_mm = hood_height_mm
        else:
            raise RuntimeError(
                "Windshield/NECK retry exhausted: NECK-H remains below HOOD-H"
            )
    outline_segments_mm = [
        [(0.0, bed_chassis_y_mm), (0.0, bed_y_mm), (cab_start_x_mm, bed_y_mm), (cab_start_x_mm, cab_chassis_y_mm), (0.0, bed_chassis_y_mm)],
        [(cab_start_x_mm, cab_chassis_y_mm), (cab_start_x_mm, cab_y_mm), (roof_end_x_mm, cab_y_mm), (neck_x_mm, neck_y_mm), (neck_x_mm, neck_chassis_y_mm), (cab_start_x_mm, cab_chassis_y_mm)],
        [(neck_x_mm, neck_chassis_y_mm), (neck_x_mm, neck_y_mm), (front_x_mm, hood_y_mm), (front_x_mm, hood_chassis_y_mm), (neck_x_mm, neck_chassis_y_mm)],
    ]

    return AnnotationGeometry(
        outline_segments_mm=outline_segments_mm,
        is_pickup=True,
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
        neck_height_mm=neck_height_mm,
        hood_height_mm=hood_height_mm,
    )
