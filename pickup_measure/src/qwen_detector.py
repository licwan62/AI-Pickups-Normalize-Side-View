from __future__ import annotations

import base64
from io import BytesIO
import json
import re
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image
import cv2
import numpy as np

from .detector import VehicleDetector
from .geometry import Bounds, transform_perspective_points


class QwenVehicleDetector(VehicleDetector):
    """Vehicle localization through the Qwen vision HTTP API."""

    _LEGACY_PROMPT_TEMPLATE = {
        "bbox_1000": [40.0, 120.0, 960.0, 900.0],
        "perspective_quad_1000": [
            [300.0, 260.0],
            [700.0, 250.0],
            [710.0, 720.0],
            [290.0, 730.0],
        ],
        "wheel_centers_1000": [[250.0, 760.0], [800.0, 750.0]],
    }

    def __init__(
        self,
        *,
        model: str = "qwen3-vl-plus",
        endpoint: str = (
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        ),
        api_key: str = "",
        prompt_file: Path = Path("promting.md"),
        timeout_seconds: float = 90.0,
        max_image_size: int = 1800,
        perspective_correction: bool = False,
    ):
        self.model = model
        self.endpoint = endpoint
        self.api_key = api_key
        self.prompt_file = prompt_file
        self.timeout_seconds = timeout_seconds
        self.max_image_size = max_image_size
        self.perspective_correction = perspective_correction
        self.last_attempt_bounds: Bounds | None = None
        self.last_response_content: str | None = None
        self.last_response_contents: list[str] = []
        self.last_front: str = "unknown"
        self.last_background_type: str = "unknown"
        self.last_perspective_quad: tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ] | None = None
        self.last_image_wheel_centers: list[tuple[float, float]] | None = None
        self.last_image_wheel_contacts: list[tuple[float, float]] | None = None
        self.last_body_chassis_line: list[tuple[float, float]] | None = None
        self.last_boundary_touch_points: dict[
            str, tuple[float, float]
        ] | None = None

    def _load_prompt(self) -> str:
        if not self.prompt_file.is_file():
            raise RuntimeError(f"Qwen prompt file not found: {self.prompt_file}")
        prompt = self.prompt_file.read_text(encoding="utf-8").strip()
        if not prompt:
            raise RuntimeError(f"Qwen prompt file is empty: {self.prompt_file}")
        return prompt

    def _phase_prompt(self, phase: str) -> str:
        """Return the localization-only prompt used by the AI boundary."""
        if phase != "localization":
            raise ValueError(f"Unknown Qwen prompt phase: {phase}")
        return self._load_prompt()

    def _image_data_url(self, image: Image.Image) -> str:
        prepared = image.convert("RGB")
        if max(prepared.size) > self.max_image_size:
            prepared = prepared.copy()
            prepared.thumbnail(
                (self.max_image_size, self.max_image_size),
                Image.Resampling.LANCZOS,
            )
        stream = BytesIO()
        prepared.save(stream, format="JPEG", quality=90, optimize=True)
        encoded = base64.b64encode(stream.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"

    @staticmethod
    def _extract_json(content: str) -> dict[str, object]:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
        decoder = json.JSONDecoder()
        for index, character in enumerate(cleaned):
            if character != "{":
                continue
            try:
                payload, _ = decoder.raw_decode(cleaned[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                return payload
        raise RuntimeError("Qwen did not return a valid JSON object")

    @classmethod
    def _reject_prompt_template_copy(
        cls,
        payload: dict[str, object],
    ) -> None:
        """Reject the old documentation sample being returned as detection."""
        copied_fields = 0
        for field_name, template_value in cls._LEGACY_PROMPT_TEMPLATE.items():
            raw_value = payload.get(field_name)
            try:
                matches = np.allclose(
                    np.asarray(raw_value, dtype=np.float64),
                    np.asarray(template_value, dtype=np.float64),
                    atol=0.01,
                    rtol=0.0,
                )
            except (TypeError, ValueError):
                matches = False
            if matches:
                copied_fields += 1
        if copied_fields >= 2:
            raise RuntimeError(
                "Qwen copied placeholder coordinates instead of inspecting "
                "the vehicle image"
            )

    @staticmethod
    def _bounds_from_payload(
        payload: dict[str, object],
        image_width: int,
        image_height: int,
    ) -> Bounds:
        raw_box = payload.get("bbox_1000")
        if not isinstance(raw_box, list) or len(raw_box) != 4:
            raise RuntimeError("Qwen response is missing bbox_1000")
        try:
            left, top, right, bottom = (float(value) for value in raw_box)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Qwen bbox_1000 contains non-numeric coordinates") from exc
        if not (
            0 <= left < right <= 1000
            and 0 <= top < bottom <= 1000
        ):
            raise RuntimeError(f"Qwen bbox_1000 is invalid: {raw_box}")
        if right - left < 100 or bottom - top < 80:
            raise RuntimeError(f"Qwen vehicle box is implausibly small: {raw_box}")

        bounds = Bounds(
            left=max(0, int(left * image_width // 1000)),
            right=min(image_width, int(-(-right * image_width // 1000))),
            roof=max(0, int(top * image_height // 1000)),
            ground=min(image_height, int(-(-bottom * image_height // 1000))),
        )
        bounds.validate(image_width, image_height)
        if bounds.pixel_width / bounds.pixel_height < 1.2:
            raise RuntimeError(f"Qwen vehicle box aspect is implausible: {bounds}")
        return bounds

    @staticmethod
    def _quad_from_payload(
        payload: dict[str, object],
        image_width: int,
        image_height: int,
    ) -> tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]:
        raw_quad = payload.get("perspective_quad_1000")
        if not isinstance(raw_quad, list) or len(raw_quad) != 4:
            raise RuntimeError("Qwen response is missing perspective_quad_1000")
        points: list[tuple[float, float]] = []
        for raw_point in raw_quad:
            if not isinstance(raw_point, list) or len(raw_point) != 2:
                raise RuntimeError("Qwen perspective quad contains an invalid point")
            try:
                x, y = (float(value) for value in raw_point)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "Qwen perspective quad contains non-numeric coordinates"
                ) from exc
            if not 0 <= x <= 1000 or not 0 <= y <= 1000:
                raise RuntimeError(
                    f"Qwen perspective quad coordinate is out of range: {raw_point}"
                )
            points.append((x * image_width / 1000, y * image_height / 1000))

        polygon = np.asarray(points, dtype=np.float32)
        if not cv2.isContourConvex(polygon.astype(np.int32)):
            raise RuntimeError("Qwen perspective quad is not convex or correctly ordered")
        area = abs(float(cv2.contourArea(polygon)))
        if area < image_width * image_height * 0.02:
            raise RuntimeError("Qwen perspective quad is implausibly small")
        top_width = float(np.linalg.norm(polygon[1] - polygon[0]))
        bottom_width = float(np.linalg.norm(polygon[2] - polygon[3]))
        left_height = float(np.linalg.norm(polygon[3] - polygon[0]))
        right_height = float(np.linalg.norm(polygon[2] - polygon[1]))
        horizontal_ratio = min(top_width, bottom_width) / max(
            top_width,
            bottom_width,
        )
        vertical_ratio = min(left_height, right_height) / max(
            left_height,
            right_height,
        )
        # Catalog side views may need a small keystone correction, but a large
        # width/height change means the model used the sloped vehicle silhouette
        # (roof/hood/bumpers) instead of equal-length mechanical calibration
        # lines. Applying that quad severely distorts an otherwise side-on car.
        if horizontal_ratio < 0.85 or vertical_ratio < 0.75:
            raise RuntimeError(
                "Qwen perspective quad implies implausibly strong keystone; "
                "use matching body lines and vertical panel seams"
            )
        raw_box = payload.get("bbox_1000")
        if isinstance(raw_box, list) and len(raw_box) == 4:
            left, top, right, bottom = (float(value) for value in raw_box)
            bbox_corners = np.asarray([
                [left * image_width / 1000, top * image_height / 1000],
                [right * image_width / 1000, top * image_height / 1000],
                [right * image_width / 1000, bottom * image_height / 1000],
                [left * image_width / 1000, bottom * image_height / 1000],
            ], dtype=np.float32)
            corner_distances = np.linalg.norm(polygon - bbox_corners, axis=1)
            if float(np.max(corner_distances)) < min(image_width, image_height) * 0.02:
                raise RuntimeError(
                    "Qwen perspective quad copied bbox instead of calibration lines"
                )
        return tuple(points)  # type: ignore[return-value]

    @staticmethod
    def _normalized_point(
        raw_point: object,
        image_width: int,
        image_height: int,
        field_name: str,
    ) -> tuple[float, float]:
        if isinstance(raw_point, dict):
            raw_point = [
                raw_point.get("x", raw_point.get("left")),
                raw_point.get("y", raw_point.get("top")),
            ]
        if isinstance(raw_point, tuple):
            raw_point = list(raw_point)
        if not isinstance(raw_point, list) or len(raw_point) != 2:
            raise RuntimeError(f"Qwen {field_name} contains an invalid point")
        try:
            x, y = (float(value) for value in raw_point)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Qwen {field_name} contains non-numeric coordinates"
            ) from exc
        if not 0 <= x <= 1000 or not 0 <= y <= 1000:
            raise RuntimeError(f"Qwen {field_name} coordinate is out of range")
        return x * image_width / 1000, y * image_height / 1000

    @classmethod
    def _part_quad(
        cls,
        raw_quad: object,
        image_width: int,
        image_height: int,
        field_name: str,
    ) -> list[tuple[float, float]]:
        if isinstance(raw_quad, dict):
            if isinstance(raw_quad.get("points"), list):
                raw_quad = raw_quad["points"]
            else:
                raw_quad = [
                    raw_quad.get("top_left", raw_quad.get("tl")),
                    raw_quad.get("top_right", raw_quad.get("tr")),
                    raw_quad.get("bottom_right", raw_quad.get("br")),
                    raw_quad.get("bottom_left", raw_quad.get("bl")),
                ]
        if (
            isinstance(raw_quad, list)
            and len(raw_quad) == 8
            and all(isinstance(value, (int, float)) for value in raw_quad)
        ):
            raw_quad = [
                raw_quad[index:index + 2]
                for index in range(0, 8, 2)
            ]
        if not isinstance(raw_quad, list) or len(raw_quad) != 4:
            raise RuntimeError(f"Qwen response is missing {field_name}")
        points = [
            cls._normalized_point(point, image_width, image_height, field_name)
            for point in raw_quad
        ]
        polygon = np.asarray(points, dtype=np.float32)
        if not cv2.isContourConvex(polygon.astype(np.int32)):
            raise RuntimeError(f"Qwen {field_name} is not convex or correctly ordered")
        if abs(float(cv2.contourArea(polygon))) < image_width * image_height * 0.002:
            raise RuntimeError(f"Qwen {field_name} is implausibly small")
        return points

    @classmethod
    def _parts_from_payload(
        cls,
        payload: dict[str, object],
        image_width: int,
        image_height: int,
    ) -> dict[str, object]:
        nested_parts = payload.get("parts")
        if not isinstance(nested_parts, dict):
            nested_parts = {}

        def part_value(primary: str, *aliases: str) -> object:
            if primary in payload:
                return payload[primary]
            for name in (primary, *aliases):
                if name in nested_parts:
                    return nested_parts[name]
            return None

        vehicle_type = str(
            payload.get(
                "vehicle_type",
                payload.get("type", nested_parts.get("vehicle_type", "")),
            )
        ).lower().replace("-", "_")
        if vehicle_type in {"truck", "pickup_truck"}:
            vehicle_type = "pickup"
        if vehicle_type in {"suv", "car", "sedan", "nonpickup"}:
            vehicle_type = "non_pickup"
        if vehicle_type not in {"pickup", "non_pickup"}:
            raise RuntimeError("Qwen vehicle_type must be pickup or non_pickup")
        cab_quad = cls._part_quad(
            part_value("cab_quad_1000", "cab", "cabin", "cab_quad"),
            image_width,
            image_height,
            "cab_quad_1000",
        )
        hood_quad = cls._part_quad(
            part_value(
                "hood_quad_1000",
                "hood",
                "front",
                "engine_compartment",
                "hood_quad",
            ),
            image_width,
            image_height,
            "hood_quad_1000",
        )
        raw_bed = part_value(
            "bed_quad_1000",
            "bed",
            "cargo_bed",
            "bed_quad",
        )
        if vehicle_type == "pickup":
            bed_quad = cls._part_quad(
                raw_bed,
                image_width,
                image_height,
                "bed_quad_1000",
            )
        else:
            if raw_bed is not None:
                raise RuntimeError("Qwen bed_quad_1000 must be null for non_pickup")
            bed_quad = None

        raw_chassis = part_value(
            "chassis_line_1000",
            "chassis",
            "chassis_line",
        )
        if isinstance(raw_chassis, dict):
            raw_chassis = raw_chassis.get(
                "points",
                [
                    raw_chassis.get("left"),
                    raw_chassis.get("right"),
                ],
            )
        if not isinstance(raw_chassis, list) or len(raw_chassis) != 2:
            raise RuntimeError("Qwen response is missing chassis_line_1000")
        chassis = [
            cls._normalized_point(
                point,
                image_width,
                image_height,
                "chassis_line_1000",
            )
            for point in raw_chassis
        ]
        if chassis[1][0] < chassis[0][0]:
            chassis.reverse()
        horizontal_span = chassis[1][0] - chassis[0][0]
        if horizontal_span < image_width * 0.35:
            raise RuntimeError("Qwen chassis line is implausibly short")
        if abs(chassis[1][1] - chassis[0][1]) > horizontal_span * 0.15:
            raise RuntimeError("Qwen chassis line is implausibly steep")
        raw_wheels = part_value(
            "wheel_centers_1000",
            "wheel_centers",
            "wheels",
        )
        if not isinstance(raw_wheels, list) or len(raw_wheels) != 2:
            raise RuntimeError("Qwen response is missing wheel_centers_1000")
        wheel_centers = [
            cls._normalized_point(
                point,
                image_width,
                image_height,
                "wheel_centers_1000",
            )
            for point in raw_wheels
        ]

        chassis_slope = (
            (chassis[1][1] - chassis[0][1])
            / (chassis[1][0] - chassis[0][0])
        )

        def chassis_y_at(x: float) -> float:
            return chassis[0][1] + chassis_slope * (x - chassis[0][0])

        part_quads = [cab_quad, hood_quad]
        if bed_quad is not None:
            part_quads.append(bed_quad)
        maximum_gap = image_height * 0.06
        for part_quad in part_quads:
            for x, y in (part_quad[2], part_quad[3]):
                if abs(y - chassis_y_at(x)) > maximum_gap:
                    raise RuntimeError(
                        "Qwen part bottom corners do not meet chassis_line_1000"
                    )
        return {
            "vehicle_type": vehicle_type,
            "bed_quad": bed_quad,
            "cab_quad": cab_quad,
            "hood_quad": hood_quad,
            "chassis_line": chassis,
            "wheel_centers": wheel_centers,
        }

    @classmethod
    def _structure_from_payload(
        cls,
        payload: dict[str, object],
        image_width: int,
        image_height: int,
    ) -> dict[str, object]:
        """Parse semantic landmarks and real edge polylines; never infer quads."""
        forbidden = {
            "bed_quad_1000",
            "cab_quad_1000",
            "hood_quad_1000",
            "bed_quad",
            "cab_quad",
            "hood_quad",
        }
        if forbidden.intersection(payload):
            raise RuntimeError(
                "Qwen structure response contains forbidden direct region quads"
            )

        vehicle_type = str(payload.get("vehicle_type", "")).lower().replace("-", "_")
        if vehicle_type in {"truck", "pickup_truck"}:
            vehicle_type = "pickup"
        if vehicle_type in {"suv", "car", "sedan", "nonpickup"}:
            vehicle_type = "non_pickup"
        if vehicle_type not in {"pickup", "non_pickup"}:
            raise RuntimeError("Qwen vehicle_type must be pickup or non_pickup")

        direction = str(
            payload.get("direction", payload.get("front", ""))
        ).lower()
        if direction == "right":
            direction = "front_right"
        elif direction == "left":
            direction = "front_left"
        if direction != "front_right":
            raise RuntimeError(
                "Qwen corrected-crop structure direction must be front_right"
            )

        raw_keypoints = payload.get("keypoints_1000")
        if not isinstance(raw_keypoints, dict):
            raise RuntimeError("Qwen response is missing keypoints_1000")
        keypoint_names = (
            "rear_bumper_outermost",
            "tailgate_top_rear",
            "bed_rail_rear",
            "bed_rail_front",
            "bed_cab_gap_top",
            "bed_cab_gap_bottom",
            "cab_rear_roof",
            "roof_highest",
            "cab_roof_front",
            "windshield_base",
            "hood_rear_top",
            "hood_front_top",
            "grille_front_top",
            "front_bumper_outermost",
            "rocker_rear",
            "rocker_front",
            "rear_tire_contact",
            "front_tire_contact",
            "rear_wheel_center",
            "front_wheel_center",
        )
        keypoints: dict[str, tuple[float, float] | None] = {}
        for name in keypoint_names:
            raw_point = raw_keypoints.get(name)
            keypoints[name] = (
                None
                if raw_point is None
                else cls._normalized_point(
                    raw_point,
                    image_width,
                    image_height,
                    f"keypoints_1000.{name}",
                )
            )

        required_points = {
            "rear_bumper_outermost",
            "cab_rear_roof",
            "roof_highest",
            "cab_roof_front",
            "windshield_base",
            "hood_rear_top",
            "hood_front_top",
            "front_bumper_outermost",
            "rocker_rear",
            "rocker_front",
            "rear_tire_contact",
            "front_tire_contact",
            "rear_wheel_center",
            "front_wheel_center",
        }
        if vehicle_type == "pickup":
            required_points.update({
                "tailgate_top_rear",
                "bed_rail_rear",
                "bed_rail_front",
                "bed_cab_gap_top",
                "bed_cab_gap_bottom",
            })
        missing_points = sorted(
            name for name in required_points if keypoints[name] is None
        )
        if missing_points:
            raise RuntimeError(
                "Qwen semantic keypoints are missing required points: "
                + ", ".join(missing_points)
            )

        raw_polylines = payload.get("polylines_1000")
        if not isinstance(raw_polylines, dict):
            raise RuntimeError("Qwen response is missing polylines_1000")

        def parse_polyline(name: str, required: bool = True) -> list[tuple[float, float]] | None:
            raw_line = raw_polylines.get(name)
            if raw_line is None and not required:
                return None
            if not isinstance(raw_line, list) or len(raw_line) < 2:
                raise RuntimeError(
                    f"Qwen polylines_1000.{name} must contain at least two points"
                )
            line = [
                cls._normalized_point(
                    point,
                    image_width,
                    image_height,
                    f"polylines_1000.{name}",
                )
                for point in raw_line
            ]
            if any(line[index + 1][0] < line[index][0] - image_width * 0.01
                   for index in range(len(line) - 1)):
                raise RuntimeError(
                    f"Qwen polylines_1000.{name} is not ordered rear-to-front"
                )
            return line

        polylines = {
            "bed_top_line": parse_polyline(
                "bed_top_line",
                required=vehicle_type == "pickup",
            ),
            "cab_roof_line": parse_polyline("cab_roof_line"),
            "cab_rear_line": parse_polyline("cab_rear_line"),
            "windshield_line": parse_polyline("windshield_line"),
            "hood_top_line": parse_polyline("hood_top_line"),
            "front_profile_line": parse_polyline("front_profile_line"),
            "rocker_line": parse_polyline("rocker_line"),
            "vehicle_lower_body_line": parse_polyline("vehicle_lower_body_line"),
            "ground_line": parse_polyline("ground_line"),
            "wheel_center_line": parse_polyline("wheel_center_line"),
        }

        rear_wheel = keypoints["rear_wheel_center"]
        front_wheel = keypoints["front_wheel_center"]
        rear_contact = keypoints["rear_tire_contact"]
        front_contact = keypoints["front_tire_contact"]
        roof_front = keypoints["cab_roof_front"]
        windshield_base = keypoints["windshield_base"]
        hood_rear = keypoints["hood_rear_top"]
        hood_front = keypoints["hood_front_top"]
        rear_bumper = keypoints["rear_bumper_outermost"]
        front_bumper = keypoints["front_bumper_outermost"]
        assert rear_wheel and front_wheel and rear_contact and front_contact
        assert roof_front and windshield_base and hood_rear and hood_front
        assert rear_bumper and front_bumper
        if not (
            rear_wheel[0] < front_wheel[0]
            and roof_front[0] < windshield_base[0] <= hood_front[0]
            and hood_rear[0] <= hood_front[0] <= front_bumper[0] + image_width * 0.03
        ):
            raise RuntimeError("Qwen semantic keypoints violate front-right topology")
        if abs(rear_contact[1] - front_contact[1]) > image_height * 0.04:
            raise RuntimeError("Qwen ground contact points are not level")
        if vehicle_type == "pickup":
            bed_rear = keypoints["bed_rail_rear"]
            bed_front = keypoints["bed_rail_front"]
            gap_top = keypoints["bed_cab_gap_top"]
            cab_rear = keypoints["cab_rear_roof"]
            assert bed_rear and bed_front and gap_top and cab_rear
            if not (
                bed_rear[0] < bed_front[0]
                and abs(bed_front[0] - gap_top[0]) <= image_width * 0.08
                and gap_top[0] <= cab_rear[0] + image_width * 0.05
                and cab_rear[0] < roof_front[0]
            ):
                raise RuntimeError("Qwen pickup keypoints violate BED/CAB topology")

        def require_anchors(
            line_name: str,
            start: tuple[float, float],
            end: tuple[float, float],
        ) -> None:
            polyline = polylines[line_name]
            assert polyline is not None
            tolerance_x = image_width * 0.04
            tolerance_y = image_height * 0.08

            def close(actual: tuple[float, float], expected: tuple[float, float]) -> bool:
                return (
                    abs(actual[0] - expected[0]) <= tolerance_x
                    and abs(actual[1] - expected[1]) <= tolerance_y
                )

            if not close(polyline[0], start) or not close(polyline[-1], end):
                raise RuntimeError(
                    f"Qwen polylines_1000.{line_name} endpoints do not match "
                    "their semantic keypoint anchors"
                )

        cab_rear_roof = keypoints["cab_rear_roof"]
        rocker_rear = keypoints["rocker_rear"]
        rocker_front = keypoints["rocker_front"]
        assert cab_rear_roof and rocker_rear and rocker_front
        require_anchors("cab_roof_line", cab_rear_roof, roof_front)
        require_anchors(
            "cab_rear_line",
            keypoints["bed_cab_gap_bottom"] or rocker_rear,
            cab_rear_roof,
        )
        require_anchors("windshield_line", roof_front, windshield_base)
        require_anchors("hood_top_line", hood_rear, hood_front)
        require_anchors(
            "rocker_line",
            rocker_rear,
            rocker_front,
        )
        if vehicle_type == "pickup":
            require_anchors(
                "bed_top_line",
                keypoints["bed_rail_rear"],  # type: ignore[arg-type]
                keypoints["bed_rail_front"],  # type: ignore[arg-type]
            )

        return {
            "schema": "semantic_structure_v1",
            "vehicle_type": vehicle_type,
            "direction": direction,
            "keypoints": keypoints,
            "polylines": polylines,
        }

    @classmethod
    def _wheel_centers_from_payload(
        cls,
        payload: dict[str, object],
        image_width: int,
        image_height: int,
    ) -> list[tuple[float, float]]:
        nested_parts = payload.get("parts")
        if not isinstance(nested_parts, dict):
            nested_parts = {}
        raw_wheels = payload.get(
            "wheel_centers_1000",
            nested_parts.get("wheel_centers_1000", nested_parts.get("wheel_centers")),
        )
        if not isinstance(raw_wheels, list) or len(raw_wheels) != 2:
            raise RuntimeError("Qwen response is missing wheel_centers_1000")
        return [
            cls._normalized_point(
                point,
                image_width,
                image_height,
                "wheel_centers_1000",
            )
            for point in raw_wheels
        ]

    @classmethod
    def _wheel_contacts_from_payload(
        cls,
        payload: dict[str, object],
        image_width: int,
        image_height: int,
    ) -> list[tuple[float, float]] | None:
        raw_contacts = payload.get("wheel_contact_points_1000")
        if raw_contacts is None:
            return None
        if not isinstance(raw_contacts, list) or len(raw_contacts) != 2:
            raise RuntimeError(
                "Qwen wheel_contact_points_1000 must contain two points"
            )
        return [
            cls._normalized_point(
                point,
                image_width,
                image_height,
                "wheel_contact_points_1000",
            )
            for point in raw_contacts
        ]

    @classmethod
    def _body_chassis_line_from_payload(
        cls,
        payload: dict[str, object],
        image_width: int,
        image_height: int,
        *,
        required: bool = False,
    ) -> list[tuple[float, float]] | None:
        raw_line = payload.get("body_chassis_line_1000")
        if raw_line is None:
            if required:
                raise RuntimeError(
                    "Qwen response is missing body_chassis_line_1000"
                )
            return None
        if not isinstance(raw_line, list) or len(raw_line) != 2:
            raise RuntimeError(
                "Qwen body_chassis_line_1000 must contain two points"
            )
        points = [
            cls._normalized_point(
                point,
                image_width,
                image_height,
                "body_chassis_line_1000",
            )
            for point in raw_line
        ]
        points.sort(key=lambda point: point[0])
        horizontal_span = points[1][0] - points[0][0]
        if horizontal_span < image_width * 0.35:
            raise RuntimeError(
                "Qwen body_chassis_line_1000 is implausibly short"
            )
        if abs(points[1][1] - points[0][1]) > horizontal_span * 0.12:
            raise RuntimeError(
                "Qwen body_chassis_line_1000 is implausibly steep"
            )
        return points

    @classmethod
    def _boundary_touch_points_from_payload(
        cls,
        payload: dict[str, object],
        image_width: int,
        image_height: int,
        bounds: Bounds,
    ) -> dict[str, tuple[float, float]] | None:
        """Parse exact fixed-vehicle contacts for a zero-margin crop."""
        raw_points = payload.get("boundary_touch_points_1000")
        if raw_points is None:
            # Backward compatibility for saved detections made before this
            # field existed. New prompt responses always include it.
            return None
        if not isinstance(raw_points, dict):
            raise RuntimeError(
                "Qwen boundary_touch_points_1000 must be an object"
            )
        parsed: dict[str, tuple[float, float]] = {}
        for name in ("leftmost", "rightmost", "topmost"):
            parsed[name] = cls._normalized_point(
                raw_points.get(name),
                image_width,
                image_height,
                f"boundary_touch_points_1000.{name}",
            )

        horizontal_tolerance = max(3.0, bounds.pixel_width * 0.06)
        vertical_tolerance = max(3.0, bounds.pixel_height * 0.08)
        if abs(parsed["leftmost"][0] - bounds.left) > horizontal_tolerance:
            raise RuntimeError(
                "Qwen leftmost touch point is inconsistent with bbox_1000"
            )
        if abs(parsed["rightmost"][0] - bounds.right) > horizontal_tolerance:
            raise RuntimeError(
                "Qwen rightmost touch point is inconsistent with bbox_1000"
            )
        if abs(parsed["topmost"][1] - bounds.roof) > vertical_tolerance:
            raise RuntimeError(
                "Qwen topmost touch point is inconsistent with bbox_1000"
            )
        return parsed

    @staticmethod
    def _trim_antenna_from_bounds(
        image: Image.Image,
        bounds: Bounds,
    ) -> Bounds:
        """Detect and remove a thin antenna protrusion from the top of the bbox.

        Scans the top portion of the bounding box row-by-row. If the uppermost
        rows contain only a narrow vertical strip (antenna) and then suddenly
        widen into the vehicle body, the roof is moved down to the body top.
        Returns the original bounds unchanged when no antenna is detected.
        """
        rgb = np.asarray(image.convert("RGB"))
        roi = rgb[bounds.roof:bounds.ground, bounds.left:bounds.right]
        if roi.size == 0:
            return bounds
        gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape

        # Use Otsu's thresholding to separate foreground from background
        # without relying on corner sampling (which can be contaminated by
        # the vehicle body when the bbox is tight).
        _, fg_mask = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        # For each row, measure the horizontal span of foreground pixels.
        row_spans: list[int] = []
        for row_idx in range(height):
            cols = np.flatnonzero(fg_mask[row_idx])
            if cols.size == 0:
                row_spans.append(0)
            else:
                row_spans.append(int(cols[-1] - cols[0] + 1))

        # Find the first row where the span is "wide" (vehicle body).
        # A row is considered wide if its span exceeds a fraction of the bbox
        # width. The antenna is narrow; the body is wide.
        body_width_threshold = max(20, int(width * 0.15))
        # Require several consecutive wide rows to avoid noise.
        min_consecutive = max(3, int(height * 0.02))

        body_top_rel = None
        consecutive = 0
        for row_idx, span in enumerate(row_spans):
            if span >= body_width_threshold:
                consecutive += 1
                if consecutive >= min_consecutive:
                    body_top_rel = row_idx - consecutive + 1
                    break
            else:
                consecutive = 0

        if body_top_rel is None:
            return bounds

        # Only trim if the antenna region is small relative to the bbox height
        # (avoids false positives when the bbox is already tight).
        antenna_height = body_top_rel
        bbox_height = bounds.ground - bounds.roof
        if antenna_height < max(3, int(bbox_height * 0.03)):
            return bounds
        if antenna_height > int(bbox_height * 0.25):
            return bounds

        return Bounds(
            left=bounds.left,
            right=bounds.right,
            roof=bounds.roof + body_top_rel,
            ground=bounds.ground,
        )

    @staticmethod
    def _image_wheel_centers(
        image: Image.Image,
        bounds: Bounds,
        expected_centers: list[tuple[float, float]],
    ) -> list[tuple[float, float]] | None:
        """Lock onto two real circular wheel/rim centers near Qwen's x hints."""
        rgb = np.asarray(image.convert("RGB"))
        roi = cv2.cvtColor(
            rgb[bounds.roof:bounds.ground, bounds.left:bounds.right],
            cv2.COLOR_RGB2GRAY,
        )
        height, width = roi.shape
        blurred = cv2.GaussianBlur(roi, (7, 7), 1.5)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=width * 0.22,
            param1=100,
            param2=32,
            minRadius=max(10, round(height * 0.04)),
            maxRadius=max(20, round(height * 0.18)),
        )
        if circles is None:
            return None
        candidates = [
            (
                float(center_x + bounds.left),
                float(center_y + bounds.roof),
                float(radius),
            )
            for center_x, center_y, radius in np.round(circles[0]).astype(int)
            if center_y >= height * 0.50
        ]
        expected = sorted(expected_centers, key=lambda point: point[0])
        pairs: list[
            tuple[
                float,
                tuple[float, float, float],
                tuple[float, float, float],
            ]
        ] = []
        for index, first in enumerate(candidates):
            for second in candidates[index + 1:]:
                left, right = sorted((first, second), key=lambda circle: circle[0])
                separation = right[0] - left[0]
                largest_radius = max(left[2], right[2])
                if not bounds.pixel_width * 0.35 <= separation <= bounds.pixel_width * 0.85:
                    continue
                if abs(left[2] - right[2]) > largest_radius * 0.50:
                    continue
                if abs(left[1] - right[1]) > largest_radius * 0.80:
                    continue
                hint_error = (
                    abs(left[0] - expected[0][0])
                    + abs(right[0] - expected[1][0])
                )
                score = (
                    separation
                    - hint_error * 2.0
                    - abs(left[2] - right[2]) * 2.0
                    - abs(left[1] - right[1])
                )
                pairs.append((score, left, right))
        if not pairs:
            return None
        _, left, right = max(pairs, key=lambda item: item[0])
        return [(left[0], left[1]), (right[0], right[1])]

    @staticmethod
    def _image_wheel_contacts(
        image: Image.Image,
        bounds: Bounds,
        wheel_centers: list[tuple[float, float]],
        expected_contacts: list[tuple[float, float]] | None = None,
    ) -> list[tuple[float, float]] | None:
        """Snap each lowest tyre point to the outer circular edge."""
        grayscale = cv2.cvtColor(
            np.asarray(image.convert("RGB")),
            cv2.COLOR_RGB2GRAY,
        )
        blurred = cv2.GaussianBlur(grayscale, (5, 5), 1.0)
        gradient = cv2.magnitude(
            cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3),
        )
        box_height = bounds.pixel_height
        angles = np.linspace(
            np.deg2rad(35.0),
            np.deg2rad(145.0),
            111,
        )
        centers = sorted(wheel_centers, key=lambda point: point[0])
        hints = (
            sorted(expected_contacts, key=lambda point: point[0])
            if expected_contacts is not None
            else [None, None]
        )
        contacts: list[tuple[float, float]] = []
        for (center_x, center_y), hint in zip(centers, hints):
            minimum_radius = max(8, round(box_height * 0.15))
            maximum_radius = min(
                round(box_height * 0.24),
                round(bounds.ground - center_y + box_height * 0.03),
            )
            if maximum_radius <= minimum_radius:
                return None
            candidates: list[tuple[float, int]] = []
            for radius in range(minimum_radius, maximum_radius + 1):
                xs = np.rint(
                    center_x + radius * np.cos(angles)
                ).astype(np.int32)
                ys = np.rint(
                    center_y + radius * np.sin(angles)
                ).astype(np.int32)
                valid = (
                    (xs >= 0)
                    & (xs < grayscale.shape[1])
                    & (ys >= 0)
                    & (ys < grayscale.shape[0])
                )
                if np.count_nonzero(valid) < 30:
                    continue
                score = float(np.mean(gradient[ys[valid], xs[valid]]))
                candidates.append((score, radius))
            if not candidates:
                return None

            outer_candidates = [
                item
                for item in candidates
                if item[1] >= maximum_radius * 0.72
            ]
            if hint is not None:
                hinted_radius = max(1.0, hint[1] - center_y)
                search_radius = max(4.0, box_height * 0.06)
                near_hint = [
                    item
                    for item in outer_candidates
                    if abs(item[1] - hinted_radius) <= search_radius
                ]
                if near_hint:
                    outer_candidates = near_hint
            # Select the strongest coherent outer arc. Choosing the outermost
            # merely acceptable radius can walk off the tyre onto a road,
            # shadow, or bbox boundary and invent a false perspective taper.
            _, radius = max(outer_candidates, key=lambda item: item[0])
            contact_y = min(
                float(bounds.ground),
                center_y + float(radius),
            )
            contacts.append((center_x, contact_y))
        return contacts

    @staticmethod
    def _align_quad_to_wheel_axis(
        quad: tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ],
        wheel_centers: list[tuple[float, float]],
        wheel_contacts: list[tuple[float, float]] | None = None,
    ) -> tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]:
        """Make wheel-center and tyre-contact lines exact rectification axes."""
        wheels = sorted(wheel_centers, key=lambda point: point[0])
        wheel_span = wheels[1][0] - wheels[0][0]
        if wheel_span <= 0:
            return quad
        center_slope = float(np.clip(
            (wheels[1][1] - wheels[0][1]) / wheel_span,
            -0.08,
            0.08,
        ))
        contact_slope = center_slope
        if wheel_contacts is not None:
            contacts = sorted(wheel_contacts, key=lambda point: point[0])
            contact_span = contacts[1][0] - contacts[0][0]
            if contact_span > 0:
                contact_slope = float(np.clip(
                    (contacts[1][1] - contacts[0][1]) / contact_span,
                    -0.10,
                    0.10,
                ))

        center_intercept = wheels[0][1] - center_slope * wheels[0][0]
        if wheel_contacts is not None:
            contacts = sorted(wheel_contacts, key=lambda point: point[0])
            contact_intercept = (
                contacts[0][1] - contact_slope * contacts[0][0]
            )
        else:
            # Preserve the original quad height when no tyre-edge estimate is
            # available, while still using the wheel plane as the upper axis.
            original_height = (
                abs(quad[3][1] - quad[0][1])
                + abs(quad[2][1] - quad[1][1])
            ) / 2
            contact_intercept = center_intercept + original_height

        def side_intersection(
            upper: tuple[float, float],
            lower: tuple[float, float],
            slope: float,
            intercept: float,
        ) -> tuple[float, float] | None:
            delta_x = lower[0] - upper[0]
            delta_y = lower[1] - upper[1]
            denominator = delta_y - slope * delta_x
            if abs(denominator) < 1e-6:
                return None
            fraction = (
                slope * upper[0] + intercept - upper[1]
            ) / denominator
            x = upper[0] + fraction * delta_x
            return x, slope * x + intercept

        top_left = side_intersection(
            quad[0], quad[3], center_slope, center_intercept
        )
        top_right = side_intersection(
            quad[1], quad[2], center_slope, center_intercept
        )
        bottom_left = side_intersection(
            quad[0], quad[3], contact_slope, contact_intercept
        )
        bottom_right = side_intersection(
            quad[1], quad[2], contact_slope, contact_intercept
        )
        if None in {top_left, top_right, bottom_left, bottom_right}:
            return quad
        assert top_left is not None
        assert top_right is not None
        assert bottom_left is not None
        assert bottom_right is not None
        if (
            top_right[0] <= top_left[0]
            or bottom_right[0] <= bottom_left[0]
            or min(
                bottom_left[1] - top_left[1],
                bottom_right[1] - top_right[1],
            ) <= 3.0
        ):
            return quad
        return top_left, top_right, bottom_right, bottom_left

    def _request_content(
        self,
        image: Image.Image,
        prompt: str,
        previous_content: str | None = None,
        validation_error: str | None = None,
    ) -> str:
        messages: list[dict[str, object]] = [{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": self._image_data_url(image)},
                },
                {"type": "text", "text": prompt},
            ],
        }]
        if previous_content is not None and validation_error is not None:
            messages.extend([
                {"role": "assistant", "content": previous_content},
                {
                    "role": "user",
                    "content": (
                        "Your previous JSON failed validation with this error: "
                        f"{validation_error}\n"
                        "Inspect the image again and return a corrected, complete JSON "
                        "object. Include every required key and use arrays [x,y] for "
                        "all points. Return JSON only."
                    ),
                },
            ])
        request_payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 800,
        }
        request = Request(
            self.endpoint,
            data=json.dumps(request_payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key.strip()}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Qwen API HTTP {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"Qwen API connection failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError("Qwen API request timed out") from exc
        except json.JSONDecodeError as exc:
            raise RuntimeError("Qwen API returned invalid response JSON") from exc

        try:
            content = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError("Qwen API response has no assistant content") from exc
        if not isinstance(content, str):
            raise RuntimeError("Qwen API assistant content is not text")
        return content

    def detect(
        self,
        image: Image.Image,
        expected_aspect: float | None = None,
    ) -> Bounds:
        del expected_aspect  # Qwen localizes visible pixels; QC validates proportions later.
        self.last_attempt_bounds = None
        self.last_response_content = None
        self.last_response_contents = []
        self.last_front = "unknown"
        self.last_background_type = "unknown"
        self.last_perspective_quad = None
        self.last_image_wheel_centers = None
        self.last_image_wheel_contacts = None
        self.last_body_chassis_line = None
        self.last_boundary_touch_points = None
        if not self.api_key.strip():
            raise RuntimeError(
                "Qwen API key is missing; set detection.api_key in config.yaml"
            )
        prompt = self._phase_prompt("localization")
        previous_content = None
        validation_error = None

        for attempt in range(2):
            content = self._request_content(
                image,
                prompt,
                previous_content=previous_content,
                validation_error=validation_error,
            )
            self.last_response_content = content
            self.last_response_contents.append(content)
            try:
                result = self._extract_json(content)
                self._reject_prompt_template_copy(result)
                front = str(result.get("front", "unknown")).lower()
                if front in {"left", "right", "unknown"}:
                    self.last_front = front
                background_type = str(
                    result.get("background_type", "unknown")
                ).lower()
                if background_type in {
                    "white",
                    "transparent",
                    "environment",
                }:
                    self.last_background_type = background_type
                bounds = self._bounds_from_payload(
                    result,
                    image.width,
                    image.height,
                )
                trimmed_bounds = self._trim_antenna_from_bounds(image, bounds)
                antenna_trimmed = trimmed_bounds.roof > bounds.roof
                bounds = trimmed_bounds
                self.last_attempt_bounds = bounds
                try:
                    boundary_touch_points = self._boundary_touch_points_from_payload(
                        result,
                        image.width,
                        image.height,
                        bounds,
                    )
                except RuntimeError:
                    if antenna_trimmed:
                        # The topmost point was the antenna tip; re-parse with
                        # the trimmed bounds and override topmost to the new roof.
                        raw_points = result.get("boundary_touch_points_1000")
                        if isinstance(raw_points, dict):
                            boundary_touch_points = {
                                "leftmost": self._normalized_point(
                                    raw_points.get("leftmost"),
                                    image.width,
                                    image.height,
                                    "boundary_touch_points_1000.leftmost",
                                ),
                                "rightmost": self._normalized_point(
                                    raw_points.get("rightmost"),
                                    image.width,
                                    image.height,
                                    "boundary_touch_points_1000.rightmost",
                                ),
                                "topmost": (float(bounds.left + bounds.right) / 2, float(bounds.roof)),
                            }
                        else:
                            raise
                    else:
                        raise
                quad = None
                wheel_contacts = None
                image_wheel_centers_result = None
                if self.perspective_correction:
                    quad = self._quad_from_payload(
                        result,
                        image.width,
                        image.height,
                    )
                wheel_centers = self._wheel_centers_from_payload(
                    result,
                    image.width,
                    image.height,
                )
                qwen_wheel_contacts = self._wheel_contacts_from_payload(
                    result,
                    image.width,
                    image.height,
                )
                body_chassis_line = self._body_chassis_line_from_payload(
                    result,
                    image.width,
                    image.height,
                )
                if body_chassis_line is not None:
                    ordered_wheels = sorted(
                        wheel_centers,
                        key=lambda point: point[0],
                    )
                    source_wheel_span = (
                        ordered_wheels[1][0] - ordered_wheels[0][0]
                    )
                    endpoint_tolerance = source_wheel_span * 0.04
                    if (
                        body_chassis_line[0][0]
                        < ordered_wheels[0][0] - endpoint_tolerance
                        or body_chassis_line[1][0]
                        > ordered_wheels[1][0] + endpoint_tolerance
                    ):
                        raise RuntimeError(
                            "Qwen body_chassis_line_1000 must stay between "
                            "the two wheel centers"
                        )
                    wheel_center_y = float(np.mean(
                        [point[1] for point in ordered_wheels]
                    ))
                    chassis_y = float(np.mean(
                        [point[1] for point in body_chassis_line]
                    ))
                    if chassis_y < wheel_center_y - bounds.pixel_height * 0.025:
                        raise RuntimeError(
                            "Qwen body_chassis_line_1000 is too high; use "
                            "the lower rocker, sill, or fixed side step"
                        )
                if self.perspective_correction:
                    image_wheel_centers_result = self._image_wheel_centers(
                        image,
                        bounds,
                        wheel_centers,
                    )
                    if image_wheel_centers_result is not None:
                        image_wheel_contacts = self._image_wheel_contacts(
                            image,
                            bounds,
                            image_wheel_centers_result,
                            qwen_wheel_contacts,
                        )
                        quad = self._align_quad_to_wheel_axis(
                            quad,
                            image_wheel_centers_result,
                            image_wheel_contacts,
                        )
                        wheel_centers = image_wheel_centers_result
                        wheel_contacts = (
                            image_wheel_contacts
                            if image_wheel_contacts is not None
                            else qwen_wheel_contacts
                        )
                    else:
                        image_wheel_contacts = None
                        wheel_contacts = qwen_wheel_contacts
                    mapped_wheels = transform_perspective_points(
                        wheel_centers,
                        quad,
                        bounds,
                    )
                    wheel_span = abs(mapped_wheels[1][0] - mapped_wheels[0][0])
                    if (
                        wheel_span <= 0
                        or abs(mapped_wheels[1][1] - mapped_wheels[0][1])
                        > max(2.0, wheel_span * 0.02)
                    ):
                        raise RuntimeError(
                            "Qwen perspective calibration does not level wheel centers"
                        )
                    if wheel_contacts is not None:
                        mapped_contacts = transform_perspective_points(
                            wheel_contacts,
                            quad,
                            bounds,
                        )
                        contact_span = abs(
                            mapped_contacts[1][0] - mapped_contacts[0][0]
                        )
                        if (
                            contact_span <= 0
                            or abs(
                                mapped_contacts[1][1] - mapped_contacts[0][1]
                            )
                            > max(2.0, contact_span * 0.015)
                        ):
                            raise RuntimeError(
                                "Perspective calibration does not level both tyre "
                                "contact points"
                            )
            except RuntimeError as exc:
                if attempt == 0:
                    previous_content = content
                    validation_error = str(exc)
                    continue
                raise RuntimeError(
                    f"Qwen response validation failed after repair: {exc}"
                ) from exc
            self.last_perspective_quad = quad
            self.last_image_wheel_centers = image_wheel_centers_result
            self.last_image_wheel_contacts = (
                image_wheel_contacts if self.perspective_correction else None
            )
            self.last_body_chassis_line = body_chassis_line
            self.last_boundary_touch_points = boundary_touch_points
            return bounds

        raise RuntimeError("Qwen response validation failed")
