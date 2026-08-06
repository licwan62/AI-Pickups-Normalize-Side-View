from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Bounds:
    """Vehicle bounds in source-image pixels; right and ground are exclusive."""

    left: int
    right: int
    roof: int
    ground: int

    @property
    def pixel_width(self) -> int:
        return self.right - self.left

    @property
    def pixel_height(self) -> int:
        return self.ground - self.roof

    def as_pillow_box(self) -> tuple[int, int, int, int]:
        return self.left, self.roof, self.right, self.ground

    def validate(self, image_width: int, image_height: int) -> None:
        if not (0 <= self.left < self.right <= image_width):
            raise ValueError(f"Invalid horizontal bounds for image width {image_width}: {self}")
        if not (0 <= self.roof < self.ground <= image_height):
            raise ValueError(f"Invalid vertical bounds for image height {image_height}: {self}")

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"coordinate_system": "source_pixels", "edge_convention": "right_ground_exclusive", **asdict(self)}
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "Bounds":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{key: int(payload[key]) for key in ("left", "right", "roof", "ground")})


def _perspective_transform(
    quad: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ],
    crop_bounds: Bounds | None = None,
) -> tuple[np.ndarray, int, int]:
    source = np.asarray(quad, dtype=np.float32)
    if source.shape != (4, 2) or not np.isfinite(source).all():
        raise ValueError("Perspective quad must contain four finite points")
    top_left, top_right, bottom_right, bottom_left = source
    width = max(
        np.linalg.norm(top_right - top_left),
        np.linalg.norm(bottom_right - bottom_left),
    )
    height = max(
        np.linalg.norm(bottom_left - top_left),
        np.linalg.norm(bottom_right - top_right),
    )
    output_width = max(2, int(round(float(width))))
    output_height = max(2, int(round(float(height))))
    destination = np.asarray([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1],
    ], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source, destination)
    if crop_bounds is not None:
        crop_corners = np.asarray([
            [crop_bounds.left, crop_bounds.roof],
            [crop_bounds.right, crop_bounds.roof],
            [crop_bounds.right, crop_bounds.ground],
            [crop_bounds.left, crop_bounds.ground],
        ], dtype=np.float32).reshape(1, 4, 2)
        mapped_crop = cv2.perspectiveTransform(
            crop_corners,
            transform,
        ).reshape(4, 2)
        minimum_x = float(np.min(mapped_crop[:, 0]))
        maximum_x = float(np.max(mapped_crop[:, 0]))
        minimum_y = float(np.min(mapped_crop[:, 1]))
        maximum_y = float(np.max(mapped_crop[:, 1]))
        translation = np.asarray([
            [1.0, 0.0, -minimum_x],
            [0.0, 1.0, -minimum_y],
            [0.0, 0.0, 1.0],
        ])
        transform = translation @ transform
        output_width = max(2, int(round(maximum_x - minimum_x)))
        output_height = max(2, int(round(maximum_y - minimum_y)))
    return transform, output_width, output_height


def rectify_perspective(
    image: Image.Image,
    quad: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ],
    crop_bounds: Bounds | None = None,
) -> Image.Image:
    """Map top-left, top-right, bottom-right, bottom-left to a rectangle."""
    transform, output_width, output_height = _perspective_transform(
        quad,
        crop_bounds,
    )
    rgb = np.asarray(image.convert("RGB"))
    rectified = cv2.warpPerspective(
        rgb,
        transform,
        (output_width, output_height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return Image.fromarray(rectified, mode="RGB")


def transform_perspective_points(
    points: list[tuple[float, float]],
    quad: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ],
    crop_bounds: Bounds | None = None,
) -> list[tuple[float, float]]:
    """Apply the crop's perspective transform to source-image points."""
    transform, _, _ = _perspective_transform(quad, crop_bounds)
    source = np.asarray(points, dtype=np.float32).reshape(1, -1, 2)
    mapped = cv2.perspectiveTransform(source, transform).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in mapped]


def detect_rectified_vehicle_edges(
    image: Image.Image,
    quad: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ],
    crop_bounds: Bounds,
    max_analysis_dimension: int = 1000,
) -> tuple[int, int, int] | None:
    """Find the vehicle's left, right-exclusive, and top edges after rectification.

    Qwen supplies the semantic search rectangle. GrabCut is only used inside
    that rectangle to snap its three non-ground edges to visible vehicle
    pixels; the ground edge is deliberately left to the wheel-contact logic.
    """
    rgb = np.asarray(image.convert("RGB"))
    source_height, source_width = rgb.shape[:2]
    scale = min(
        1.0,
        max_analysis_dimension / max(source_width, source_height),
    )
    if scale < 1.0:
        analysis_width = max(2, int(round(source_width * scale)))
        analysis_height = max(2, int(round(source_height * scale)))
        analysis = cv2.resize(
            rgb,
            (analysis_width, analysis_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        analysis = rgb
        analysis_height, analysis_width = source_height, source_width

    left = max(1, int(np.floor(crop_bounds.left * scale)))
    top = max(1, int(np.floor(crop_bounds.roof * scale)))
    right = min(
        analysis_width - 1,
        int(np.ceil(crop_bounds.right * scale)),
    )
    bottom = min(
        analysis_height - 1,
        int(np.ceil(crop_bounds.ground * scale)),
    )
    if right - left < 3 or bottom - top < 3:
        return None

    mask = np.zeros((analysis_height, analysis_width), dtype=np.uint8)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)
    try:
        cv2.grabCut(
            cv2.cvtColor(analysis, cv2.COLOR_RGB2BGR),
            mask,
            (left, top, right - left, bottom - top),
            background_model,
            foreground_model,
            3,
            cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error:
        return None

    foreground = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)
    if not np.any(foreground):
        return None

    transform, output_width, output_height = _perspective_transform(
        quad,
        crop_bounds,
    )
    analysis_to_source = np.asarray(
        [
            [1.0 / scale, 0.0, 0.0],
            [0.0, 1.0 / scale, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rectified_mask = cv2.warpPerspective(
        foreground,
        transform @ analysis_to_source,
        (output_width, output_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    minimum_column_pixels = max(3, int(round(output_height * 0.01)))
    minimum_row_pixels = max(3, int(round(output_width * 0.01)))
    columns = np.flatnonzero(
        np.count_nonzero(rectified_mask, axis=0) >= minimum_column_pixels
    )
    rows = np.flatnonzero(
        np.count_nonzero(rectified_mask, axis=1) >= minimum_row_pixels
    )
    if columns.size == 0 or rows.size == 0:
        return None
    return int(columns[0]), int(columns[-1]) + 1, int(rows[0])


def select_bounds(image: Image.Image, window_title: str = "Select vehicle bounds") -> Bounds:
    """Open an ROI selector. Drag bumper-to-bumper and roof-to-ground, then press Enter."""
    rgb = np.asarray(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    try:
        x, y, width, height = cv2.selectROI(window_title, bgr, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(window_title)
    except cv2.error as exc:
        cv2.destroyAllWindows()
        raise RuntimeError("OpenCV could not open the manual selection window") from exc
    if width <= 0 or height <= 0:
        raise RuntimeError("Vehicle selection was cancelled or empty")
    return Bounds(left=int(x), right=int(x + width), roof=int(y), ground=int(y + height))
