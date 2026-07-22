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
