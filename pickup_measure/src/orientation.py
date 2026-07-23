from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image, ImageOps

from .annotation import _foreground_mask


@dataclass(frozen=True)
class OrientationResult:
    detected_front: str
    normalized_front: str
    flipped: bool
    confidence: float
    direction_score: float
    profile_score: float
    taillight_score: float
    mirror_edge_score: float

    def payload(self) -> dict[str, object]:
        return asdict(self)


def _profile_score(image: Image.Image) -> float:
    """Return a signed upper-profile score; positive means the front is left."""
    mask = _foreground_mask(image)
    height, width = mask.shape
    upper_limit = max(1, round(height * 0.75))
    top_rows = np.full(width, np.nan, dtype=np.float32)
    for x in range(width):
        rows = np.flatnonzero(mask[:upper_limit, x])
        if rows.size:
            top_rows[x] = float(rows[0])
    valid = ~np.isnan(top_rows)
    if np.count_nonzero(valid) < width * 0.4:
        return 0.0
    x_axis = np.arange(width)
    profile = np.interp(x_axis, x_axis[valid], top_rows[valid])
    kernel_width = max(5, round(width * 0.025))
    if kernel_width % 2 == 0:
        kernel_width += 1
    profile = cv2.GaussianBlur(
        profile.reshape(1, -1),
        (kernel_width, 1),
        0,
    ).reshape(-1)
    left_top = float(np.median(profile[round(width * 0.04):round(width * 0.18)]))
    right_top = float(np.median(profile[round(width * 0.82):round(width * 0.96)]))
    # A selected photographic background can touch both upper corners. One
    # vehicle end touching the crop is still a useful direction cue.
    if left_top <= 1 and right_top <= 1:
        return 0.0
    return float(np.clip((left_top - right_top) / max(height * 0.15, 1), -1, 1))


def _taillight_score(rgb: np.ndarray) -> float:
    """Use saturated red lamps as a tail cue; positive means the tail is right."""
    height, width = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    red = (
        ((hsv[:, :, 0] < 12) | (hsv[:, :, 0] > 170))
        & (hsv[:, :, 1] > 110)
        & (hsv[:, :, 2] > 70)
    )
    y_start, y_end = round(height * 0.08), round(height * 0.72)
    side_width = max(1, round(width * 0.22))
    left_red = int(np.count_nonzero(red[y_start:y_end, :side_width]))
    right_red = int(np.count_nonzero(red[y_start:y_end, width - side_width:]))
    total = left_red + right_red
    reliability = min(1.0, total / max(height * width * 0.0005, 1))
    return float((right_red - left_red) / (total + 1) * reliability)


def _mirror_edge_score(rgb: np.ndarray) -> float:
    """Compare A-pillar/mirror-region edge energy on the two cabin sides."""
    height, width = rgb.shape[:2]
    grayscale = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(grayscale, 60, 150)
    y_start, y_end = round(height * 0.18), round(height * 0.55)
    left_edges = int(np.count_nonzero(
        edges[y_start:y_end, round(width * 0.12):round(width * 0.40)]
    ))
    right_edges = int(np.count_nonzero(
        edges[y_start:y_end, round(width * 0.60):round(width * 0.88)]
    ))
    return float((left_edges - right_edges) / (left_edges + right_edges + 1))


def normalize_front_to_right(
    image: Image.Image,
    minimum_confidence: float = 0.05,
) -> tuple[Image.Image, OrientationResult]:
    """Detect the vehicle front and mirror left-facing vehicles."""
    rgb = np.asarray(image.convert("RGB"))
    try:
        profile_score = _profile_score(image)
    except RuntimeError:
        profile_score = 0.0
    taillight_score = _taillight_score(rgb)
    mirror_edge_score = _mirror_edge_score(rgb)
    direction_score = (
        0.65 * profile_score
        + 0.30 * taillight_score
        + 0.05 * mirror_edge_score
    )
    confidence = min(1.0, abs(direction_score))

    if confidence < minimum_confidence:
        detected_front = "unknown"
        flipped = False
    else:
        detected_front = "left" if direction_score > 0 else "right"
        flipped = detected_front == "left"
    normalized = ImageOps.mirror(image) if flipped else image.copy()
    result = OrientationResult(
        detected_front=detected_front,
        normalized_front="right" if detected_front != "unknown" else "unknown",
        flipped=flipped,
        confidence=round(confidence, 4),
        direction_score=round(float(direction_score), 4),
        profile_score=round(profile_score, 4),
        taillight_score=round(taillight_score, 4),
        mirror_edge_score=round(mirror_edge_score, 4),
    )
    return normalized, result
