from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScaleMapping:
    scale_x: float
    scale_y: float

    @classmethod
    def from_dimensions(
        cls, pixel_width: int, pixel_height: int, length_mm: float, height_mm: float
    ) -> "ScaleMapping":
        if pixel_width <= 0 or pixel_height <= 0:
            raise ValueError("Pixel dimensions must be positive")
        if length_mm <= 0 or height_mm <= 0:
            raise ValueError("Real dimensions must be positive")
        return cls(scale_x=length_mm / pixel_width, scale_y=height_mm / pixel_height)

    def pixel_to_mm(self, x_pixel: float, y_pixel: float) -> tuple[float, float]:
        return x_pixel * self.scale_x, y_pixel * self.scale_y


def millimetres_to_pixels(millimetres: float, ppi: float) -> int:
    if millimetres <= 0 or ppi <= 0:
        raise ValueError("Millimetres and PPI must be positive")
    return max(1, round(millimetres / 25.4 * ppi))


def png_effective_ppi(requested_ppi: float) -> float:
    """Return the nearest PPI representable by PNG's integer pixels/metre field."""
    if requested_ppi <= 0:
        raise ValueError("PPI must be positive")
    pixels_per_metre = round(requested_ppi / 0.0254)
    return pixels_per_metre * 0.0254
