from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class OutputSettings:
    ppi: float = 72.0


@dataclass(frozen=True)
class AnnotationStyle:
    background_color: str = "#FFFFFF"
    image_opacity: float = 0.5
    outline_color: str = "#C8242A"
    outline_width_mm: float = 8.0
    dimension_color: str = "#202124"
    dimension_width_mm: float = 4.0
    font_size_mm: float = 82.0
    font_family: str = "Arial"


@dataclass(frozen=True)
class Settings:
    output: OutputSettings = OutputSettings()
    annotation: AnnotationStyle = AnnotationStyle()


def load_settings(path: Path) -> Settings:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    output_payload = payload.get("output") or {}
    ppi = float(output_payload.get("ppi", 72.0))
    if ppi <= 0:
        raise ValueError("output.ppi must be positive")
    annotation_payload = payload.get("annotation") or {}
    annotation = AnnotationStyle(
        background_color=str(annotation_payload.get("background_color", "#FFFFFF")),
        image_opacity=float(annotation_payload.get("image_opacity", 0.5)),
        outline_color=str(annotation_payload.get("outline_color", "#C8242A")),
        outline_width_mm=float(annotation_payload.get("outline_width_mm", 8.0)),
        dimension_color=str(annotation_payload.get("dimension_color", "#202124")),
        dimension_width_mm=float(annotation_payload.get("dimension_width_mm", 4.0)),
        font_size_mm=float(annotation_payload.get("font_size_mm", 82.0)),
        font_family=str(annotation_payload.get("font_family", "Arial")),
    )
    if not 0 <= annotation.image_opacity <= 1:
        raise ValueError("annotation.image_opacity must be between 0 and 1")
    if min(annotation.outline_width_mm, annotation.dimension_width_mm, annotation.font_size_mm) <= 0:
        raise ValueError("annotation widths and font size must be positive")
    return Settings(output=OutputSettings(ppi=ppi), annotation=annotation)
