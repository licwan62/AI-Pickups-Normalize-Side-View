from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class OutputSettings:
    ppi: float = 72.0


@dataclass(frozen=True)
class QualitySettings:
    pass_max_percent: float = 3.0
    warning_max_percent: float = 6.0
    error_max_percent: float = 6.0


@dataclass(frozen=True)
class DetectionSettings:
    provider: str = "qwen"
    model: str = "qwen3-vl-plus"
    endpoint: str = (
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    )
    api_key: str = ""
    prompt_file: Path = Path("promting.md")
    timeout_seconds: float = 90.0
    manual_fallback: bool = False
    perspective_correction: bool = False


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
    input_path: Path = Path("input/vehicles.csv")
    output: OutputSettings = OutputSettings()
    annotation: AnnotationStyle = AnnotationStyle()
    quality: QualitySettings = QualitySettings()
    detection: DetectionSettings = DetectionSettings()


def load_settings(path: Path) -> Settings:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    configured_input = payload.get("input_csv", payload.get("input_tsv"))
    if configured_input is None:
        csv_default = path.parent / "input" / "vehicles.csv"
        tsv_default = path.parent / "input" / "vehicles.tsv"
        input_path = csv_default if csv_default.is_file() else tsv_default
    else:
        input_path = path.parent / Path(str(configured_input))
    input_path = input_path.resolve()
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
    quality_payload = payload.get("quality") or {}
    pass_max_percent = float(quality_payload.get("pass_max_percent", 3.0))
    warning_max_percent = float(quality_payload.get("warning_max_percent", 6.0))
    # Older configurations used warning_max_percent as the blocking threshold.
    error_max_percent = float(
        quality_payload.get("error_max_percent", warning_max_percent)
    )
    quality = QualitySettings(
        pass_max_percent=pass_max_percent,
        warning_max_percent=warning_max_percent,
        error_max_percent=error_max_percent,
    )
    if quality.pass_max_percent < 0:
        raise ValueError("quality.pass_max_percent must not be negative")
    if quality.warning_max_percent < quality.pass_max_percent:
        raise ValueError(
            "quality.warning_max_percent must be greater than or equal to "
            "quality.pass_max_percent"
        )
    if quality.error_max_percent < quality.warning_max_percent:
        raise ValueError(
            "quality.error_max_percent must be greater than or equal to "
            "quality.warning_max_percent"
        )
    detection_payload = payload.get("detection") or {}
    detection = DetectionSettings(
        provider=str(detection_payload.get("provider", "qwen")).lower(),
        model=str(detection_payload.get("model", "qwen3-vl-plus")),
        endpoint=str(
            detection_payload.get(
                "endpoint",
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            )
        ),
        api_key=str(detection_payload.get("api_key", "")),
        prompt_file=(
            path.parent
            / Path(str(detection_payload.get("prompt_file", "promting.md")))
        ).resolve(),
        timeout_seconds=float(detection_payload.get("timeout_seconds", 90.0)),
        manual_fallback=bool(detection_payload.get("manual_fallback", False)),
        perspective_correction=bool(detection_payload.get("perspective_correction", False)),
    )
    if detection.provider != "qwen":
        raise ValueError("detection.provider must be qwen")
    if detection.timeout_seconds <= 0:
        raise ValueError("detection.timeout_seconds must be positive")
    return Settings(
        input_path=input_path,
        output=OutputSettings(ppi=ppi),
        annotation=annotation,
        quality=quality,
        detection=detection,
    )
