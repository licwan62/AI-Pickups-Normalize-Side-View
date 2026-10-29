from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import yaml


DEFAULT_MASTER_TABLE_SOURCE = (
    r"\\NAS8824B4\Public\PQData\pub_all_cars_data\data\基础数据"
)
DEFAULT_MASTER_TABLE_PATTERNS = ("*/全量/全量表.csv",)


@dataclass(frozen=True)
class OutputSettings:
    ppi: float = 72.0


@dataclass(frozen=True)
class DetectionSettings:
    provider: str = "qwen"
    model: str = "qwen3-vl-plus"
    endpoint: str = (
        "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    )
    api_key: str = ""
    api_key_env: str = "DASHSCOPE_API_KEY"
    prompt_file: Path = Path("promting.md")
    timeout_seconds: float = 90.0
    manual_fallback: bool = False
    perspective_correction: bool = False


@dataclass(frozen=True)
class MasterTableSettings:
    """Locations and matching rules for the authoritative vehicle table."""

    source: Path = Path(DEFAULT_MASTER_TABLE_SOURCE)
    patterns: tuple[str, ...] = DEFAULT_MASTER_TABLE_PATTERNS
    cache_path: Path = Path("data/cache/master_vehicles.csv")
    refresh: str = "auto"
    fallback_size: str = "UNCLASSIFIED"
    preferred_markets: tuple[str, ...] = ("US", "EU", "RU")
    min_score: float = 0.62
    identify_prompt_file: Path = Path("promting_identify.md")
    columns: Mapping[str, str] = field(default_factory=dict)


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
    detection: DetectionSettings = DetectionSettings()
    master_table: MasterTableSettings = MasterTableSettings()
    output_dir: Path = Path("output")


def _resolve_config_path(config_path: Path, value: object) -> Path:
    candidate = Path(str(value)).expanduser()
    if candidate.is_absolute():
        return candidate
    return (config_path.parent / candidate).resolve()


def _string_tuple(value: object, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, (list, tuple)):
        values = tuple(str(item) for item in value)
    else:
        raise ValueError("master_table.patterns must be a string or a list")
    cleaned = tuple(item.strip() for item in values if str(item).strip())
    return cleaned or default


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
    detection_payload = payload.get("detection") or {}
    api_key_env = str(
        detection_payload.get("api_key_env", "DASHSCOPE_API_KEY")
    ).strip()
    configured_api_key = str(detection_payload.get("api_key", "")).strip()
    api_key = configured_api_key or os.environ.get(api_key_env, "").strip()
    detection = DetectionSettings(
        provider=str(detection_payload.get("provider", "qwen")).lower(),
        model=str(detection_payload.get("model", "qwen3-vl-plus")),
        endpoint=str(
            detection_payload.get(
                "endpoint",
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            )
        ),
        api_key=api_key,
        api_key_env=api_key_env,
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

    master_payload = payload.get("master_table") or {}
    refresh = str(master_payload.get("refresh", "auto")).lower()
    if refresh not in {"auto", "always", "never"}:
        raise ValueError("master_table.refresh must be auto, always, or never")
    min_score = float(master_payload.get("min_score", 0.62))
    if not 0 <= min_score <= 1:
        raise ValueError("master_table.min_score must be between 0 and 1")
    configured_columns = master_payload.get("columns") or {}
    if not isinstance(configured_columns, dict):
        raise ValueError("master_table.columns must be a mapping")
    master_table = MasterTableSettings(
        source=_resolve_config_path(
            path,
            master_payload.get("source", DEFAULT_MASTER_TABLE_SOURCE),
        ),
        patterns=_string_tuple(
            master_payload.get("patterns"),
            DEFAULT_MASTER_TABLE_PATTERNS,
        ),
        cache_path=_resolve_config_path(
            path,
            master_payload.get("cache_path", "data/cache/master_vehicles.csv"),
        ),
        refresh=refresh,
        fallback_size=str(
            master_payload.get("fallback_size", "UNCLASSIFIED")
        ).strip() or "UNCLASSIFIED",
        preferred_markets=_string_tuple(
            master_payload.get("preferred_markets"),
            ("US", "EU", "RU"),
        ),
        min_score=min_score,
        identify_prompt_file=_resolve_config_path(
            path,
            master_payload.get("identify_prompt_file", "promting_identify.md"),
        ),
        columns={
            str(key): str(value)
            for key, value in configured_columns.items()
        },
    )
    return Settings(
        input_path=input_path,
        output_dir=(path.parent / Path(str(payload.get("output_dir", "output")))).resolve(),
        output=OutputSettings(ppi=ppi),
        annotation=annotation,
        detection=detection,
        master_table=master_table,
    )
