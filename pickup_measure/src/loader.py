from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd
from PIL import Image, ImageOps


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
REQUIRED_COLUMNS = {"id", "name", "Size", "length_mm", "width_mm", "height_mm"}
SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]+$")
SAFE_SIZE = re.compile(r"^[A-Za-z0-9+_.-]+$")


@dataclass(frozen=True)
class VehicleRecord:
    id: str
    name: str
    image_path: Path
    length_mm: float
    width_mm: float
    height_mm: float
    size: str = "UNCLASSIFIED"


def _resolve_image_path(raw_path: str, tsv_path: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    beside_tsv = (tsv_path.parent / path).resolve()
    if beside_tsv.exists():
        return beside_tsv
    return (Path.cwd() / path).resolve()


def _find_image_by_id(vehicle_id: str, images_dir: Path) -> Path:
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Default image directory not found: {images_dir}")
    matches = [
        path for path in images_dir.iterdir()
        if path.is_file() and path.stem.lower() == vehicle_id.lower()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not matches:
        raise FileNotFoundError(
            f"No image named {vehicle_id}.jpg/.jpeg/.png/.webp in {images_dir}"
        )
    if len(matches) > 1:
        raise ValueError(f"Multiple images found for ID {vehicle_id}: {matches}")
    return matches[0].resolve()


def load_records(tsv_path: Path, images_dir: Path | None = None) -> list[VehicleRecord]:
    tsv_path = tsv_path.resolve()
    separator = "," if tsv_path.suffix.lower() == ".csv" else "\t"
    frame = pd.read_csv(
        tsv_path,
        sep=separator,
        dtype={"id": str, "name": str, "Size": str, "image_path": str},
    )
    missing = REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required TSV columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("Input TSV contains no vehicles")
    if frame["id"].duplicated().any():
        duplicates = frame.loc[frame["id"].duplicated(keep=False), "id"].tolist()
        raise ValueError(f"Duplicate vehicle IDs: {duplicates}")

    records: list[VehicleRecord] = []
    for row_number, row in frame.iterrows():
        try:
            vehicle_id = str(row["id"]).strip()
            if not vehicle_id or vehicle_id in {".", ".."} or not SAFE_ID.fullmatch(vehicle_id):
                raise ValueError("id may contain only letters, numbers, dot, underscore, and hyphen")
            vehicle_size = str(row["Size"]).strip()
            if (
                not vehicle_size
                or vehicle_size in {".", ".."}
                or not SAFE_SIZE.fullmatch(vehicle_size)
            ):
                raise ValueError(
                    "Size may contain only letters, numbers, plus, dot, underscore, and hyphen"
                )
            dimensions = [float(row[key]) for key in ("length_mm", "width_mm", "height_mm")]
            if any(value <= 0 for value in dimensions):
                raise ValueError("dimensions must be positive")
            raw_image_path = row.get("image_path")
            if raw_image_path is not None and pd.notna(raw_image_path) and str(raw_image_path).strip():
                raw_path_text = str(raw_image_path).strip()
                image_path = _resolve_image_path(raw_path_text, tsv_path)
                if not image_path.is_file() and images_dir is not None:
                    configured_fallback = images_dir.resolve() / Path(raw_path_text).name
                    if configured_fallback.is_file():
                        image_path = configured_fallback
            else:
                default_dir = images_dir.resolve() if images_dir else tsv_path.parent / "images"
                image_path = _find_image_by_id(vehicle_id, default_dir)
            if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"unsupported image type: {image_path.suffix}")
            records.append(
                VehicleRecord(
                    id=vehicle_id,
                    name=str(row["name"]).strip(),
                    image_path=image_path,
                    length_mm=dimensions[0],
                    width_mm=dimensions[1],
                    height_mm=dimensions[2],
                    size=vehicle_size,
                )
            )
        except Exception as exc:
            raise ValueError(f"Invalid TSV row {row_number + 2}: {exc}") from exc
    return records


def load_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as opened:
        corrected = ImageOps.exif_transpose(opened)
        if "A" in corrected.getbands() or "transparency" in corrected.info:
            rgba = corrected.convert("RGBA")
            white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            return Image.alpha_composite(white, rgba).convert("RGB")
        return corrected.convert("RGB")
