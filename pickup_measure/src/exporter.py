from __future__ import annotations

import json
import csv
from pathlib import Path

from PIL import Image


class Exporter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_source(self, image: Image.Image) -> Path:
        path = self.output_dir / "source.jpg"
        image.save(path, format="JPEG", quality=95, subsampling=0)
        return path

    def save_source_crop(self, image: Image.Image) -> Path:
        path = self.output_dir / "crop_source.png"
        image.save(path, format="PNG")
        return path

    def save_crop(self, image: Image.Image, ppi: float) -> Path:
        path = self.output_dir / "crop.png"
        image.save(path, format="PNG", dpi=(ppi, ppi))
        return path

    def write_json(self, filename: str, payload: object) -> Path:
        path = self.output_dir / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def write_measurements(self, payload: dict[str, object]) -> Path:
        path = self.output_dir / "measurements.tsv"
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(payload), delimiter="\t")
            writer.writeheader()
            writer.writerow(payload)
        return path
