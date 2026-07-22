from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

from .src.annotation import measure_and_trace
from .src.exporter import Exporter
from .src.detector import OpenCVVehicleDetector
from .src.geometry import Bounds, select_bounds
from .src.loader import VehicleRecord, load_image, load_records
from .src.quality import QCStatus, evaluate_ratio
from .src.renderer import render_annotated_svg, render_vehicle_svg
from .src.scaler import ScaleMapping
from .src.settings import AnnotationStyle, load_settings


LOGGER = logging.getLogger("pickup_measure")


def configure_logging(log_path: Path, verbose: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
        force=True,
    )


def process_vehicle(
    record: VehicleRecord,
    output_root: Path,
    approve_warning: bool,
    manual: bool = False,
    reuse_points: bool = False,
    ppi: float = 72.0,
    annotation_style: AnnotationStyle = AnnotationStyle(),
) -> str:
    output_dir = output_root / record.id
    exporter = Exporter(output_dir)
    # A blocked or failed rerun must not leave a stale SVG that looks current.
    vehicle_svg = output_dir / "vehicle.svg"
    vehicle_svg.unlink(missing_ok=True)
    for stale_name in ("annotated.svg", "annotated.pdf", "preview.png", "annotation_points.json", "measurements.tsv"):
        (output_dir / stale_name).unlink(missing_ok=True)
    final_crop_path = output_dir / "crop.png"
    final_crop_path.unlink(missing_ok=True)
    image = load_image(record.image_path)
    exporter.save_source(image)

    points_path = output_dir / "points.json"
    if reuse_points and points_path.exists():
        bounds = Bounds.from_json(points_path)
        LOGGER.info("%s: using saved bounds from %s", record.id, points_path)
    elif manual:
        bounds = select_bounds(image, window_title=f"Select vehicle bounds - {record.name}")
        bounds.to_json(points_path)
        LOGGER.info("%s: saved manual bounds to %s", record.id, points_path)
    else:
        bounds = OpenCVVehicleDetector().detect(image)
        bounds.to_json(points_path)
        LOGGER.info("%s: automatically detected and saved bounds to %s", record.id, points_path)

    bounds.validate(image.width, image.height)
    crop = image.crop(bounds.as_pillow_box())
    exporter.save_source_crop(crop)

    mapping = ScaleMapping.from_dimensions(
        pixel_width=bounds.pixel_width,
        pixel_height=bounds.pixel_height,
        length_mm=record.length_mm,
        height_mm=record.height_mm,
    )
    qc = evaluate_ratio(
        name=record.name,
        source_width=bounds.pixel_width,
        source_height=bounds.pixel_height,
        length_mm=record.length_mm,
        height_mm=record.height_mm,
    )
    qc_payload = {
        **asdict(qc),
        "distortion_percent": round(qc.distortion * 100, 4),
        "scale_x_mm_per_px": mapping.scale_x,
        "scale_y_mm_per_px": mapping.scale_y,
        "warning_approved": bool(qc.status is QCStatus.WARNING and approve_warning),
    }
    exporter.write_json("qc_report.json", qc_payload)

    if qc.status is QCStatus.BLOCKED:
        LOGGER.error("%s: blocked (ratio distortion %.2f%%)", record.id, qc.distortion * 100)
        return "BLOCKED"
    if qc.status is QCStatus.WARNING and not approve_warning:
        LOGGER.warning(
            "%s: warning (ratio distortion %.2f%%); export paused. Review and rerun with --approve-warning.",
            record.id,
            qc.distortion * 100,
        )
        return "WARNING"

    render_vehicle_svg(
        image=crop,
        output_path=vehicle_svg,
        width_mm=record.length_mm,
        height_mm=record.height_mm,
    )
    annotation = measure_and_trace(crop, mapping, record.height_mm)
    exporter.write_json("annotation_points.json", annotation.points_payload())
    render_annotated_svg(
        image=crop,
        geometry=annotation,
        output_path=output_dir / "annotated.svg",
        width_mm=record.length_mm,
        height_mm=record.height_mm,
        style=annotation_style,
    )
    exporter.write_measurements({
        "id": record.id,
        "name": record.name,
        "length_mm": round(record.length_mm),
        "height_mm": round(record.height_mm),
        "width_mm": round(record.width_mm),
        "bed_height_mm": round(annotation.bed_height_mm),
        "cab_height_mm": round(annotation.cab_height_mm),
        "neck_height_mm": round(annotation.neck_height_mm),
        "hood_height_mm": round(annotation.hood_height_mm),
        "door_seam_x_mm": round(annotation.door_seam_x_mm),
        "door_to_front_mm": round(annotation.door_to_front_mm),
        "qc_status": qc.status.value,
    })
    LOGGER.info(
        "%s: exported vehicle.svg and annotated.svg at %.1f mm x %.1f mm",
        record.id,
        record.length_mm,
        record.height_mm,
    )
    return "EXPORTED"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate true-size pickup side-profile SVG files.")
    parser.add_argument("--input", type=Path, default=Path("input/vehicles.tsv"), help="Input TSV/CSV path")
    parser.add_argument(
        "--images", type=Path, default=Path("input/images"),
        help="Default image directory; filenames match vehicle IDs",
    )
    parser.add_argument("--output", type=Path, default=Path("output"), help="Output directory")
    parser.add_argument(
        "--config", type=Path, default=Path("config.yaml"), help="YAML configuration path"
    )
    parser.add_argument(
        "--manual",
        action="store_true",
        help="Open the manual selection window instead of automatic detection",
    )
    parser.add_argument(
        "--reuse-points",
        action="store_true",
        help="Reuse an existing points.json instead of detecting again",
    )
    parser.add_argument(
        "--approve-warning",
        action="store_true",
        help="Export 3%%-6%% distortion warnings after human review",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.output / "pickup_measure.log", args.verbose)
    try:
        settings = load_settings(args.config)
        records = load_records(args.input, args.images)
    except Exception:
        LOGGER.exception("Could not load input TSV: %s", args.input)
        return 2

    summary: list[dict[str, str]] = []
    for record in records:
        try:
            status = process_vehicle(
                record,
                output_root=args.output,
                approve_warning=args.approve_warning,
                manual=args.manual,
                reuse_points=args.reuse_points,
                ppi=settings.output.ppi,
                annotation_style=settings.annotation,
            )
            summary.append({"id": record.id, "status": status, "error": ""})
        except Exception as exc:
            LOGGER.exception("%s: processing failed", record.id)
            summary.append({"id": record.id, "status": "FAILED", "error": str(exc)})

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    record_by_id = {record.id: record for record in records}
    aggregate_path = args.output / "measurements.tsv"
    aggregate_fields = ["车型", "车长", "车宽", "CAB高", "车头高", "车颈高", "车尾高"]
    with aggregate_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=aggregate_fields, delimiter="\t")
        writer.writeheader()
        for item in summary:
            if item["status"] != "EXPORTED":
                continue
            record = record_by_id[item["id"]]
            annotation_payload = json.loads(
                (args.output / record.id / "annotation_points.json").read_text(encoding="utf-8")
            )
            writer.writerow({
                "车型": record.name,
                "车长": round(record.length_mm),
                "车宽": round(record.width_mm),
                "CAB高": round(annotation_payload["cab_height_mm"]),
                "车头高": round(annotation_payload["hood_height_mm"]),
                "车颈高": round(annotation_payload["neck_height_mm"]),
                "车尾高": round(annotation_payload["bed_height_mm"]),
            })
    exported = sum(item["status"] == "EXPORTED" for item in summary)
    LOGGER.info("Run complete: %d/%d exported", exported, len(summary))
    return 0 if exported == len(summary) else 1


if __name__ == "__main__":
    sys.exit(main())
