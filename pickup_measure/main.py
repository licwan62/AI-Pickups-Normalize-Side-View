from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .src.annotation import classify_background, measure_and_trace
from .src.exporter import Exporter
from .src.geometry import (
    Bounds,
    detect_rectified_vehicle_edges,
    rectify_perspective,
    select_bounds,
    transform_perspective_points,
)
from .src.loader import VehicleRecord, load_image, load_records
from .src.orientation import normalize_front_to_right
from .src.quality import QCStatus, evaluate_ratio
from .src.qwen_detector import QwenVehicleDetector
from .src.renderer import (
    render_annotated_svg,
    render_dimensioned_vehicle_svg,
    render_vehicle_svg,
)
from .src.scaler import ScaleMapping
from .src.settings import (
    AnnotationStyle,
    DetectionSettings,
    QualitySettings,
    load_settings,
)


LOGGER = logging.getLogger("pickup_measure")

AGGREGATE_MEASUREMENT_FIELDS = [
    "NAME",
    "SIZE",
    "L-MM",
    "W-MM",
    "H-MM",
    "CAB-H",
    "HOOD-H",
    "NECK-H",
    "BED-H",
]


def terminal_print(message: str) -> None:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        message.encode(encoding)
    except UnicodeEncodeError:
        message = (
            message
            .replace("✅", "[OK]")
            .replace("❌", "[FAIL]")
            .replace("⏭️", "[SKIP]")
            .replace("—", "-")
        )
    print(message, flush=True)


def terminal_stage_done(stage: str, details: str) -> None:
    terminal_print(f"  ✅ {stage} done [{details}]")


def configure_logging(log_path: Path, verbose: bool = False) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    console_handler = logging.StreamHandler()
    # Detailed errors go to the log file. main() prints one concise ❌ line.
    console_handler.setLevel(logging.CRITICAL)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[file_handler, console_handler],
        force=True,
    )


def write_aggregate_measurements(
    output_root: Path,
    records: list[VehicleRecord],
    included_ids: set[str] | None = None,
) -> Path:
    """Write the public one-row-per-vehicle measurements table."""
    aggregate_path = output_root / "measurements.tsv"
    output_root.mkdir(parents=True, exist_ok=True)
    with aggregate_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=AGGREGATE_MEASUREMENT_FIELDS,
            delimiter="\t",
        )
        writer.writeheader()
        for record in records:
            if included_ids is not None and record.id not in included_ids:
                continue
            annotation_path = (
                output_root
                / record.size
                / record.id
                / "annotation_points.json"
            )
            if not annotation_path.is_file():
                continue
            annotation = json.loads(
                annotation_path.read_text(encoding="utf-8")
            )
            writer.writerow({
                "NAME": record.name,
                "SIZE": record.size,
                "L-MM": round(record.length_mm),
                "W-MM": round(record.width_mm),
                "H-MM": round(record.height_mm),
                "CAB-H": round(annotation["cab_height_mm"]),
                "HOOD-H": round(annotation["hood_height_mm"]),
                "NECK-H": round(annotation["neck_height_mm"]),
                "BED-H": round(annotation["bed_height_mm"]),
            })
    return aggregate_path


def process_vehicle(
    record: VehicleRecord,
    output_root: Path,
    approve_warning: bool,
    manual: bool = False,
    reuse_points: bool = False,
    ppi: float = 72.0,
    annotation_style: AnnotationStyle = AnnotationStyle(),
    quality_settings: QualitySettings = QualitySettings(),
    detection_settings: DetectionSettings = DetectionSettings(),
    progress_state: dict[str, str] | None = None,
    measure: bool = True,
) -> str:
    process_started_at = time.perf_counter()

    def set_stage(stage: str) -> None:
        if progress_state is not None:
            progress_state["stage"] = stage

    size_output_dir = output_root / record.size
    output_dir = size_output_dir / record.id
    exporter = Exporter(output_dir)
    # A blocked or failed rerun must not leave a stale SVG that looks current.
    vehicle_svg = output_dir / "vehicle.svg"
    annotated_svg = size_output_dir / f"{record.id}.svg"
    vehicle_svg.unlink(missing_ok=True)
    annotated_svg.unlink(missing_ok=True)
    for stale_name in (
        "annotated.svg",
        "annotated.pdf",
        "preview.png",
        "annotation_points.json",
        "orientation.json",
        "measurements.tsv",
        "qc_report.json",
        "crop_attempt.png",
        "detection_attempt.json",
        "detection.json",
    ):
        if reuse_points and stale_name == "detection.json":
            continue
        (output_dir / stale_name).unlink(missing_ok=True)
    final_crop_path = output_dir / "crop.png"
    final_crop_path.unlink(missing_ok=True)
    set_stage("LOAD_IMAGE")
    image = load_image(record.image_path)
    exporter.save_source(image)

    points_path = output_dir / "points.json"
    perspective_quad = None
    boundary_touch_points = None
    wheel_contact_points = None
    body_chassis_line = None
    cached_crop_chassis_hint = None
    api_front = "unknown"
    api_background_type = "unknown"
    detector = None
    detection_payload: dict[str, object] | None = None
    detection_succeeded = False
    localization_source = "qwen"
    if reuse_points and points_path.exists():
        localization_source = "reuse"
        set_stage("QWEN_LOCALIZE")
        bounds = Bounds.from_json(points_path)
        detection_path = output_dir / "detection.json"
        if detection_path.is_file():
            detection_payload = json.loads(
                detection_path.read_text(encoding="utf-8")
            )
            raw_crop_chassis_hint = detection_payload.get(
                "crop_body_chassis_line"
            )
            if (
                isinstance(raw_crop_chassis_hint, list)
                and len(raw_crop_chassis_hint) == 2
            ):
                cached_crop_chassis_hint = [
                    (float(point[0]), float(point[1]))
                    for point in raw_crop_chassis_hint
                    if isinstance(point, (list, tuple))
                    and len(point) == 2
                ]
                if len(cached_crop_chassis_hint) != 2:
                    cached_crop_chassis_hint = None
            raw_quad = detection_payload.get("perspective_quad")
            if isinstance(raw_quad, list) and len(raw_quad) == 4:
                perspective_quad = tuple(
                    tuple(float(value) for value in point)
                    for point in raw_quad
                )
                raw_response = detection_payload.get("response")
                if isinstance(raw_response, str):
                    try:
                        response_payload = QwenVehicleDetector._extract_json(
                            raw_response
                        )
                        raw_background_type = str(
                            response_payload.get(
                                "background_type",
                                detection_payload.get(
                                    "background_type",
                                    "unknown",
                                ),
                            )
                        ).lower()
                        if raw_background_type in {
                            "white",
                            "transparent",
                            "environment",
                        }:
                            api_background_type = raw_background_type
                        response_quad = QwenVehicleDetector._quad_from_payload(
                            response_payload,
                            image.width,
                            image.height,
                        )
                        boundary_touch_points = (
                            QwenVehicleDetector
                            ._boundary_touch_points_from_payload(
                                response_payload,
                                image.width,
                                image.height,
                                bounds,
                            )
                        )
                        qwen_wheels = (
                            QwenVehicleDetector._wheel_centers_from_payload(
                                response_payload,
                                image.width,
                                image.height,
                            )
                        )
                        qwen_wheel_contacts = (
                            QwenVehicleDetector._wheel_contacts_from_payload(
                                response_payload,
                                image.width,
                                image.height,
                            )
                        )
                        body_chassis_line = (
                            QwenVehicleDetector
                            ._body_chassis_line_from_payload(
                                response_payload,
                                image.width,
                                image.height,
                            )
                        )
                        image_wheels = QwenVehicleDetector._image_wheel_centers(
                            image,
                            bounds,
                            qwen_wheels,
                        )
                        if image_wheels is not None:
                            # Old saved responses did not contain tyre-contact
                            # semantics. Do not invent an aggressive keystone
                            # from an unhinted road/tyre edge search: a ground
                            # or shadow edge can reverse the apparent near/far
                            # correction after orientation normalization.
                            wheel_contact_points = (
                                QwenVehicleDetector._image_wheel_contacts(
                                    image,
                                    bounds,
                                    image_wheels,
                                    qwen_wheel_contacts,
                                )
                                if qwen_wheel_contacts is not None
                                else None
                            )
                            perspective_quad = (
                                QwenVehicleDetector._align_quad_to_wheel_axis(
                                    response_quad,
                                    image_wheels,
                                    wheel_contact_points,
                                )
                            )
                            detection_payload["perspective_quad"] = (
                                perspective_quad
                            )
                            detection_payload["image_wheel_centers"] = (
                                image_wheels
                            )
                            detection_payload["image_wheel_contacts"] = (
                                wheel_contact_points
                            )
                            exporter.write_json(
                                "detection.json",
                                detection_payload,
                            )
                    except RuntimeError:
                        LOGGER.debug(
                            "%s: could not upgrade reused perspective from "
                            "image wheel centers",
                            record.id,
                            exc_info=True,
                        )
            api_front = str(detection_payload.get("front", "unknown"))
        LOGGER.info("%s: using saved bounds from %s", record.id, points_path)
    elif manual:
        localization_source = "manual"
        set_stage("QWEN_LOCALIZE")
        bounds = select_bounds(image, window_title=f"Select vehicle bounds - {record.name}")
        bounds.to_json(points_path)
        LOGGER.info("%s: saved manual bounds to %s", record.id, points_path)
    else:
        detector = QwenVehicleDetector(
            model=detection_settings.model,
            endpoint=detection_settings.endpoint,
            api_key=detection_settings.api_key,
            prompt_file=detection_settings.prompt_file,
            timeout_seconds=detection_settings.timeout_seconds,
            perspective_correction=detection_settings.perspective_correction,
        )
        try:
            set_stage("QWEN_LOCALIZE")
            bounds = detector.detect(
                image,
                expected_aspect=record.length_mm / record.height_mm,
            )
        except Exception as exc:
            attempt_bounds = detector.last_attempt_bounds
            attempt_payload: dict[str, object] = {
                "status": "REJECTED",
                "error": str(exc),
                "provider": "qwen",
                "model": detection_settings.model,
                "expected_aspect": record.length_mm / record.height_mm,
                "source_width": image.width,
                "source_height": image.height,
                "bounds": None,
                "responses": detector.last_response_contents,
            }
            if attempt_bounds is not None:
                attempt_payload["bounds"] = asdict(attempt_bounds)
                attempt_payload["attempt_aspect"] = (
                    attempt_bounds.pixel_width / attempt_bounds.pixel_height
                )
                image.crop(attempt_bounds.as_pillow_box()).save(
                    output_dir / "crop_attempt.png",
                    format="PNG",
                )
            exporter.write_json("detection_attempt.json", attempt_payload)
            if not detection_settings.manual_fallback:
                raise
            localization_source = "manual_fallback"
            LOGGER.error(
                "%s: automatic detection failed (%s); opening manual crop",
                record.id,
                exc,
            )
            bounds = select_bounds(
                image,
                window_title=f"Automatic crop failed - select vehicle bounds - {record.name}",
            )
            attempt_payload["status"] = "MANUAL_FALLBACK"
            attempt_payload["manual_bounds"] = asdict(bounds)
            exporter.write_json("detection_attempt.json", attempt_payload)
            LOGGER.info(
                "%s: automatic detection failed; manual fallback bounds selected",
                record.id,
            )
        else:
            perspective_quad = detector.last_perspective_quad
            boundary_touch_points = detector.last_boundary_touch_points
            wheel_contact_points = detector.last_image_wheel_contacts
            body_chassis_line = detector.last_body_chassis_line
            api_front = detector.last_front
            api_background_type = detector.last_background_type
            detection_succeeded = True
        bounds.to_json(points_path)
        LOGGER.info("%s: automatically detected and saved bounds to %s", record.id, points_path)

    terminal_stage_done(
        "QWEN_LOCALIZE",
        f"source={localization_source}, "
        f"bbox=[{bounds.left},{bounds.roof},{bounds.right},{bounds.ground}]",
    )
    set_stage("PERSPECTIVE_CROP")
    bounds.validate(image.width, image.height)
    rectified_trim: dict[str, int] | None = None
    crop_left = 0
    crop_top = 0
    if perspective_quad is not None:
        raw_crop = rectify_perspective(image, perspective_quad, bounds)
        midpoint_x = (bounds.left + bounds.right) / 2
        boundary_source_points = (
            [
                boundary_touch_points["leftmost"],
                boundary_touch_points["rightmost"],
                boundary_touch_points["topmost"],
                (midpoint_x, float(bounds.ground)),
            ]
            if boundary_touch_points is not None
            else [
                (float(bounds.left), float(bounds.roof)),
                (float(bounds.right), float(bounds.roof)),
                (float(bounds.left), float(bounds.roof)),
                (midpoint_x, float(bounds.ground)),
            ]
        )
        (
            mapped_left,
            mapped_right,
            mapped_top,
            mapped_ground,
        ) = transform_perspective_points(
            boundary_source_points,
            perspective_quad,
            bounds,
        )
        if wheel_contact_points is not None:
            mapped_contacts = transform_perspective_points(
                wheel_contact_points,
                perspective_quad,
                bounds,
            )
            # Both contacts are level after the aggressive wheel-plane
            # correction. Include their antialiased edge pixel and no ground
            # safety margin.
            mapped_ground = (
                mapped_ground[0],
                max(point[1] for point in mapped_contacts) + 1.0,
            )
        # A perspective-transformed bbox has wedge-shaped padding. New Qwen
        # responses provide the actual vehicle pixels touching left/right/top,
        # which remove those wedges without cutting fixed body parts.
        crop_left = max(0, min(raw_crop.width - 2, math.floor(mapped_left[0])))
        crop_right = max(
            crop_left + 2,
            min(raw_crop.width, math.ceil(mapped_right[0])),
        )
        crop_top = max(0, min(raw_crop.height - 2, math.floor(mapped_top[1])))
        crop_bottom = max(
            crop_top + 2,
            min(raw_crop.height, math.ceil(mapped_ground[1])),
        )
        pixel_edges = detect_rectified_vehicle_edges(
            image,
            perspective_quad,
            bounds,
        )
        if pixel_edges is not None:
            pixel_left, pixel_right, pixel_top = pixel_edges
            maximum_side_snap = max(2, math.ceil(raw_crop.width * 0.02))
            maximum_top_snap = max(2, math.ceil(raw_crop.height * 0.02))
            if crop_left <= pixel_left <= crop_left + maximum_side_snap:
                crop_left = pixel_left
            if crop_right - maximum_side_snap <= pixel_right <= crop_right:
                crop_right = pixel_right
            if crop_top <= pixel_top <= crop_top + maximum_top_snap:
                crop_top = pixel_top
        rectified_trim = {
            "left": crop_left,
            "top": crop_top,
            "right": raw_crop.width - crop_right,
            "bottom": raw_crop.height - crop_bottom,
        }
        if (
            crop_left > 0
            or crop_top > 0
            or crop_right < raw_crop.width
            or crop_bottom < raw_crop.height
        ):
            raw_crop = raw_crop.crop(
                (crop_left, crop_top, crop_right, crop_bottom)
            )
    else:
        raw_crop = image.crop(bounds.as_pillow_box())
    crop, orientation = normalize_front_to_right(
        raw_crop,
        detected_front_override=api_front,
    )
    crop_chassis_hint: list[tuple[float, float]] | None = None
    if body_chassis_line is not None:
        if perspective_quad is not None:
            mapped_chassis_hint = transform_perspective_points(
                body_chassis_line,
                perspective_quad,
                bounds,
            )
            crop_chassis_hint = [
                (x - crop_left, y - crop_top)
                for x, y in mapped_chassis_hint
            ]
        else:
            crop_chassis_hint = [
                (x - bounds.left, y - bounds.roof)
                for x, y in body_chassis_line
            ]
        if orientation.flipped:
            crop_chassis_hint = [
                (float(crop.width - 1) - x, y)
                for x, y in crop_chassis_hint
            ]
            crop_chassis_hint.sort(key=lambda point: point[0])
    elif cached_crop_chassis_hint is not None:
        crop_chassis_hint = cached_crop_chassis_hint
    background_type = classify_background(
        image,
        ai_hint=api_background_type,
    )
    exporter.write_json("orientation.json", orientation.payload())
    if orientation.flipped:
        LOGGER.info(
            "%s: detected front on the left (confidence %.2f); flipped to face right",
            record.id,
            orientation.confidence,
        )
    elif orientation.detected_front == "unknown":
        LOGGER.warning(
            "%s: vehicle direction confidence is too low; crop was not flipped",
            record.id,
        )
    else:
        LOGGER.info(
            "%s: detected front on the right (confidence %.2f)",
            record.id,
            orientation.confidence,
        )

    if detection_succeeded and detector is not None:
        exporter.write_json("detection.json", {
            "schema_version": 2,
            "provider": "qwen",
            "model": detection_settings.model,
            "bounds": asdict(bounds),
            "perspective_quad": perspective_quad,
            "rectified_trim": rectified_trim,
            "image_wheel_centers": detector.last_image_wheel_centers,
            "image_wheel_contacts": wheel_contact_points,
            "body_chassis_line": body_chassis_line,
            "crop_body_chassis_line": crop_chassis_hint,
            "boundary_touch_points": boundary_touch_points,
            "background_type": background_type,
            "front": api_front,
            "response": detector.last_response_content,
            "responses": detector.last_response_contents,
        })
    elif detection_payload is not None:
        detection_payload.update({
            "schema_version": 2,
            "perspective_quad": perspective_quad,
            "rectified_trim": rectified_trim,
            "boundary_touch_points": boundary_touch_points,
            "image_wheel_contacts": wheel_contact_points,
            "body_chassis_line": body_chassis_line,
            "crop_body_chassis_line": crop_chassis_hint,
            "background_type": background_type,
            "front": api_front,
        })
        exporter.write_json("detection.json", detection_payload)
    exporter.save_source_crop(crop)
    terminal_stage_done(
        "PERSPECTIVE_CROP",
        f"crop={crop.width}x{crop.height}, "
        f"perspective={'yes' if perspective_quad is not None else 'no'}, "
        f"front={orientation.normalized_front}, background={background_type}",
    )

    if not measure:
        set_stage("RENDER_EXPORT")
        render_dimensioned_vehicle_svg(
            image=crop,
            output_path=vehicle_svg,
            width_mm=record.length_mm,
            height_mm=record.height_mm,
            style=annotation_style,
        )
        LOGGER.info(
            "%s: exported crop-only vehicle.svg at %.1f mm x %.1f mm",
            record.id,
            record.length_mm,
            record.height_mm,
        )
        terminal_stage_done(
            "RENDER_EXPORT",
            f"status=EXPORTED, mode=no-measure, "
            f"elapsed={time.perf_counter() - process_started_at:.1f}s",
        )
        return "EXPORTED"

    set_stage("MEASURE_RENDER_EXPORT")
    mapping = ScaleMapping.from_dimensions(
        pixel_width=crop.width,
        pixel_height=crop.height,
        length_mm=record.length_mm,
        height_mm=record.height_mm,
    )
    qc = evaluate_ratio(
        name=record.name,
        source_width=crop.width,
        source_height=crop.height,
        length_mm=record.length_mm,
        height_mm=record.height_mm,
        pass_max_percent=quality_settings.pass_max_percent,
        warning_max_percent=quality_settings.warning_max_percent,
        error_max_percent=quality_settings.error_max_percent,
    )
    qc_payload = {
        **asdict(qc),
        "distortion_percent": round(qc.distortion * 100, 4),
        "pass_max_percent": quality_settings.pass_max_percent,
        "warning_max_percent": quality_settings.warning_max_percent,
        "error_max_percent": quality_settings.error_max_percent,
        "warning_limit_exceeded": (
            qc.distortion * 100 > quality_settings.warning_max_percent
        ),
        "scale_x_mm_per_px": mapping.scale_x,
        "scale_y_mm_per_px": mapping.scale_y,
        "warning_approved": bool(qc.status is QCStatus.WARNING and approve_warning),
        "warning_auto_continued": qc.status is QCStatus.WARNING,
    }
    exporter.write_json("qc_report.json", qc_payload)

    if qc.status is QCStatus.BLOCKED:
        LOGGER.error("%s: blocked (ratio distortion %.2f%%)", record.id, qc.distortion * 100)
        if progress_state is not None:
            progress_state["error"] = (
                f"ratio distortion {qc.distortion * 100:.2f}% exceeds limit"
            )
        return "BLOCKED"
    if qc.status is QCStatus.WARNING:
        LOGGER.warning(
            "%s: warning (ratio distortion %.2f%%); continuing to dimension annotation.",
            record.id,
            qc.distortion * 100,
        )

    render_vehicle_svg(
        image=crop,
        output_path=vehicle_svg,
        width_mm=record.length_mm,
        height_mm=record.height_mm,
    )
    # AI stops at localization, direction normalization, perspective correction,
    # and cropping. All red vehicle structure is measured from the corrected
    # pixels by the deterministic Python annotation pipeline.
    annotation = measure_and_trace(
        crop,
        mapping,
        record.height_mm,
        background_type=background_type,
        chassis_hint_line=crop_chassis_hint,
    )
    exporter.write_json("annotation_points.json", annotation.points_payload())
    render_annotated_svg(
        image=crop,
        geometry=annotation,
        output_path=annotated_svg,
        width_mm=record.length_mm,
        height_mm=record.height_mm,
        model_name=record.name,
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
        "qc_status": qc.status.value,
    })
    LOGGER.info(
        "%s: exported vehicle.svg and %s at %.1f mm x %.1f mm",
        record.id,
        annotated_svg,
        record.length_mm,
        record.height_mm,
    )
    terminal_stage_done(
        "MEASURE_RENDER_EXPORT",
        f"status=EXPORTED, qc={qc.status.value}, "
        f"elapsed={time.perf_counter() - process_started_at:.1f}s",
    )
    return "EXPORTED"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate true-size pickup side-profile SVG files.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input CSV/TSV path (defaults to input_csv/input_tsv in config.yaml)",
    )
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
        "--continue",
        dest="continue_mode",
        action="store_true",
        help="Skip IDs whose final SVG and annotation data already exist",
    )
    parser.add_argument(
        "--approve-warning",
        action="store_true",
        help="Deprecated compatibility flag; warnings now continue automatically",
    )
    measurement_mode = parser.add_mutually_exclusive_group()
    measurement_mode.add_argument(
        "--measure",
        dest="measure",
        action="store_true",
        help="Run measurement, quality checks, and red-box annotation",
    )
    measurement_mode.add_argument(
        "--no-measure",
        dest="measure",
        action="store_false",
        help="Only crop/normalize and export vehicle.svg (default)",
    )
    parser.set_defaults(measure=False)
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.output / "pickup_measure.log", args.verbose)
    try:
        settings = load_settings(args.config)
        input_path = args.input or settings.input_path
        records = load_records(input_path, args.images)
    except Exception as exc:
        input_path = args.input or getattr(locals().get("settings"), "input_path", None)
        LOGGER.exception("Could not load input CSV/TSV: %s", input_path)
        terminal_print(f"❌ INPUT [{exc}]")
        return 2

    summary: list[dict[str, str]] = []
    total_records = len(records)
    for index, record in enumerate(records, start=1):
        progress_state = {"stage": "INITIALIZE"}
        terminal_print(f"[{index}/{total_records}] {record.id}")
        final_svg = args.output / record.size / f"{record.id}.svg"
        vehicle_svg = args.output / record.size / record.id / "vehicle.svg"
        annotation_points = (
            args.output / record.size / record.id / "annotation_points.json"
        )
        already_generated = (
            final_svg.is_file() and annotation_points.is_file()
            if args.measure
            else vehicle_svg.is_file()
        )
        if args.continue_mode and already_generated:
            LOGGER.info("%s: already generated; skipped by --continue", record.id)
            summary.append({
                "id": record.id,
                "size": record.size,
                "status": "SKIPPED",
                "error": "",
                "stage": "SKIPPED",
            })
            terminal_print("  ⏭️ SKIPPED [already generated]")
            continue
        try:
            status = process_vehicle(
                record,
                output_root=args.output,
                approve_warning=args.approve_warning,
                manual=args.manual,
                reuse_points=args.reuse_points,
                ppi=settings.output.ppi,
                annotation_style=settings.annotation,
                quality_settings=settings.quality,
                detection_settings=settings.detection,
                progress_state=progress_state,
                measure=args.measure,
            )
            status_error = progress_state.get("error", "")
            summary.append({
                "id": record.id,
                "size": record.size,
                "status": status,
                "error": status_error,
                "stage": progress_state["stage"],
            })
            if status != "EXPORTED":
                terminal_print(
                    f"  ❌ {progress_state['stage']} "
                    f"[{status_error or status}]"
                )
        except Exception as exc:
            LOGGER.error("%s: processing failed: %s", record.id, exc)
            LOGGER.debug("%s: processing traceback", record.id, exc_info=True)
            summary.append({
                "id": record.id,
                "size": record.size,
                "status": "FAILED",
                "error": str(exc),
                "stage": progress_state["stage"],
            })
            terminal_print(f"  ❌ {progress_state['stage']} [{exc}]")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    aggregate_measurements = args.output / "measurements.tsv"
    if args.measure:
        write_aggregate_measurements(
            args.output,
            records,
            included_ids={
                item["id"]
                for item in summary
                if item["status"] in {"EXPORTED", "SKIPPED"}
            },
        )
    else:
        aggregate_measurements.unlink(missing_ok=True)
    exported = sum(item["status"] == "EXPORTED" for item in summary)
    skipped = sum(item["status"] == "SKIPPED" for item in summary)
    completed = exported + skipped
    failed_items = [
        item
        for item in summary
        if item["status"] not in {"EXPORTED", "SKIPPED"}
    ]
    LOGGER.info(
        "Run complete: %d exported, %d skipped, %d/%d complete",
        exported,
        skipped,
        completed,
        len(summary),
    )
    terminal_print(
        f"Run complete: {exported} exported, {skipped} skipped, "
        f"{completed}/{len(summary)} complete, {len(failed_items)} failed"
    )
    if failed_items:
        terminal_print("Failed items:")
        for item in failed_items:
            terminal_print(
                f"  ❌ {item['id']} — {item.get('stage', 'UNKNOWN')} "
                f"[{item.get('error') or item['status']}]"
            )
    return 0 if completed == len(summary) else 1


if __name__ == "__main__":
    sys.exit(main())
