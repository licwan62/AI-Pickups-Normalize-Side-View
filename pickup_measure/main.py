from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .src.background import classify_background
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
from .src.qwen_detector import QwenVehicleDetector
from .src.renderer import render_dimensioned_vehicle_svg
from .src.settings import AnnotationStyle, DetectionSettings, load_settings


LOGGER = logging.getLogger("pickup_measure")

HASH_MANIFEST_NAME = "image_generation_hashes.json"
CACHEABLE_FAILURE_PREFIXES = (
    "Qwen response validation failed after repair:",
    "Qwen response validation failed",
)


def image_sha256(image_path: Path) -> str:
    """Return the content hash used to decide whether an image is unchanged."""
    digest = hashlib.sha256()
    with image_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hash_manifest(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {"version": 1, "images": {}}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid image hash manifest: {path}") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("images"), dict):
        raise ValueError(f"Invalid image hash manifest structure: {path}")
    return manifest


def save_hash_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def manifest_image_key(record: VehicleRecord, manifest_dir: Path) -> str:
    """Return a relocatable image key relative to the manifest directory."""
    image_path = record.image_path.resolve()
    manifest_dir = manifest_dir.resolve()
    try:
        return Path(os.path.relpath(image_path, manifest_dir)).as_posix()
    except ValueError:
        # Windows cannot create a relative path across drive letters.
        return image_path.as_posix()


def manifest_hash_matches(
    manifest: dict[str, object], record: VehicleRecord, digest: str,
    manifest_dir: Path,
) -> bool:
    entry = manifest_image_entry(manifest, record, manifest_dir)
    return (
        isinstance(entry, dict)
        and entry.get("sha256") == digest
        and entry.get("record_id") == record.id
    )


def manifest_image_entry(
    manifest: dict[str, object], record: VehicleRecord, manifest_dir: Path
) -> dict[str, object] | None:
    images = manifest["images"]
    assert isinstance(images, dict)
    entry = images.get(manifest_image_key(record, manifest_dir))
    if isinstance(entry, dict):
        return entry

    # Read version-1 manifests that used an absolute path as the key. Matching
    # by record ID also keeps the cache useful after the repository is moved.
    legacy_entry = images.get(record.image_path.resolve().as_posix())
    if isinstance(legacy_entry, dict):
        return legacy_entry
    for candidate in images.values():
        if isinstance(candidate, dict) and candidate.get("record_id") == record.id:
            return candidate
    return None


def _remove_legacy_manifest_entries(
    images: dict[str, object], record: VehicleRecord, relative_key: str
) -> None:
    for key, entry in list(images.items()):
        if (
            key != relative_key
            and isinstance(entry, dict)
            and entry.get("record_id") == record.id
        ):
            del images[key]


def record_generated_hash(
    manifest: dict[str, object], record: VehicleRecord, digest: str,
    manifest_dir: Path,
) -> None:
    images = manifest["images"]
    assert isinstance(images, dict)
    relative_key = manifest_image_key(record, manifest_dir)
    _remove_legacy_manifest_entries(images, record, relative_key)
    images[relative_key] = {
        "sha256": digest,
        "record_id": record.id,
        "status": "EXPORTED",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def is_cacheable_failure(error: str) -> bool:
    """Cache deterministic Qwen validation failures, not transient API errors."""
    return error.startswith(CACHEABLE_FAILURE_PREFIXES)


def record_failed_hash(
    manifest: dict[str, object],
    record: VehicleRecord,
    digest: str,
    error: str,
    stage: str,
    manifest_dir: Path,
) -> None:
    images = manifest["images"]
    assert isinstance(images, dict)
    relative_key = manifest_image_key(record, manifest_dir)
    _remove_legacy_manifest_entries(images, record, relative_key)
    images[relative_key] = {
        "sha256": digest,
        "record_id": record.id,
        "status": "FAILED",
        "error": error,
        "stage": stage,
        "failed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


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


def process_vehicle(
    record: VehicleRecord,
    output_root: Path,
    manual: bool = False,
    reuse_points: bool = False,
    annotation_style: AnnotationStyle = AnnotationStyle(),
    detection_settings: DetectionSettings = DetectionSettings(),
    progress_state: dict[str, str] | None = None,
) -> str:
    process_started_at = time.perf_counter()

    def set_stage(stage: str) -> None:
        if progress_state is not None:
            progress_state["stage"] = stage

    size_output_dir = output_root / record.size
    output_dir = size_output_dir / record.id
    exporter = Exporter(output_dir)
    # A failed rerun must not leave a stale SVG that looks current.
    final_svg = size_output_dir / f"{record.id}.svg"
    final_svg.unlink(missing_ok=True)
    # Also clears files from earlier runs, including the removed --measure mode.
    for stale_name in (
        "vehicle.svg",
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
            saved_background_type = str(
                detection_payload.get("background_type", "unknown")
            ).lower()
            if saved_background_type in {
                "white",
                "transparent",
                "environment",
            }:
                api_background_type = saved_background_type
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
            "bounds": asdict(bounds),
            "perspective_quad": perspective_quad,
            "rectified_trim": rectified_trim,
            "boundary_touch_points": (
                boundary_touch_points
                if boundary_touch_points is not None
                else detection_payload.get("boundary_touch_points")
            ),
            "image_wheel_contacts": (
                wheel_contact_points
                if wheel_contact_points is not None
                else detection_payload.get("image_wheel_contacts")
            ),
            "body_chassis_line": (
                body_chassis_line
                if body_chassis_line is not None
                else detection_payload.get("body_chassis_line")
            ),
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

    set_stage("RENDER_EXPORT")
    render_dimensioned_vehicle_svg(
        image=crop,
        output_path=final_svg,
        width_mm=record.length_mm,
        height_mm=record.height_mm,
        style=annotation_style,
        model_name=record.name,
    )
    LOGGER.info(
        "%s: exported %s at %.1f mm x %.1f mm",
        record.id,
        final_svg,
        record.length_mm,
        record.height_mm,
    )
    terminal_stage_done(
        "RENDER_EXPORT",
        f"status=EXPORTED, "
        f"elapsed={time.perf_counter() - process_started_at:.1f}s",
    )
    return "EXPORTED"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate true-size pickup side-profile SVG files.")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input CSV/TSV file or directory of CSV files (defaults to input_csv/input_tsv in config.yaml)",
    )
    parser.add_argument(
        "--images", type=Path, default=Path("input/images"),
        help="Default image directory; filenames match vehicle IDs",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output directory (defaults to output_dir in config.yaml, or output)",
    )
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
        help=(
            "Skip unchanged image hashes with final output or a cached Qwen "
            "validation failure"
        ),
    )
    parser.add_argument(
        "--only-name",
        action="append",
        default=[],
        metavar="NAME",
        help="Process only the exact vehicle name; repeat to process multiple records",
    )
    parser.add_argument(
        "--sync-vehicles",
        action="store_true",
        help="Match images and update the input CSV file(s) before processing",
    )
    # Accepted so existing commands keep working; measurement was removed.
    parser.add_argument(
        "--approve-warning",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-measure",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = load_settings(args.config)
        args.output = args.output or settings.output_dir
        configure_logging(args.output / "pickup_measure.log", args.verbose)
        input_path = args.input or settings.input_path
        if args.sync_vehicles:
            if input_path.is_dir():
                sync_script = Path(__file__).resolve().parents[1] / "scripts" / "sync_project_csvs.py"
                subprocess.run(
                    [
                        sys.executable,
                        str(sync_script),
                        "--projects",
                        str(input_path),
                        "--apply",
                    ],
                    check=True,
                )
            else:
                sync_script = Path(__file__).resolve().parents[1] / "scripts" / "sync_vehicle_names.py"
                subprocess.run(
                    [sys.executable, str(sync_script), "--csv", str(input_path), "--apply"],
                    check=True,
                )
        records = load_records(input_path, args.images)
        if args.only_name:
            requested_names = set(args.only_name)
            records = [record for record in records if record.name in requested_names]
            missing_names = requested_names - {record.name for record in records}
            if missing_names:
                raise ValueError(f"Vehicle name not found: {', '.join(sorted(missing_names))}")
        manifest_path = (
            input_path if input_path.is_dir() else input_path.parent
        ) / HASH_MANIFEST_NAME
        hash_manifest = load_hash_manifest(manifest_path)
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
        output_exists = final_svg.is_file()
        if not record.image_path.is_file():
            message = f"image_path does not exist: {record.image_path}"
            LOGGER.warning("%s: %s; skipped", record.id, message)
            summary.append({
                "id": record.id,
                "size": record.size,
                "status": "SKIPPED",
                "error": message,
                "stage": "UNMATCHED_IMAGE",
            })
            terminal_print("  SKIPPED [unmatched image_path]")
            continue
        current_hash = image_sha256(record.image_path)
        cached_entry = manifest_image_entry(hash_manifest, record, manifest_path.parent)
        hash_matches = manifest_hash_matches(
            hash_manifest, record, current_hash, manifest_path.parent
        )
        cached_failure = (
            hash_matches
            and cached_entry is not None
            and cached_entry.get("status") == "FAILED"
        )
        already_generated = (
            output_exists
            and hash_matches
            and (cached_entry is None or cached_entry.get("status") != "FAILED")
        )
        if args.continue_mode and cached_failure:
            cached_error = str(cached_entry.get("error", "cached validation failure"))
            cached_stage = str(cached_entry.get("stage", "DETECT"))
            images = hash_manifest["images"]
            assert isinstance(images, dict)
            relative_key = manifest_image_key(record, manifest_path.parent)
            _remove_legacy_manifest_entries(images, record, relative_key)
            images[relative_key] = cached_entry
            save_hash_manifest(manifest_path, hash_manifest)
            LOGGER.warning(
                "%s: unchanged image hash has cached failure; skipped by --continue: %s",
                record.id,
                cached_error,
            )
            summary.append({
                "id": record.id,
                "size": record.size,
                "status": "CACHED_FAILURE",
                "error": cached_error,
                "stage": cached_stage,
            })
            terminal_print(f"  SKIPPED [cached failure: {cached_error}]")
            continue
        if args.continue_mode and already_generated:
            LOGGER.info("%s: unchanged image hash; skipped by --continue", record.id)
            record_generated_hash(
                hash_manifest, record, current_hash, manifest_path.parent
            )
            save_hash_manifest(manifest_path, hash_manifest)
            summary.append({
                "id": record.id,
                "size": record.size,
                "status": "SKIPPED",
                "error": "",
                "stage": "SKIPPED",
            })
            terminal_print("  ⏭️ SKIPPED [unchanged image hash]")
            continue
        try:
            status = process_vehicle(
                record,
                output_root=args.output,
                manual=args.manual,
                reuse_points=args.reuse_points,
                annotation_style=settings.annotation,
                detection_settings=settings.detection,
                progress_state=progress_state,
            )
            status_error = progress_state.get("error", "")
            summary.append({
                "id": record.id,
                "size": record.size,
                "status": status,
                "error": status_error,
                "stage": progress_state["stage"],
            })
            if status == "EXPORTED":
                record_generated_hash(
                    hash_manifest, record, current_hash, manifest_path.parent
                )
                save_hash_manifest(manifest_path, hash_manifest)
            if status != "EXPORTED":
                terminal_print(
                    f"  ❌ {progress_state['stage']} "
                    f"[{status_error or status}]"
                )
        except Exception as exc:
            LOGGER.error("%s: processing failed: %s", record.id, exc)
            LOGGER.debug("%s: processing traceback", record.id, exc_info=True)
            error = str(exc)
            summary.append({
                "id": record.id,
                "size": record.size,
                "status": "FAILED",
                "error": error,
                "stage": progress_state["stage"],
            })
            if is_cacheable_failure(error):
                record_failed_hash(
                    hash_manifest,
                    record,
                    current_hash,
                    error,
                    progress_state["stage"],
                    manifest_path.parent,
                )
                save_hash_manifest(manifest_path, hash_manifest)
            terminal_print(f"  ❌ {progress_state['stage']} [{exc}]")

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output / "measurements.tsv").unlink(missing_ok=True)
    exported = sum(item["status"] == "EXPORTED" for item in summary)
    skipped = sum(
        item["status"] in {"SKIPPED", "CACHED_FAILURE"} for item in summary
    )
    completed = exported + sum(item["status"] == "SKIPPED" for item in summary)
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
