import json

from PIL import Image

from pickup_measure.main import build_parser, main, process_vehicle
from pickup_measure.src.geometry import Bounds
from pickup_measure.src.loader import VehicleRecord
from pickup_measure.src.qwen_detector import QwenVehicleDetector
from pickup_measure.src.settings import DetectionSettings


def test_saved_points_produce_true_size_svg(tmp_path):
    image_path = tmp_path / "truck.png"
    Image.new("RGB", (320, 120), "gray").save(image_path)
    record = VehicleRecord("TRUCK_01", "Test Truck", image_path, 6000, 2000, 2000)
    item_dir = tmp_path / "output" / record.size / record.id
    Bounds(left=10, right=310, roof=10, ground=110).to_json(item_dir / "points.json")

    status = process_vehicle(
        record, tmp_path / "output", approve_warning=False, reuse_points=True, ppi=1
    )

    assert status == "EXPORTED"
    svg = (item_dir / "vehicle.svg").read_text(encoding="utf-8")
    assert 'width="6000mm"' in svg
    assert 'height="2000mm"' in svg
    assert (item_dir / "crop_source.png").exists()
    assert (item_dir / "orientation.json").exists()
    assert not (item_dir / "crop.png").exists()
    qc = json.loads((item_dir / "qc_report.json").read_text(encoding="utf-8"))
    assert qc["status"] == "PASS"


def test_warning_continues_to_dimension_annotation(tmp_path):
    image_path = tmp_path / "truck.png"
    Image.new("RGB", (330, 100), "gray").save(image_path)
    record = VehicleRecord("TRUCK_02", "Warning Truck", image_path, 3000, 2000, 1000)
    item_dir = tmp_path / "output" / record.size / record.id
    Bounds(left=7, right=322, roof=0, ground=100).to_json(item_dir / "points.json")
    (item_dir / "vehicle.svg").write_text("stale", encoding="utf-8")

    status = process_vehicle(
        record, tmp_path / "output", approve_warning=False, reuse_points=True, ppi=1
    )

    assert status == "EXPORTED"
    assert (item_dir / "vehicle.svg").exists()
    assert (item_dir.parent / f"{record.id}.svg").exists()
    assert not (item_dir / "annotated.svg").exists()
    assert (item_dir / "annotation_points.json").exists()
    qc = json.loads((item_dir / "qc_report.json").read_text(encoding="utf-8"))
    assert qc["status"] == "WARNING"
    assert qc["warning_auto_continued"] is True


def test_rejected_detection_preserves_attempt_crop_and_diagnostics(
    tmp_path,
    monkeypatch,
):
    image_path = tmp_path / "truck.png"
    Image.new("RGB", (320, 120), "gray").save(image_path)
    record = VehicleRecord("TRUCK_FAIL", "Failed Truck", image_path, 6000, 2000, 2000)
    attempted = Bounds(left=10, right=310, roof=10, ground=110)

    def reject(self, image, expected_aspect=None):
        self.last_attempt_bounds = attempted
        raise RuntimeError("low confidence")

    monkeypatch.setattr(QwenVehicleDetector, "detect", reject)
    monkeypatch.setattr(
        "pickup_measure.main.select_bounds",
        lambda image, window_title: attempted,
    )

    status = process_vehicle(
        record,
        tmp_path / "output",
        approve_warning=False,
        detection_settings=DetectionSettings(manual_fallback=True),
    )

    item_dir = tmp_path / "output" / record.size / record.id
    assert status == "EXPORTED"
    assert (item_dir / "crop_attempt.png").exists()
    assert (item_dir / "points.json").exists()
    diagnostics = json.loads(
        (item_dir / "detection_attempt.json").read_text(encoding="utf-8")
    )
    assert diagnostics["status"] == "MANUAL_FALLBACK"
    assert diagnostics["bounds"] == {
        "left": 10,
        "right": 310,
        "roof": 10,
        "ground": 110,
    }
    assert diagnostics["manual_bounds"] == diagnostics["bounds"]


def test_qwen_perspective_quad_is_applied_before_export(tmp_path, monkeypatch):
    image_path = tmp_path / "truck.png"
    Image.new("RGB", (200, 120), "gray").save(image_path)
    record = VehicleRecord(
        "TRUCK_PERSPECTIVE",
        "Perspective Truck",
        image_path,
        1600,
        800,
        920,
    )
    bounds = Bounds(left=20, right=180, roof=10, ground=110)
    quad = (
        (30.0, 20.0),
        (160.0, 10.0),
        (180.0, 100.0),
        (20.0, 110.0),
    )

    def detect(self, image, expected_aspect=None):
        self.last_attempt_bounds = bounds
        self.last_perspective_quad = quad
        self.last_response_content = "test response"
        self.last_front = "right"
        return bounds

    monkeypatch.setattr(QwenVehicleDetector, "detect", detect)
    monkeypatch.setattr(
        QwenVehicleDetector,
        "detect_parts",
        lambda self, image: (_ for _ in ()).throw(
            AssertionError("AI structure detection must not be called")
        ),
        raising=False,
    )

    status = process_vehicle(
        record,
        tmp_path / "output",
        approve_warning=False,
    )

    item_dir = tmp_path / "output" / record.size / record.id
    with Image.open(item_dir / "crop_source.png") as crop:
        assert crop.size == (197, 109)
    assert status == "EXPORTED"
    detection = json.loads(
        (item_dir / "detection.json").read_text(encoding="utf-8")
    )
    assert detection["perspective_quad"] == [list(point) for point in quad]
    assert "parts" not in detection
    annotation = json.loads(
        (item_dir / "annotation_points.json").read_text(encoding="utf-8")
    )
    assert annotation["is_pickup"] is True
    assert len(annotation["outline_segments_mm"]) == 3


def test_continue_skips_an_already_generated_id(tmp_path, capsys):
    image_path = tmp_path / "truck.png"
    Image.new("RGB", (320, 120), "gray").save(image_path)
    table = tmp_path / "vehicles.tsv"
    table.write_text(
        "name\tSize\timage_path\tlength_mm\twidth_mm\theight_mm\n"
        "Test Truck\tTEST\ttruck.png\t6000\t2000\t2000\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "output"
    vehicle_id = "Test_Truck_TEST_6000_2000_2000"
    item_dir = output / "TEST" / vehicle_id
    item_dir.mkdir(parents=True)
    final_svg = output / "TEST" / f"{vehicle_id}.svg"
    final_svg.write_text("keep existing SVG", encoding="utf-8")
    (item_dir / "annotation_points.json").write_text(
        json.dumps({
            "cab_height_mm": 1500,
            "hood_height_mm": 900,
            "neck_height_mm": 1000,
            "bed_height_mm": 1100,
        }),
        encoding="utf-8",
    )

    result = main([
        "--continue",
        "--measure",
        "--input", str(table),
        "--images", str(tmp_path),
        "--output", str(output),
        "--config", str(config),
    ])
    captured = capsys.readouterr()

    assert result == 0
    assert captured.err == ""
    assert f"[1/1] {vehicle_id}" in captured.out
    assert "⏭️ SKIPPED [already generated]" in captured.out
    assert captured.out.strip().endswith(
        "Run complete: 0 exported, 1 skipped, 1/1 complete, 0 failed"
    )
    assert final_svg.read_text(encoding="utf-8") == "keep existing SVG"
    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    assert summary[0]["status"] == "SKIPPED"
    measurements = (output / "measurements.tsv").read_text(
        encoding="utf-8-sig"
    )
    assert measurements.splitlines()[0] == (
        "NAME\tSIZE\tL-MM\tW-MM\tH-MM\t"
        "CAB-H\tHOOD-H\tNECK-H\tBED-H"
    )
    assert "Test Truck\tTEST\t6000\t2000\t2000\t1500\t900\t1000\t1100" in measurements


def test_continue_flag_uses_a_safe_attribute_name():
    args = build_parser().parse_args(["--continue"])

    assert args.continue_mode is True


def test_no_measure_is_the_default_cli_mode():
    parser = build_parser()

    assert parser.parse_args([]).measure is False
    assert parser.parse_args(["--no-measure"]).measure is False
    assert parser.parse_args(["--measure"]).measure is True


def test_no_measure_only_exports_crop_svg(tmp_path):
    image_path = tmp_path / "truck.png"
    Image.new("RGB", (320, 120), "gray").save(image_path)
    record = VehicleRecord("TRUCK_PLAIN", "Plain Truck", image_path, 6000, 2000, 2000)
    item_dir = tmp_path / "output" / record.size / record.id
    Bounds(left=10, right=310, roof=10, ground=110).to_json(item_dir / "points.json")

    status = process_vehicle(
        record,
        tmp_path / "output",
        approve_warning=False,
        reuse_points=True,
        measure=False,
    )

    assert status == "EXPORTED"
    svg = (item_dir / "vehicle.svg").read_text(encoding="utf-8")
    assert "LENGTH  6000" in svg
    assert "HEIGHT  2000" in svg
    assert "HOOD" not in svg
    assert "CAB" not in svg
    assert "NECK" not in svg
    assert "BED" not in svg
    assert "#C8242A" not in svg
    assert not (item_dir / "measurements.tsv").exists()
    assert not (item_dir / "annotation_points.json").exists()
    assert not (item_dir / "qc_report.json").exists()
    assert not (item_dir.parent / f"{record.id}.svg").exists()


def test_terminal_summary_names_failed_vehicle_and_stage(tmp_path, capsys):
    table = tmp_path / "vehicles.tsv"
    table.write_text(
        "name\tSize\timage_path\tlength_mm\twidth_mm\theight_mm\n"
        "Missing Truck\tTEST\tmissing.png\t6000\t2000\t2000\n",
        encoding="utf-8",
    )
    config = tmp_path / "config.yaml"
    config.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "output"
    vehicle_id = "Missing_Truck_TEST_6000_2000_2000"

    result = main([
        "--input", str(table),
        "--output", str(output),
        "--config", str(config),
    ])
    captured = capsys.readouterr()

    assert result == 1
    assert captured.err == ""
    assert f"[1/1] {vehicle_id}" in captured.out
    assert "❌ LOAD_IMAGE [Image not found:" in captured.out
    assert "Run complete: 0 exported, 0 skipped, 0/1 complete, 1 failed" in captured.out
    assert f"❌ {vehicle_id} — LOAD_IMAGE" in captured.out
