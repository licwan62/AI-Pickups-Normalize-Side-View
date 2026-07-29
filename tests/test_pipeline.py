import json

from PIL import Image

from pickup_measure.main import build_parser, main, process_vehicle
from pickup_measure.src.geometry import Bounds
from pickup_measure.src.loader import VehicleRecord


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


def test_continue_skips_an_already_generated_id(tmp_path):
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
        "--input", str(table),
        "--images", str(tmp_path),
        "--output", str(output),
        "--config", str(config),
    ])

    assert result == 0
    assert final_svg.read_text(encoding="utf-8") == "keep existing SVG"
    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    assert summary[0]["status"] == "SKIPPED"
    assert "Test Truck" in (output / "measurements.tsv").read_text(encoding="utf-8-sig")


def test_continue_flag_uses_a_safe_attribute_name():
    args = build_parser().parse_args(["--continue"])

    assert args.continue_mode is True
