import json

from PIL import Image

from pickup_measure.main import process_vehicle
from pickup_measure.src.geometry import Bounds
from pickup_measure.src.loader import VehicleRecord


def test_saved_points_produce_true_size_svg(tmp_path):
    image_path = tmp_path / "truck.png"
    Image.new("RGB", (320, 120), "gray").save(image_path)
    record = VehicleRecord("TRUCK_01", "Test Truck", image_path, 6000, 2000, 2000)
    item_dir = tmp_path / "output" / record.id
    Bounds(left=10, right=310, roof=10, ground=110).to_json(item_dir / "points.json")

    status = process_vehicle(
        record, tmp_path / "output", approve_warning=False, reuse_points=True, ppi=1
    )

    assert status == "EXPORTED"
    svg = (item_dir / "vehicle.svg").read_text(encoding="utf-8")
    assert 'width="6000mm"' in svg
    assert 'height="2000mm"' in svg
    assert (item_dir / "crop_source.png").exists()
    assert not (item_dir / "crop.png").exists()
    qc = json.loads((item_dir / "qc_report.json").read_text(encoding="utf-8"))
    assert qc["status"] == "PASS"


def test_warning_pauses_export(tmp_path):
    image_path = tmp_path / "truck.png"
    Image.new("RGB", (330, 100), "gray").save(image_path)
    record = VehicleRecord("TRUCK_02", "Warning Truck", image_path, 3000, 2000, 1000)
    item_dir = tmp_path / "output" / record.id
    Bounds(left=7, right=322, roof=0, ground=100).to_json(item_dir / "points.json")
    (item_dir / "vehicle.svg").write_text("stale", encoding="utf-8")

    status = process_vehicle(
        record, tmp_path / "output", approve_warning=False, reuse_points=True, ppi=1
    )

    assert status == "WARNING"
    assert not (item_dir / "vehicle.svg").exists()
