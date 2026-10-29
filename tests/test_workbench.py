import io
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import replace
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from PIL import Image

from pickup_measure.main import process_vehicle
from pickup_measure.src.geometry import Bounds
from pickup_measure.src.identifier import VehicleIdentity
from pickup_measure.src.master_table import MasterTable, MasterVehicle
from pickup_measure.workbench import Workbench, WorkbenchError, make_handler


def vehicle(**kwargs):
    return MasterVehicle(**{
        "dimension_id": "Ford F-150 Raptor Pickup 2021-2026 SuperCrew 5.5 US",
        "name": "Ford F-150 Raptor Pickup 2021-2026 SuperCrew 5.5",
        "make": "Ford", "model": "F-150", "trim": "Raptor", "year": "2021-2026",
        "size": "PK-XL", "length_mm": 5908, "width_mm": 2200, "height_mm": 2027,
        "market": "US", "source": "US/全量表.csv", **kwargs,
    })


def image_bytes(color="gray"):
    stream = io.BytesIO()
    Image.new("RGB", (320, 120), color).save(stream, "PNG")
    return stream.getvalue()


@pytest.fixture
def wb(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("images_dir: img/input\noutput_dir: img/output\n", encoding="utf-8")
    app = Workbench(config, table=MasterTable([vehicle()]))
    yield app
    app.close()


def drain(app):
    app.executor.submit(lambda: None).result(timeout=10)


def upload(app, color="gray", name=None):
    result = app.upload(name or vehicle().name + ".png", image_bytes(color))
    drain(app)
    return app.tasks[result["id"]]


def confirm(app, task):
    app.confirm(task["id"], task["candidates"][0]["row_key"])


def test_upload_matches_without_ai_and_requires_confirmation(wb):
    def no_ai(image):
        raise AssertionError("Exact filenames must not call AI")
    wb.identifier = SimpleNamespace(identify=no_ai)
    task = upload(wb)
    assert task["status"] == "NEEDS_REVIEW"
    assert task["match_method"] == "filename"
    assert not task["confirmed"]
    assert task["candidates"][0]["width_mm"] == 2200
    with pytest.raises(WorkbenchError, match="核对"):
        wb.queue_export([task["id"]])
    confirm(wb, task)
    assert task["status"] == "READY"
    assert task["selected"]["dimension_id"] == vehicle().dimension_id
    assert wb.upload("duplicate.png", image_bytes())["id"] == task["id"]
    assert len(wb.tasks) == 1
    state = wb.snapshot()
    assert "image_path" not in state["tasks"][0]
    assert "api_key" not in state
    assert state["api_key_configured"] is False


def test_uninformative_filename_uses_identity_but_not_ai_dimensions(wb):
    wb.identifier = SimpleNamespace(identify=lambda image: VehicleIdentity(
        "Ford", "F-150", "Raptor", year=2024, cab="SuperCrew", bed="5.5", confidence=.9))
    task = upload(wb, name="IMG_0001.png")
    assert task["match_method"] == "qwen"
    assert task["identity"]["confidence"] == .9
    assert task["candidates"][0]["length_mm"] == 5908


def test_ai_failure_still_allows_manual_search_and_retry(wb):
    def fail(image):
        raise RuntimeError("API unavailable")
    wb.identifier = SimpleNamespace(identify=fail)
    task = upload(wb, name="unknown.png")
    assert "API unavailable" in task["warning"]
    assert task["status"] == "NEEDS_REVIEW"
    wb.queue_match(task["id"], query="Ford F-150 Raptor 2024", market="US")
    drain(wb)
    assert task["candidates"]
    assert task["warning"] == ""
    confirm(wb, task)


def test_width_ambiguity_and_forged_or_stale_selection(wb):
    wb.table = MasterTable([vehicle(), vehicle(
        dimension_id=vehicle().dimension_id.replace("US", "EU"), width_mm=2100, market="EU")])
    task = upload(wb)
    assert task["ambiguous"] is True
    with pytest.raises(WorkbenchError, match="候选"):
        wb.confirm(task["id"], "forged")
    wb.table = MasterTable([vehicle(width_mm=2300)])
    with pytest.raises(WorkbenchError, match="变化"):
        confirm(wb, task)


def test_explicit_market_filters_even_an_exact_dimension_id(wb):
    eu = vehicle(dimension_id=vehicle().dimension_id.replace("US", "EU"), market="EU", width_mm=2100)
    wb.table = MasterTable([vehicle(), eu])
    task = upload(wb, name=vehicle().dimension_id + ".png")
    wb.queue_match(task["id"], query=vehicle().dimension_id, market="EU")
    drain(wb)
    assert len(task["candidates"]) == 1
    assert task["candidates"][0]["market"] == "EU"


def test_real_pipeline_exports_millimeter_svg_with_audit_and_isolated_outputs(wb):
    # Reuse known bounds to exercise the real image pipeline without a paid API.
    def processor(record, output_root, **kwargs):
        bounds = output_root / record.size / record.id / "points.json"
        Bounds(left=10, right=310, roof=10, ground=110).to_json(bounds)
        return process_vehicle(record, output_root, reuse_points=True, **kwargs)
    wb.processor = processor
    tasks = [upload(wb, color) for color in ["gray", "white"]]
    for task in tasks:
        confirm(wb, task)
    wb.queue_export([t["id"] for t in tasks])
    drain(wb)
    for task in tasks:
        assert task["status"] == "EXPORTED", task["error"]
        path, _ = wb.asset(task["id"], "svg")
        root = ET.parse(path).getroot()
        assert root.attrib["width"].endswith("mm")
        picture = next(e for e in root if e.tag.endswith("image"))
        assert float(picture.attrib["width"]) == 5908
        assert float(picture.attrib["height"]) == 2027
        assert picture.attrib["preserveAspectRatio"] == "none"
        audit_path, _ = wb.asset(task["id"], "review")
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        assert audit["sha256"] == task["sha256"]
        assert audit["selected"]["width_mm"] == 2200
    assert tasks[0]["svg_path"] != tasks[1]["svg_path"]


def test_source_changes_invalidate_confirmation_and_hide_previous_svg(wb):
    task = upload(wb)
    confirm(wb, task)
    from pathlib import Path
    Path(task["image_path"]).write_bytes(image_bytes("white"))
    wb.queue_export([task["id"]])
    drain(wb)
    assert task["status"] == "FAILED"
    assert not task["confirmed"]
    assert "原图已变化" in task["error"]
    with pytest.raises(WorkbenchError):
        wb.asset(task["id"], "svg")


def test_export_failure_can_retry_but_rematch_revokes_confirmation(wb):
    task = upload(wb)
    confirm(wb, task)
    def fail(*args, **kwargs):
        raise RuntimeError("timeout")
    wb.processor = fail
    wb.queue_export([task["id"]])
    drain(wb)
    assert task["status"] == "FAILED"
    assert task["confirmed"]
    wb.queue_match(task["id"], query=vehicle().name)
    drain(wb)
    assert not task["confirmed"]


def test_persistence_and_interrupted_tasks_recover(wb):
    task = upload(wb)
    confirm(wb, task)
    wb._update(task, status="PROCESSING")
    reopened = Workbench(wb.config, table=wb.table)
    try:
        recovered = reopened.tasks[task["id"]]
        assert recovered["status"] == "FAILED"
        assert recovered["stage"] == "INTERRUPTED"
        assert recovered["selected"]["width_mm"] == 2200
        assert recovered["confirmed"]
    finally:
        reopened.close()


def test_queue_rejects_duplicates_while_working_and_validates_entire_batch(wb):
    task = upload(wb)
    confirm(wb, task)
    other = upload(wb, "white")
    with pytest.raises(WorkbenchError):
        wb.queue_export([task["id"], other["id"]])
    assert task["status"] == "READY"
    wb._update(task, status="EXPORT_QUEUED")
    with pytest.raises(WorkbenchError, match="排队"):
        wb.queue_match(task["id"])


def test_invalid_uploads_and_scan_leave_originals_unchanged(wb):
    with pytest.raises(WorkbenchError):
        wb.upload("x.svg", b"<svg/>")
    with pytest.raises(Exception):
        wb.upload("bad.png", b"not an image")
    folder = wb.images_dir / "PK-XL"
    folder.mkdir(parents=True)
    path = folder / "source.png"
    raw = image_bytes()
    path.write_bytes(raw)
    assert wb.scan()["added"] == 1
    assert wb.scan()["added"] == 0
    assert path.read_bytes() == raw
    assert next(iter(wb.tasks.values()))["status"] == "NEW"


def test_http_serves_single_page_and_rejects_foreign_origins(wb):
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(wb))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base) as response:
            assert b"<!doctype html>" in response.read()
        request = urllib.request.Request(base + "/api/upload", data=image_bytes(),
                                         headers={"X-Filename": "photo.png", "Origin": "https://foreign.example"})
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 403
        with urllib.request.urlopen(base + "/api/state") as response:
            assert json.load(response)["tasks"] == []
        request = urllib.request.Request(base + "/api/upload", data=image_bytes(),
            headers={"X-Filename": urllib.parse.quote(vehicle().name + ".png"), "Origin": base})
        with urllib.request.urlopen(request) as response:
            task_id = json.load(response)["id"]
        drain(wb)
        with urllib.request.urlopen(base + "/api/state") as response:
            task = json.load(response)["tasks"][0]
        assert task["status"] == "NEEDS_REVIEW"
        assert task["id"] == task_id
        payload = json.dumps({"id": task_id, "row_key": task["candidates"][0]["row_key"]}).encode()
        request = urllib.request.Request(base + "/api/confirm", data=payload,
            headers={"Content-Type": "application/json", "Origin": base})
        with urllib.request.urlopen(request) as response:
            assert json.load(response)["ok"]
        with urllib.request.urlopen(base + f"/api/assets/{task_id}/thumbnail") as response:
            assert response.headers["Content-Type"] == "image/jpeg"
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(base + f"/api/assets/{task_id}/svg")
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_source_table_changes_are_detected_without_manual_refresh(tmp_path):
    source = tmp_path / "master.csv"
    header = "DIMENSION-ID,MAKE,MODEL,YEAR,SIZE,L-MM,W-MM,H-MM\n"
    row = f"{vehicle().dimension_id},Ford,F-150,2021-2026,PK-XL,5908,2200,2027\n"
    source.write_text(header + row, encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("master_table:\n  source: master.csv\n  cache_path: cache/master.csv\n", encoding="utf-8")
    app = Workbench(config)
    try:
        task = upload(app)
        confirm(app, task)
        source.write_text(header + row.replace("2200", "2300"), encoding="utf-8")
        app.queue_export([task["id"]])
        drain(app)
        assert task["status"] == "FAILED"
        assert not task["confirmed"]
        assert "尺寸已变化" in task["error"]
    finally:
        app.close()


def test_unavailable_source_uses_local_snapshot_and_then_its_cache(tmp_path):
    local = tmp_path / "data/dimensions/全量表_汇总.csv"
    local.parent.mkdir(parents=True)
    local.write_text("DIMENSION-ID,L-MM,W-MM,H-MM\nFord Test US,5000,2000,1800\n", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text("master_table:\n  source: missing-source\n  cache_path: cache/master.csv\n", encoding="utf-8")
    app = Workbench(config)
    try:
        table = app._get_table()
        assert len(table) == 1
        assert app.master["mode"] == "local"
        assert app.master["source"] == str(local)
        assert not (tmp_path / "cache/master.csv").exists()
        assert len(app._get_table(check_source=True)) == 1
    finally:
        app.close()
