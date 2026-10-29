import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from pickup_measure.src.identifier import VehicleIdentity
from pickup_measure.src.settings import MasterTableSettings, load_settings
from pickup_measure.src.side_view import run_side_view


ROOT = Path(__file__).resolve().parents[1]

CSV_HEADER = "DIMENSION-ID,MAKE,MODEL,TRIM,YEAR,SIZE,结构,L-MM,W-MM,H-MM,车型销量\n"
FORD_ROW = (
    "Ford F-150 Raptor Pickup 2021-2026 SuperCrew 5.5 US,"
    "Ford,F-150,Raptor,2021-2026,PK-XL,Pickup,5908,2200,2027,100\n"
)
SECOND_ROW = (
    "Ford F-150 Raptor Pickup 2021-2026 SuperCrew 5.5 2 US,"
    "Ford,F-150,Raptor,2021-2026,PK-XL,Pickup,6195,2200,2027,50\n"
)


def _prepare(tmp_path: Path, rows: str) -> tuple[Path, object]:
    source = tmp_path / "全量表.csv"
    source.write_text(CSV_HEADER + rows, encoding="utf-8-sig")
    settings = replace(
        load_settings(ROOT / "config.yaml"),
        master_table=MasterTableSettings(
            source=source,
            cache_path=tmp_path / "cache" / "master.csv",
            refresh="never",
        ),
    )
    image_path = tmp_path / "Ford_F-150_Raptor_US.png"
    Image.new("RGB", (480, 270), (180, 180, 180)).save(image_path)
    return image_path, settings


def _identifier(make: str, model: str, trim: str = "Raptor", year: int = 2024):
    identity = VehicleIdentity(
        make=make,
        model=model,
        trim=trim,
        year=year,
        cab="SuperCrew",
        bed="5.5",
        confidence=0.9,
    )
    return SimpleNamespace(identify=lambda image: identity)


def test_run_side_view_matches_and_writes_payload(tmp_path):
    image_path, settings = _prepare(tmp_path, FORD_ROW)

    result = run_side_view(
        image_path,
        settings,
        tmp_path / "out",
        identifier=_identifier("Ford", "F-150"),
        render=False,
    )

    assert result.status == "MATCHED"
    assert result.selected is not None
    assert result.selected.vehicle.dimension_id.startswith("Ford F-150 Raptor")
    assert result.match_path is not None

    payload = json.loads(result.match_path.read_text(encoding="utf-8"))
    assert payload["status"] == "MATCHED"
    assert payload["market_filter"] == "US"
    assert payload["matched"]["length_mm"] == 5908
    assert payload["identity"]["query"] == "Ford F-150 Raptor SuperCrew 5.5 2024"


def test_run_side_view_reports_ambiguity_without_rendering(tmp_path):
    image_path, settings = _prepare(tmp_path, FORD_ROW + SECOND_ROW)

    result = run_side_view(
        image_path,
        settings,
        tmp_path / "out",
        identifier=_identifier("Ford", "F-150"),
        render=False,
    )

    assert result.status == "AMBIGUOUS"
    assert result.selected is not None
    assert result.match_path is not None
    payload = json.loads(result.match_path.read_text(encoding="utf-8"))
    assert payload["ambiguous"] is True
    assert payload["reason"] == "multiple length/height combinations"
    assert len(payload["candidates"]) == 2


def test_run_side_view_reports_no_match(tmp_path):
    image_path, settings = _prepare(tmp_path, FORD_ROW)

    result = run_side_view(
        image_path,
        settings,
        tmp_path / "out",
        identifier=_identifier("Toyota", "Hilux", trim="", year=2020),
        render=False,
    )

    assert result.status == "NO_MATCH"
    assert result.match_path is not None
    payload = json.loads(result.match_path.read_text(encoding="utf-8"))
    assert payload["status"] == "NO_MATCH"
    assert "below 0.62" in payload["reason"]


def test_run_side_view_reports_identification_failure(tmp_path):
    image_path, settings = _prepare(tmp_path, FORD_ROW)

    def fail(image):
        raise RuntimeError("Qwen API is unavailable")

    result = run_side_view(
        image_path,
        settings,
        tmp_path / "out",
        identifier=SimpleNamespace(identify=fail),
        render=False,
    )

    assert result.status == "FAILED"
    assert "Qwen API is unavailable" in result.error
    assert result.match_path is not None
    payload = json.loads(result.match_path.read_text(encoding="utf-8"))
    assert payload["status"] == "FAILED"
    assert payload["identity"] is None
    assert payload["market_filter"] == "US"


def test_run_side_view_reports_master_table_failure(tmp_path):
    image_path, settings = _prepare(tmp_path, FORD_ROW)
    settings = replace(
        settings,
        master_table=replace(
            settings.master_table,
            source=tmp_path / "missing",
            cache_path=tmp_path / "missing.csv",
        ),
    )

    result = run_side_view(
        image_path,
        settings,
        tmp_path / "out",
        identifier=_identifier("Ford", "F-150"),
        render=False,
    )

    assert result.status == "FAILED"
    assert result.match_path is not None
    payload = json.loads(result.match_path.read_text(encoding="utf-8"))
    assert payload["status"] == "FAILED"
    assert "No master table found" in payload["error"]
