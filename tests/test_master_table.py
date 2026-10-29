import pandas as pd
import pytest

from pickup_measure.src.master_table import (
    MasterTable,
    MasterVehicle,
    display_name,
    load_master_table,
    market_from_id,
    query_market,
)
from pickup_measure.src.settings import MasterTableSettings


def _vehicle(
    dimension_id: str,
    *,
    length: int = 5908,
    width: int = 2200,
    height: int = 2027,
    market: str = "US",
    sales: int = 0,
    trim: str = "Raptor",
    year: str = "2021-2026",
) -> MasterVehicle:
    return MasterVehicle(
        dimension_id=dimension_id,
        name=display_name(dimension_id),
        make="Ford",
        model="F-150",
        trim=trim,
        year=year,
        size="PK-XL",
        length_mm=length,
        width_mm=width,
        height_mm=height,
        structure="Pickup",
        market=market,
        sales=sales,
    )


def _table(*vehicles: MasterVehicle) -> MasterTable:
    return MasterTable(vehicles)


def test_exact_dimension_id_match_is_flagged():
    vehicle = _vehicle("Ford F-150 Raptor Pickup 2021-2026 SuperCrew 5.5 US")
    result = _table(vehicle).find(vehicle.dimension_id)

    assert result.ok
    assert result.best is not None
    assert result.best.vehicle is vehicle
    assert result.best.exact
    assert result.best.score == 1.0


def test_trim_specific_row_outranks_generic_row_of_same_model():
    generic = _vehicle(
        "Ford F-150 Pickup 2024-2026 SuperCrew 5.5 US",
        length=5893,
        height=1961,
        trim="",
        sales=10000,
    )
    raptor = _vehicle("Ford F-150 Raptor Pickup 2021-2026 SuperCrew 5.5 US")
    result = _table(generic, raptor).find(
        "Ford F-150 Raptor SuperCrew 5.5 2024", market="US"
    )

    assert result.ok
    assert result.best is not None
    assert result.best.vehicle.dimension_id == raptor.dimension_id


def test_width_only_difference_is_not_a_blocking_ambiguity():
    us = _vehicle("Ford F-150 Raptor Pickup 2021-2026 SuperCrew 5.5 US", width=2200)
    ru = _vehicle(
        "Ford F-150 Raptor Pickup 2024-2026 SuperCrew 5.5 RU",
        width=2121,
        market="RU",
    )
    result = _table(us, ru).find("Ford F-150 Raptor SuperCrew 5.5 2024")

    assert result.ok
    assert not result.ambiguous
    assert result.best is not None
    assert result.best.vehicle.market == "US"


def test_different_length_or_height_is_ambiguous():
    us = _vehicle("Ford F-150 Raptor Pickup 2021-2026 SuperCrew 5.5 US")
    eu = _vehicle(
        "Ford F-150 Raptor Pickup 2021-2026 SuperCrew 5.5 EU",
        length=5895,
        height=1991,
        market="EU",
    )
    result = _table(us, eu).find("Ford F-150 Raptor SuperCrew 5.5 2024")

    assert result.ambiguous
    assert not result.ok
    assert result.reason == "multiple length/height combinations"


def test_market_filter_limits_candidates():
    us = _vehicle("Ford F-150 Raptor Pickup 2021-2026 SuperCrew 5.5 US")
    eu = _vehicle(
        "Ford F-150 Raptor Pickup 2021-2026 SuperCrew 5.5 EU",
        market="EU",
    )
    result = _table(us, eu).find(
        "Ford F-150 Raptor SuperCrew 5.5", market="EU"
    )

    assert result.ok
    assert result.best is not None
    assert result.best.vehicle.market == "EU"


def test_compact_model_token_matches_hyphenated_name():
    vehicle = _vehicle("Ford F-150 Raptor Pickup 2021-2026 SuperCrew 5.5 US")
    result = _table(vehicle).find("F150 Raptor", market="US")

    assert result.ok
    assert result.best is not None
    assert result.best.vehicle is vehicle


def test_unrelated_query_is_reported_without_match():
    vehicle = _vehicle("Ford F-150 Raptor Pickup 2021-2026 SuperCrew 5.5 US")
    result = _table(vehicle).find("Toyota Hilux", market="US")

    assert not result.ok
    assert result.best is None
    assert "no US candidate" in result.reason


def test_query_market_and_market_from_id():
    assert market_from_id("Ford F-150 Raptor Pickup 2021-2026 US") == "US"
    assert query_market("Ford_F-150_US_2024") == "US"
    assert query_market("Isuzu D-Max") == ""
    assert query_market("MUSSO") == ""


MASTER_CSV = (
    "DIMENSION-ID,MAKE,MODEL,TRIM,YEAR,SIZE,结构,L-MM,W-MM,H-MM,车型销量\n"
    "Ford F-150 Raptor Pickup 2021-2026 SuperCrew 5.5 US,Ford,F-150,Raptor,"
    "2021-2026,PK-XL,Pickup,5908,2200,2027,100\n"
    "Ford F-150 Pickup 2024-2026 SuperCrew 5.5 US,Ford,F-150,,2024-2026,"
    "PK-XL,Pickup,5893,2029,1961,10000\n"
)


def test_load_master_table_builds_cache_and_reuses_it(tmp_path):
    source = tmp_path / "全量表.csv"
    source.write_text(MASTER_CSV, encoding="utf-8-sig")
    settings = MasterTableSettings(
        source=source,
        cache_path=tmp_path / "cache" / "master.csv",
    )

    table = load_master_table(settings)

    assert len(table) == 2
    assert settings.cache_path.is_file()
    assert settings.cache_path.with_name("master.csv.meta.json").is_file()

    frame = pd.read_csv(settings.cache_path, dtype=str, keep_default_na=False)
    frame.head(1).to_csv(settings.cache_path, index=False, encoding="utf-8-sig")
    reused = load_master_table(settings)
    assert len(reused) == 1

    rebuilt = load_master_table(settings, force_refresh=True)
    assert len(rebuilt) == 2


def test_load_master_table_decodes_utf16_source(tmp_path):
    source = tmp_path / "全量表.csv"
    source.write_bytes(MASTER_CSV.encode("utf-16"))
    settings = MasterTableSettings(source=source, cache_path=tmp_path / "cache.csv")

    table = load_master_table(settings)

    assert len(table) == 2
    assert table.vehicles[0].size == "PK-XL"
    assert table.vehicles[0].market == "US"


def test_load_master_table_requires_dimension_id_column(tmp_path):
    source = tmp_path / "全量表.csv"
    source.write_text("MAKE,MODEL,L-MM,W-MM,H-MM\nFord,F-150,5908,2200,2027\n")

    with pytest.raises(ValueError, match="DIMENSION-ID"):
        load_master_table(
            MasterTableSettings(
                source=source,
                cache_path=tmp_path / "cache.csv",
            )
        )
