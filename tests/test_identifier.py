import pytest

from pickup_measure.src.identifier import (
    QwenVehicleIdentifier,
    VehicleIdentity,
    _parse_year,
)


def test_identity_query_skips_body_style_and_empty_fields():
    identity = VehicleIdentity(
        make="Ford",
        model="F-150",
        trim="Raptor",
        year=2024,
        body_style="Pickup",
        cab="SuperCrew",
        bed="5.5",
    )

    assert identity.query == "Ford F-150 Raptor SuperCrew 5.5 2024"
    assert identity.as_dict()["query"] == identity.query


def test_identity_accepts_aliases_and_alternatives():
    identity = VehicleIdentity.from_payload(
        {
            "brand": "Toyota",
            "model": "Hilux",
            "trim": "TRD Pro",
            "year": "2016-2020",
            "cab_style": "Double Cab",
            "market": "eu",
            "confidence": 1.7,
            "alternatives": [
                {
                    "make": "Toyota",
                    "model": "Hilux",
                    "trim": "Invincible",
                    "year": 2016,
                },
                {"make": "Toyota"},
            ],
        }
    )

    assert identity.make == "Toyota"
    assert identity.year == 2016
    assert identity.cab == "Double Cab"
    assert identity.market == "EU"
    assert identity.confidence == 1.0
    assert len(identity.alternatives) == 1
    assert identity.alternatives[0].trim == "Invincible"


def test_identity_cleans_unknown_values():
    identity = VehicleIdentity.from_payload(
        {
            "make": " Isuzu ",
            "model": "D-Max",
            "trim": "unknown",
            "year": "n/a",
            "cab": "-",
            "confidence": -2,
        }
    )

    assert identity.trim == ""
    assert identity.year is None
    assert identity.cab == ""
    assert identity.confidence == 0.0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (2024, 2024),
        ("2024", 2024),
        ("Model year 2019-2022", 2019),
        ("1999.5", 1999),
        ("unknown", None),
        (None, None),
    ],
)
def test_parse_year(raw, expected):
    assert _parse_year(raw) == expected


def test_identity_requires_make_and_model():
    with pytest.raises(ValueError, match="missing make or model"):
        VehicleIdentity.from_payload({"make": "Ford"})
    with pytest.raises(ValueError, match="missing make or model"):
        VehicleIdentity.from_payload({"make": "Ford", "model": "unknown"})


def test_extract_json_accepts_fenced_and_explained_output():
    assert QwenVehicleIdentifier._extract_json(
        '```json\n{"make": "Ford", "model": "F-150"}\n```'
    ) == {"make": "Ford", "model": "F-150"}
    assert QwenVehicleIdentifier._extract_json(
        'Here is the result: {"make": "Ram", "model": "1500"} done.'
    ) == {"make": "Ram", "model": "1500"}
