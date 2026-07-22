import pytest

from pickup_measure.src.quality import QCStatus, evaluate_ratio


@pytest.mark.parametrize(
    ("source_ratio", "expected"),
    [(3.0, QCStatus.PASS), (3.12, QCStatus.WARNING), (3.3, QCStatus.BLOCKED)],
)
def test_quality_thresholds(source_ratio, expected):
    result = evaluate_ratio("truck", source_ratio, 1, 3, 1)
    assert result.status is expected


def test_threshold_boundaries():
    assert evaluate_ratio("truck", 3.09, 1, 3, 1).status is QCStatus.PASS
    assert evaluate_ratio("truck", 3.18, 1, 3, 1).status is QCStatus.WARNING
