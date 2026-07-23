from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class QCStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class QCResult:
    name: str
    source_ratio: float
    target_ratio: float
    distortion: float
    status: QCStatus


def evaluate_ratio(
    name: str,
    source_width: float,
    source_height: float,
    length_mm: float,
    height_mm: float,
    pass_max_percent: float = 3.0,
    warning_max_percent: float = 6.0,
    error_max_percent: float | None = None,
) -> QCResult:
    if min(source_width, source_height, length_mm, height_mm) <= 0:
        raise ValueError("Source and target dimensions must be positive")
    if error_max_percent is None:
        error_max_percent = warning_max_percent
    if (
        pass_max_percent < 0
        or warning_max_percent < pass_max_percent
        or error_max_percent < warning_max_percent
    ):
        raise ValueError("Quality thresholds must be ordered and non-negative")
    source_ratio = source_width / source_height
    target_ratio = length_mm / height_mm
    distortion = abs(source_ratio - target_ratio) / target_ratio
    pass_limit = pass_max_percent / 100.0
    error_limit = error_max_percent / 100.0
    if distortion < pass_limit or math.isclose(
        distortion, pass_limit, rel_tol=0.0, abs_tol=1e-12
    ):
        status = QCStatus.PASS
    elif distortion < error_limit or math.isclose(
        distortion, error_limit, rel_tol=0.0, abs_tol=1e-12
    ):
        status = QCStatus.WARNING
    else:
        status = QCStatus.BLOCKED
    return QCResult(
        name=name,
        source_ratio=source_ratio,
        target_ratio=target_ratio,
        distortion=distortion,
        status=status,
    )
