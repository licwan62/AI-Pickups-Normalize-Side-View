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
) -> QCResult:
    if min(source_width, source_height, length_mm, height_mm) <= 0:
        raise ValueError("Source and target dimensions must be positive")
    source_ratio = source_width / source_height
    target_ratio = length_mm / height_mm
    distortion = abs(source_ratio - target_ratio) / target_ratio
    if distortion < 0.03 or math.isclose(distortion, 0.03, rel_tol=0.0, abs_tol=1e-12):
        status = QCStatus.PASS
    elif distortion < 0.06 or math.isclose(distortion, 0.06, rel_tol=0.0, abs_tol=1e-12):
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
