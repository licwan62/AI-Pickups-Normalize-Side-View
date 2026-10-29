"""Single-image side-view pipeline: identify, match the master table, render."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from .identifier import QwenVehicleIdentifier, VehicleIdentity
from .loader import VehicleRecord, generate_vehicle_id, load_image, sanitize_component
from .master_table import (
    MasterTable,
    MasterVehicle,
    MatchCandidate,
    MatchResult,
    load_master_table,
    query_market,
)
from .settings import Settings


LOGGER = logging.getLogger("pickup_measure.side_view")


@dataclass(frozen=True)
class SideViewResult:
    image_path: Path
    status: str
    query: str = ""
    identity: VehicleIdentity | None = None
    match: MatchResult | None = None
    selected: MatchCandidate | None = None
    svg_path: Path | None = None
    match_path: Path | None = None
    error: str = ""
    warning: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "image": self.image_path.as_posix(),
            "query": self.query,
            "dimension_id": (
                self.selected.vehicle.dimension_id if self.selected else ""
            ),
            "svg": self.svg_path.as_posix() if self.svg_path else "",
            "match_json": self.match_path.as_posix() if self.match_path else "",
            "error": self.error,
            "warning": self.warning,
        }


def build_vehicle_record(
    vehicle: MasterVehicle,
    image_path: Path,
) -> VehicleRecord:
    """Convert a master-table row into the batch pipeline's record type."""

    return VehicleRecord(
        id=generate_vehicle_id(vehicle.dimension_id),
        name=vehicle.name,
        image_path=image_path,
        length_mm=float(vehicle.length_mm),
        width_mm=float(vehicle.width_mm),
        height_mm=float(vehicle.height_mm),
        size=vehicle.size,
    )


def _match_path(output_root: Path, image_path: Path) -> Path:
    folder = sanitize_component(image_path.stem)
    return output_root / "side_view" / folder / "match.json"


def _payload(
    image_path: Path,
    identity: VehicleIdentity | None,
    query: str,
    market: str,
    match: MatchResult | None,
    selected: MatchCandidate | None,
    status: str,
    error: str = "",
    warning: str = "",
    svg_path: Path | None = None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": status,
        "image": image_path.resolve().as_posix(),
        "identity": identity.as_dict() if identity else None,
        "query": query,
        "market_filter": market,
        "matched": selected.as_dict() if selected else None,
        "ambiguous": bool(match.ambiguous) if match else False,
        "reason": match.reason if match else "",
        "threshold": match.threshold if match else 0.0,
        "candidates": (
            [candidate.as_dict() for candidate in match.candidates[:12]]
            if match
            else []
        ),
        "svg": svg_path.as_posix() if svg_path else "",
        "error": error,
        "warning": warning,
    }


def _write_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def run_side_view(
    image_path: Path,
    settings: Settings,
    output_root: Path,
    *,
    identifier: QwenVehicleIdentifier | None = None,
    table: MasterTable | None = None,
    market: str | None = None,
    allow_ambiguous: bool = False,
    force_refresh: bool = False,
    render: bool = True,
) -> SideViewResult:
    """Identify one image, match it, and optionally render the dimensioned SVG."""

    image_path = image_path.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    output_root = output_root.resolve()
    match_path = _match_path(output_root, image_path)

    identifier = identifier or QwenVehicleIdentifier.from_settings(settings)
    if table is None:
        try:
            table = load_master_table(
                settings.master_table,
                force_refresh=force_refresh,
            )
        except Exception as exc:
            LOGGER.exception("Could not load the master table")
            market_hint = (
                str(market or "").strip().upper() or query_market(image_path.stem)
            )
            _write_payload(
                match_path,
                _payload(
                    image_path,
                    None,
                    "",
                    market_hint,
                    None,
                    None,
                    "FAILED",
                    error=str(exc),
                ),
            )
            return SideViewResult(
                image_path=image_path,
                status="FAILED",
                match_path=match_path,
                error=str(exc),
            )

    try:
        identity = identifier.identify(load_image(image_path))
    except Exception as exc:
        LOGGER.exception("%s: vehicle identification failed", image_path)
        market_hint = (
            str(market or "").strip().upper() or query_market(image_path.stem)
        )
        _write_payload(
            match_path,
            _payload(
                image_path,
                None,
                "",
                market_hint,
                None,
                None,
                "FAILED",
                error=str(exc),
            ),
        )
        return SideViewResult(
            image_path=image_path,
            status="FAILED",
            match_path=match_path,
            error=str(exc),
        )

    query = identity.query
    market_hint = (
        str(market or "").strip().upper()
        or query_market(image_path.stem)
        or identity.market
    )
    match = table.find(
        query,
        market=market_hint or None,
        min_score=settings.master_table.min_score,
    )
    warning = ""
    if identity.confidence and identity.confidence < 0.35:
        warning = f"low identification confidence {identity.confidence:.2f}"

    if match.best is None or match.best.score < settings.master_table.min_score:
        _write_payload(
            match_path,
            _payload(
                image_path,
                identity,
                query,
                market_hint,
                match,
                None,
                "NO_MATCH",
                error=match.reason,
                warning=warning,
            ),
        )
        return SideViewResult(
            image_path=image_path,
            status="NO_MATCH",
            query=query,
            identity=identity,
            match=match,
            match_path=match_path,
            error=match.reason,
            warning=warning,
        )

    selected = match.best
    if match.ambiguous and not allow_ambiguous:
        _write_payload(
            match_path,
            _payload(
                image_path,
                identity,
                query,
                market_hint,
                match,
                selected,
                "AMBIGUOUS",
                error=match.reason,
                warning=warning,
            ),
        )
        return SideViewResult(
            image_path=image_path,
            status="AMBIGUOUS",
            query=query,
            identity=identity,
            match=match,
            selected=selected,
            match_path=match_path,
            error=match.reason,
            warning=warning,
        )
    if match.ambiguous and warning:
        warning = f"{warning}; {match.reason}"
    elif match.ambiguous:
        warning = match.reason

    record = build_vehicle_record(selected.vehicle, image_path)
    if not render:
        _write_payload(
            match_path,
            _payload(
                image_path,
                identity,
                query,
                market_hint,
                match,
                selected,
                "MATCHED",
                warning=warning,
            ),
        )
        return SideViewResult(
            image_path=image_path,
            status="MATCHED",
            query=query,
            identity=identity,
            match=match,
            selected=selected,
            match_path=match_path,
            warning=warning,
        )

    # Imported lazily because main.py owns the batch CLI and imports this module.
    from ..main import process_vehicle

    try:
        status = process_vehicle(
            record,
            output_root=output_root,
            annotation_style=settings.annotation,
            detection_settings=settings.detection,
        )
    except Exception as exc:
        LOGGER.exception("%s: side-view rendering failed", image_path)
        _write_payload(
            match_path,
            _payload(
                image_path,
                identity,
                query,
                market_hint,
                match,
                selected,
                "FAILED",
                error=str(exc),
                warning=warning,
            ),
        )
        return SideViewResult(
            image_path=image_path,
            status="FAILED",
            query=query,
            identity=identity,
            match=match,
            selected=selected,
            match_path=match_path,
            error=str(exc),
            warning=warning,
        )

    svg_path = output_root / selected.vehicle.size / f"{record.id}.svg"
    _write_payload(
        match_path,
        _payload(
            image_path,
            identity,
            query,
            market_hint,
            match,
            selected,
            status,
            warning=warning,
            svg_path=svg_path if status == "EXPORTED" else None,
        ),
    )
    LOGGER.info(
        "%s: %s -> %s (%s)",
        image_path.name,
        query,
        selected.vehicle.dimension_id,
        status,
    )
    return SideViewResult(
        image_path=image_path,
        status=status,
        query=query,
        identity=identity,
        match=match,
        selected=selected,
        svg_path=svg_path if status == "EXPORTED" else None,
        match_path=match_path,
        error="" if status == "EXPORTED" else status,
        warning=warning,
    )
