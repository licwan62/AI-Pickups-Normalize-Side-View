"""Load, cache, and search the authoritative vehicle dimension tables.

The dimension master table lives on the NAS and is split by market.  Its
``DIMENSION-ID`` is already the canonical vehicle name (make, model, body,
year range, cab/bed, market), so matching is performed against that value and
the result is converted back into the pipeline's existing ``VehicleRecord``.

Only standard-library CSV helpers plus pandas are used; the module has no
network dependency, which keeps offline matching possible.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

from .settings import MasterTableSettings


LOGGER = logging.getLogger("pickup_measure.master_table")

CACHE_VERSION = 1
CACHE_COLUMNS = (
    "dimension_id",
    "name",
    "make",
    "model",
    "trim",
    "year",
    "size",
    "length_mm",
    "width_mm",
    "height_mm",
    "structure",
    "market",
    "sales",
    "source",
)

DEFAULT_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "dimension_id": ("DIMENSION-ID", "DIMENSION_ID", "DIMENSIONID", "ID"),
    "make": ("MAKE", "BRAND"),
    "model": ("MODEL",),
    "trim": ("TRIM", "版本"),
    "year": ("YEAR", "年份"),
    "size": ("自动尺码", "AUTO-SIZE", "AUTO_SIZE", "SIZE"),
    "structure": ("结构", "STRUCTURE", "BODY"),
    "length_mm": ("L-MM", "L_MM", "LENGTH-MM", "LENGTH_MM", "LENGTHMM"),
    "width_mm": ("W-MM", "W_MM", "WIDTH-MM", "WIDTH_MM", "WIDTHMM"),
    "height_mm": ("H-MM", "H_MM", "HEIGHT-MM", "HEIGHT_MM", "HEIGHTMM"),
    "length_in": ("L-IN", "L_IN", "LENGTH-IN", "LENGTH_IN"),
    "width_in": ("W-IN", "W_IN", "WIDTH-IN", "WIDTH_IN"),
    "height_in": ("H-IN", "H_IN", "HEIGHT-IN", "HEIGHT_IN"),
    "sales": ("车型销量", "SALES"),
}

MARKET_SUFFIX = re.compile(r"\s+(?:US|EU|CN|JP|KR|RU)$", re.IGNORECASE)
MARKET_IN_ID = re.compile(r"\s+(US|EU|CN|JP|KR|RU)$", re.IGNORECASE)
NON_ALNUM = re.compile(r"[^a-z0-9\u4e00-\u9fff]+")
YEAR_PATTERN = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
UNSAFE_SIZE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

GENERIC_TOKENS = {
    "2d",
    "4",
    "4d",
    "class",
    "cn",
    "convertible",
    "coupe",
    "crossover",
    "d",
    "eu",
    "hatchback",
    "jp",
    "kr",
    "minivan",
    "mpv",
    "pickup",
    "ru",
    "sedan",
    "suv",
    "truck",
    "us",
    "van",
    "wagon",
}


def normalize_text(value: object) -> str:
    """Normalize a name so punctuation, case, and token order do not matter."""

    folded = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(sorted(token for token in NON_ALNUM.split(folded) if token))


def compact_text(value: object) -> str:
    folded = unicodedata.normalize("NFKC", str(value)).casefold()
    return NON_ALNUM.sub("", folded)


def query_tokens(value: object) -> set[str]:
    """Tokens used for scoring; years and generic body words are left out."""

    text = YEAR_PATTERN.sub(" ", str(value).casefold())
    return {
        token
        for token in NON_ALNUM.split(text)
        if token
        and token not in GENERIC_TOKENS
        and not (len(token) == 4 and token.isdigit())
    }


def parse_years(value: object) -> tuple[int, int] | None:
    years = [int(match) for match in YEAR_PATTERN.findall(str(value))]
    if not years:
        return None
    return min(years), max(years)


def query_year(value: object) -> int | None:
    years = parse_years(value)
    return years[0] if years else None


def display_name(dimension_id: str) -> str:
    """Strip the market suffix, exactly like the existing project CSVs."""

    return MARKET_SUFFIX.sub("", str(dimension_id).strip())


def market_from_id(dimension_id: str) -> str:
    match = MARKET_IN_ID.search(str(dimension_id).strip())
    return match.group(1).upper() if match else ""


MARKET_IN_TEXT = re.compile(
    r"(?<![A-Za-z])(US|EU|CN|JP|KR|RU)(?![A-Za-z])",
    re.IGNORECASE,
)


def query_market(value: object) -> str:
    """Return a market code explicitly written in a query or filename."""

    match = MARKET_IN_TEXT.search(str(value))
    return match.group(1).upper() if match else ""


def sanitize_size(value: object, fallback: str) -> str:
    cleaned = re.sub(r"\s+", "_", str(value).strip())
    cleaned = UNSAFE_SIZE.sub("_", cleaned).strip("._ ")
    if not cleaned or "无" in cleaned:
        return fallback
    return cleaned


@dataclass(frozen=True)
class MasterVehicle:
    dimension_id: str
    name: str
    make: str
    model: str
    trim: str
    year: str
    size: str
    length_mm: int
    width_mm: int
    height_mm: int
    structure: str = ""
    market: str = ""
    sales: int = 0
    source: str = ""

    @property
    def year_range(self) -> tuple[int, int] | None:
        return parse_years(self.year or self.name)

    @property
    def signature(self) -> tuple[int, int, int]:
        return (self.length_mm, self.width_mm, self.height_mm)

    @property
    def tokens(self) -> set[str]:
        tokens = query_tokens(" ".join((self.name, self.structure)))
        compact_model = compact_text(self.model)
        if compact_model:
            tokens.add(compact_model)
        return tokens

    @property
    def compact_name(self) -> str:
        return compact_text(self.name)

    def as_dict(self) -> dict[str, object]:
        return {
            "dimension_id": self.dimension_id,
            "name": self.name,
            "make": self.make,
            "model": self.model,
            "trim": self.trim,
            "year": self.year,
            "size": self.size,
            "length_mm": self.length_mm,
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "structure": self.structure,
            "market": self.market,
            "sales": self.sales,
            "source": self.source,
        }


@dataclass(frozen=True)
class MatchCandidate:
    vehicle: MasterVehicle
    score: float
    exact: bool = False

    def as_dict(self) -> dict[str, object]:
        payload = self.vehicle.as_dict()
        payload["score"] = round(self.score, 4)
        payload["exact"] = self.exact
        return payload


@dataclass
class MatchResult:
    query: str
    candidates: list[MatchCandidate] = field(default_factory=list)
    ambiguous: bool = False
    reason: str = ""
    threshold: float = 0.0

    @property
    def best(self) -> MatchCandidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def ok(self) -> bool:
        return (
            bool(self.candidates)
            and not self.ambiguous
            and self.candidates[0].score >= self.threshold
        )

    @property
    def alternatives(self) -> list[MatchCandidate]:
        return self.candidates[1:]

    def selected(self, index: int | None = None) -> MatchCandidate | None:
        if index is None:
            return self.best if self.ok else None
        if 1 <= index <= len(self.candidates):
            return self.candidates[index - 1]
        return None


def _int_value(value: object) -> int | None:
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if number <= 0:
        return None
    return int(round(number))


def _mm_value(
    row: Mapping[str, str],
    mm_column: str | None,
    inch_column: str | None,
) -> int | None:
    if mm_column:
        value = _int_value(row.get(mm_column, ""))
        if value:
            return value
    if inch_column:
        inches = _int_value(row.get(inch_column, ""))
        if inches:
            return int(round(inches * 25.4))
    return None


def _pick_column(
    frame: pd.DataFrame,
    key: str,
    overrides: Mapping[str, str],
) -> str | None:
    aliases = list(DEFAULT_COLUMN_ALIASES.get(key, ()))
    override = str(overrides.get(key, "")).strip()
    if override:
        aliases.insert(0, override)
    lookup = {str(column).strip().casefold(): str(column) for column in frame.columns}
    for alias in aliases:
        match = lookup.get(alias.strip().casefold())
        if match is not None:
            return match
    return None


def _decode_table(raw: bytes) -> str:
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("gb18030")


def _read_frame(path: Path, overrides: Mapping[str, str]) -> pd.DataFrame:
    text = _decode_table(path.read_bytes())
    delimiter = ","
    try:
        delimiter = csv.Sniffer().sniff(text[:65536], delimiters=",\t;").delimiter
    except csv.Error:
        pass
    frame = pd.read_csv(
        io.StringIO(text),
        sep=delimiter,
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _frame_to_vehicles(
    frame: pd.DataFrame,
    overrides: Mapping[str, str],
    fallback_size: str,
    source_name: str,
) -> tuple[list[MasterVehicle], int]:
    columns = {
        key: _pick_column(frame, key, overrides)
        for key in (
            "dimension_id",
            "make",
            "model",
            "trim",
            "year",
            "size",
            "structure",
            "length_mm",
            "width_mm",
            "height_mm",
            "length_in",
            "width_in",
            "height_in",
            "sales",
        )
    }
    if columns["dimension_id"] is None:
        raise ValueError(f"{source_name}: missing DIMENSION-ID column")
    if columns["length_mm"] is None and columns["length_in"] is None:
        raise ValueError(f"{source_name}: missing L-MM (or L-IN) column")

    def value(row: Mapping[str, str], key: str) -> str:
        column = columns[key]
        return str(row.get(column, "")).strip() if column else ""

    vehicles: list[MasterVehicle] = []
    skipped = 0
    for row in frame.to_dict("records"):
        dimension_id = value(row, "dimension_id")
        if not dimension_id:
            skipped += 1
            continue
        length = _mm_value(row, columns["length_mm"], columns["length_in"])
        width = _mm_value(row, columns["width_mm"], columns["width_in"])
        height = _mm_value(row, columns["height_mm"], columns["height_in"])
        if length is None or width is None or height is None:
            skipped += 1
            continue
        vehicles.append(
            MasterVehicle(
                dimension_id=dimension_id,
                name=display_name(dimension_id),
                make=value(row, "make"),
                model=value(row, "model"),
                trim=value(row, "trim"),
                year=value(row, "year"),
                size=sanitize_size(value(row, "size"), fallback_size),
                length_mm=length,
                width_mm=width,
                height_mm=height,
                structure=value(row, "structure"),
                market=market_from_id(dimension_id),
                sales=_int_value(value(row, "sales")) or 0,
                source=source_name,
            )
        )
    return vehicles, skipped


def resolve_source_files(settings: MasterTableSettings) -> list[Path]:
    source = settings.source
    if source.is_file():
        return [source]
    if not source.is_dir():
        return []
    found: dict[str, Path] = {}
    for pattern in settings.patterns:
        for path in source.glob(pattern):
            if path.is_file():
                found[path.as_posix().casefold()] = path
    return sorted(found.values(), key=lambda item: item.as_posix().casefold())


def _source_signature(paths: Sequence[Path]) -> list[dict[str, object]]:
    signature: list[dict[str, object]] = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        signature.append(
            {
                "path": path.as_posix(),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return signature


def _cache_meta_path(cache_path: Path) -> Path:
    return cache_path.with_name(cache_path.name + ".meta.json")


def _cache_is_current(meta_path: Path, signature: Sequence[Mapping[str, object]]) -> bool:
    if not meta_path.is_file():
        return False
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        payload.get("version") == CACHE_VERSION
        and payload.get("sources") == list(signature)
    )


def _write_cache(
    cache_path: Path,
    vehicles: Sequence[MasterVehicle],
    signature: Sequence[Mapping[str, object]],
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [vehicle.as_dict() for vehicle in vehicles],
        columns=list(CACHE_COLUMNS),
    )
    temporary = cache_path.with_name(cache_path.name + ".tmp")
    frame.to_csv(
        temporary,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n",
    )
    temporary.replace(cache_path)
    meta_path = _cache_meta_path(cache_path)
    meta_path.write_text(
        json.dumps(
            {
                "version": CACHE_VERSION,
                "built_at": datetime.now(timezone.utc).isoformat(),
                "count": len(vehicles),
                "sources": list(signature),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _read_cache(cache_path: Path) -> list[MasterVehicle]:
    frame = pd.read_csv(
        cache_path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    missing = set(CACHE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(
            f"Master cache {cache_path} is missing columns: "
            f"{', '.join(sorted(missing))}"
        )
    vehicles: list[MasterVehicle] = []
    for row in frame.to_dict("records"):
        length = _int_value(row.get("length_mm"))
        width = _int_value(row.get("width_mm"))
        height = _int_value(row.get("height_mm"))
        dimension_id = str(row.get("dimension_id", "")).strip()
        if not dimension_id or length is None or width is None or height is None:
            continue
        vehicles.append(
            MasterVehicle(
                dimension_id=dimension_id,
                name=str(row.get("name", "")).strip() or display_name(dimension_id),
                make=str(row.get("make", "")).strip(),
                model=str(row.get("model", "")).strip(),
                trim=str(row.get("trim", "")).strip(),
                year=str(row.get("year", "")).strip(),
                size=str(row.get("size", "")).strip() or "UNCLASSIFIED",
                length_mm=length,
                width_mm=width,
                height_mm=height,
                structure=str(row.get("structure", "")).strip(),
                market=str(row.get("market", "")).strip(),
                sales=_int_value(row.get("sales")) or 0,
                source=str(row.get("source", "")).strip(),
            )
        )
    return vehicles


def load_master_table(
    settings: MasterTableSettings,
    *,
    source_override: Path | None = None,
    force_refresh: bool = False,
) -> "MasterTable":
    """Return the cached or freshly loaded master table.

    ``auto`` rebuilds the cache whenever a source file's size or modification
    time changes.  If the NAS is unreachable, an existing cache is used so a
    single-image run keeps working offline.
    """

    effective = (
        replace(settings, source=source_override)
        if source_override is not None
        else settings
    )
    paths = resolve_source_files(effective)
    signature = _source_signature(paths)
    cache_path = effective.cache_path
    meta_path = _cache_meta_path(cache_path)
    refresh = "always" if force_refresh else effective.refresh

    def from_cache() -> "MasterTable":
        return MasterTable(
            _read_cache(cache_path),
            cache_path=cache_path,
            preferred_markets=effective.preferred_markets,
        )

    if refresh != "always" and cache_path.is_file():
        if refresh == "never":
            return from_cache()
        if not paths:
            LOGGER.warning(
                "Master table source is unavailable; using cache %s", cache_path
            )
            return from_cache()
        if _cache_is_current(meta_path, signature):
            return from_cache()

    if not paths:
        raise FileNotFoundError(
            "No master table found. Checked source "
            f"{effective.source} with patterns {list(effective.patterns)}."
        )

    merged: dict[str, MasterVehicle] = {}
    for path in paths:
        frame = _read_frame(path, effective.columns)
        vehicles, skipped = _frame_to_vehicles(
            frame,
            effective.columns,
            effective.fallback_size,
            path.parent.parent.name or path.name,
        )
        if skipped:
            LOGGER.warning(
                "%s: skipped %d rows without usable dimensions", path, skipped
            )
        for vehicle in vehicles:
            key = vehicle.dimension_id.casefold()
            existing = merged.get(key)
            if existing is None:
                merged[key] = vehicle
            elif existing.signature != vehicle.signature:
                LOGGER.warning(
                    "Conflicting dimensions for %s: %s vs %s",
                    vehicle.dimension_id,
                    existing.signature,
                    vehicle.signature,
                )
        LOGGER.info("%s: loaded %d vehicles", path, len(vehicles))

    loaded = list(merged.values())
    if not loaded:
        raise ValueError(
            "Master table sources contained no rows with DIMENSION-ID and dimensions"
        )
    try:
        _write_cache(cache_path, loaded, signature)
    except OSError as exc:
        LOGGER.warning("Could not write master table cache %s: %s", cache_path, exc)
    return MasterTable(
        loaded,
        cache_path=cache_path if cache_path.is_file() else None,
        preferred_markets=effective.preferred_markets,
    )


class MasterTable:
    """In-memory index over one or more master tables."""

    def __init__(
        self,
        vehicles: Iterable[MasterVehicle],
        *,
        cache_path: Path | None = None,
        preferred_markets: Sequence[str] = ("US", "EU", "RU"),
    ) -> None:
        self.vehicles = list(vehicles)
        self.cache_path = cache_path
        self.preferred_markets = tuple(market.upper() for market in preferred_markets)
        self._market_rank = {
            market: index for index, market in enumerate(self.preferred_markets)
        }
        self._by_id: dict[str, list[int]] = defaultdict(list)
        self._by_name: dict[str, list[int]] = defaultdict(list)
        self._token_index: dict[str, set[int]] = defaultdict(set)
        for index, vehicle in enumerate(self.vehicles):
            self._by_id[vehicle.dimension_id.casefold()].append(index)
            self._by_name[normalize_text(vehicle.name)].append(index)
            for token in vehicle.tokens:
                self._token_index[token].add(index)

    def __len__(self) -> int:
        return len(self.vehicles)

    def stats(self) -> dict[str, object]:
        return {
            "vehicles": len(self.vehicles),
            "tokens": len(self._token_index),
            "cache_path": str(self.cache_path) if self.cache_path else "",
        }

    def find(
        self,
        query: str,
        *,
        dimension_id: str | None = None,
        market: str | None = None,
        limit: int = 12,
        min_score: float = 0.62,
    ) -> MatchResult:
        if dimension_id:
            return self._find_by_dimension_id(
                dimension_id, limit=limit, min_score=min_score
            )
        text = str(query).strip()
        if not text:
            return MatchResult(text, reason="empty query", threshold=min_score)
        market_key = str(market or "").strip().upper()

        def in_market(index: int) -> bool:
            if not market_key:
                return True
            return self.vehicles[index].market.upper() == market_key

        id_indices = self._by_id.get(text.casefold(), [])
        if id_indices:
            exact = [
                MatchCandidate(self.vehicles[index], 1.0, True)
                for index in id_indices
            ]
            return self._finalize(text, exact, min_score, limit)

        exact_indices = [
            index
            for index in self._by_name.get(normalize_text(text), [])
            if in_market(index)
        ]
        if exact_indices:
            exact = [
                MatchCandidate(self.vehicles[index], 1.0, True)
                for index in exact_indices
            ]
            return self._finalize(text, exact, min_score, limit)

        tokens = query_tokens(text)
        indices: set[int] = set()
        for token in tokens:
            indices.update(self._token_index.get(token, ()))
        compact = compact_text(text)
        if not indices and compact:
            indices = {
                index
                for index, vehicle in enumerate(self.vehicles)
                if compact == vehicle.compact_name
                or (len(compact) >= 4 and compact in vehicle.compact_name)
            }
        candidates: list[MatchCandidate] = []
        for index in indices:
            if not in_market(index):
                continue
            score = self._score(text, tokens, compact, self.vehicles[index])
            if score is not None:
                candidates.append(MatchCandidate(self.vehicles[index], score))
        result = self._finalize(text, candidates, min_score, limit)
        if not result.candidates:
            result.reason = (
                f"no {market_key} candidate shared a model token"
                if market_key
                else "no candidate shared a model token"
            )
        return result

    def _find_by_dimension_id(
        self,
        dimension_id: str,
        *,
        limit: int,
        min_score: float,
    ) -> MatchResult:
        key = str(dimension_id).strip()
        indices = self._by_id.get(key.casefold(), [])
        exact = True
        if not indices:
            indices = [
                index
                for stored_id, stored_indices in self._by_id.items()
                if key.casefold() in stored_id
                for index in stored_indices
            ]
            exact = False
        candidates = [
            MatchCandidate(
                self.vehicles[index],
                1.0 if exact else 0.9,
                exact,
            )
            for index in indices
        ]
        result = self._finalize(key, candidates, min_score, limit)
        if not result.candidates:
            result.reason = "DIMENSION-ID not found"
        return result

    def _score(
        self,
        query: str,
        tokens: set[str],
        compact: str,
        vehicle: MasterVehicle,
    ) -> float | None:
        if not tokens:
            return None
        vehicle_tokens = vehicle.tokens
        overlap = tokens & vehicle_tokens
        if not overlap:
            return None
        coverage = len(overlap) / len(tokens)
        precision = len(overlap) / max(1, len(vehicle_tokens))
        score = 0.72 * coverage + 0.28 * precision
        if compact and (
            compact == vehicle.compact_name or compact in vehicle.compact_name
        ):
            score += 0.06
        make_tokens = query_tokens(vehicle.make)
        if make_tokens and make_tokens <= tokens:
            score += 0.04
        year = query_year(query)
        vehicle_years = vehicle.year_range
        if year and vehicle_years:
            if vehicle_years[0] <= year <= vehicle_years[1]:
                score += 0.10
            else:
                score -= 0.35
        # Bonuses may push a perfect match slightly above 1.0.  Keep the raw
        # value so a trim-specific row outranks a generic row of the same model.
        return max(0.0, score)

    def _rank_key(self, candidate: MatchCandidate) -> tuple[object, ...]:
        vehicle = candidate.vehicle
        years = vehicle.year_range
        year_end = years[1] if years else 0
        return (
            -candidate.score,
            -int(candidate.exact),
            self._market_rank.get(vehicle.market, len(self._market_rank)),
            -year_end,
            -vehicle.sales,
            len(vehicle.name),
            vehicle.dimension_id,
        )

    def _finalize(
        self,
        query: str,
        candidates: Sequence[MatchCandidate],
        min_score: float,
        limit: int,
    ) -> MatchResult:
        ranked = sorted(candidates, key=self._rank_key)
        if not ranked:
            return MatchResult(query, reason="no match", threshold=min_score)
        top = ranked[0]
        if top.score >= min_score:
            group = [
                candidate
                for candidate in ranked
                if top.score - candidate.score <= 0.03 + 1e-9
            ]
            # A width-only difference does not change the rendered side view,
            # so it is not a blocking ambiguity.
            rendered = {
                (item.vehicle.length_mm, item.vehicle.height_mm)
                for item in group
            }
            ambiguous = len(rendered) > 1
            reason = "multiple length/height combinations" if ambiguous else ""
        else:
            ambiguous = False
            reason = f"best score {top.score:.3f} is below {min_score:.2f}"
        return MatchResult(
            query,
            list(ranked[: max(1, limit)]),
            ambiguous=ambiguous,
            reason=reason,
            threshold=min_score,
        )

