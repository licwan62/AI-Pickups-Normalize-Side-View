"""Build project CSV files from images and the master vehicle dimension table.

Existing project rows are used as confirmed aliases, so an abbreviated image
filename can keep pointing at its previously selected dimension record. New
images are matched against DIMENSION-ID. No files are written unless --apply
is supplied.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_EXTENSIONS = {".avif", ".jpg", ".jpeg", ".png", ".webp"}
OUTPUT_COLUMNS = ("name", "Size", "length_mm", "width_mm", "height_mm", "image_path")
PROJECT_SIZE_GROUPS = {
    "2M&2XXL-530": {"2M", "2XXL-530"},
    "3L&3XXL-550": {"3L", "3XL", "3XXL-520", "3XXL-550"},
    "4S&4XXL": {"4S", "4M", "4L", "4XL", "4XXL"},
    "YM&YXXL-585": {"YM", "YL", "YXL", "YXXL-525", "YXXL-545", "YXXL-585"},
    "PK-S&PK-XXL-680": {"PK-S", "PK-M", "PK-L", "PK-XL", "PK-XXL-645", "PK-XXL-680"},
}
MARKET_SUFFIX = re.compile(r"\s+(?:US|EU|CN|JP|KR|RU)$", re.IGNORECASE)
NON_ALNUM = re.compile(r"[^a-z0-9]+")
YEAR = re.compile(r"(?<!\d)(19\d{2}|20\d{2})(?:\s*-\s*(19\d{2}|20\d{2}))?(?!\d)")
IGNORED_TOKENS = {
    "4", "4d", "class", "convertible", "coupe", "crossover", "d",
    "hatchback", "mpv", "sedan", "suv", "us", "eu", "cn", "jp", "kr",
    "ru", "wagon",
}


@dataclass(frozen=True)
class Dimension:
    name: str
    length_mm: str
    width_mm: str
    height_mm: str


def normalize(value: str) -> str:
    """Normalize names while making year placement and punctuation irrelevant."""
    return " ".join(sorted(token for token in NON_ALNUM.split(value.lower()) if token))


def year_range(value: str) -> tuple[int, int] | None:
    match = YEAR.search(value)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2) or match.group(1))


def model_tokens(value: str) -> set[str]:
    without_year = YEAR.sub(" ", value.lower())
    return {
        token for token in NON_ALNUM.split(without_year)
        if token and token not in IGNORED_TOKENS
    }


def compatible_name(image_name: str, dimension_name: str) -> bool:
    image_years = year_range(image_name)
    dimension_years = year_range(dimension_name)
    if image_years and dimension_years:
        if image_years[0] < dimension_years[0] or image_years[1] > dimension_years[1]:
            return False
    image_tokens = model_tokens(image_name)
    dimension_tokens = model_tokens(dimension_name)
    return image_tokens == dimension_tokens or image_tokens < dimension_tokens


def same_model_tokens(image_name: str, dimension_name: str) -> bool:
    return model_tokens(image_name) == model_tokens(dimension_name)


def display_name(dimension_id: str) -> str:
    return MARKET_SUFFIX.sub("", dimension_id.strip())


def read_dimensions(path: Path) -> tuple[list[Dimension], dict[str, list[Dimension]]]:
    dimensions: list[Dimension] = []
    by_name: dict[str, list[Dimension]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"DIMENSION-ID", "L-MM", "W-MM", "H-MM"}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"dimension CSV is missing columns: {', '.join(sorted(missing))}")
        for line_number, row in enumerate(reader, start=2):
            name = display_name(row["DIMENSION-ID"])
            values = (row["L-MM"].strip(), row["W-MM"].strip(), row["H-MM"].strip())
            if not name or not all(values):
                continue
            if not all(value.isdigit() and int(value) > 0 for value in values):
                raise ValueError(f"invalid dimensions at {path}:{line_number}")
            item = Dimension(name, *values)
            dimensions.append(item)
            by_name.setdefault(normalize(name), []).append(item)
    return dimensions, by_name


def existing_aliases(project_dir: Path) -> dict[str, Dimension]:
    aliases: dict[str, Dimension] = {}
    for path in sorted(project_dir.glob("*.csv")):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                image_path = (row.get("image_path") or "").replace("\\", "/").lower()
                if not image_path:
                    continue
                aliases[image_path] = Dimension(
                    (row.get("name") or "").strip(),
                    (row.get("length_mm") or "").strip(),
                    (row.get("width_mm") or "").strip(),
                    (row.get("height_mm") or "").strip(),
                )
    return aliases


def project_filename(size: str) -> str:
    """Return the aggregate project CSV filename for an input image size."""
    for filename, sizes in PROJECT_SIZE_GROUPS.items():
        if size in sizes:
            return f"{filename}.csv"
    return f"{size}.csv"


def image_key(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix().lower()


def match_image(
    image: Path,
    root: Path,
    aliases: dict[str, Dimension],
    dimensions: list[Dimension],
    by_name: dict[str, list[Dimension]],
) -> Dimension:
    alias = aliases.get(image_key(image, root))
    if alias and alias.name and all((alias.length_mm, alias.width_mm, alias.height_mm)):
        return alias
    matches = by_name.get(normalize(image.stem), [])
    if not matches:
        compatible = [item for item in dimensions if compatible_name(image.stem, item.name)]
        exact_model = [item for item in compatible if same_model_tokens(image.stem, item.name)]
        matches = exact_model or compatible
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError("no DIMENSION-ID match")
    raise ValueError(f"ambiguous DIMENSION-ID match ({len(matches)} rows)")


def render_csv(rows: list[dict[str, str]]) -> str:
    import io

    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=OUTPUT_COLUMNS,
        lineterminator="\n",
        quoting=csv.QUOTE_ALL,
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("img/input"))
    parser.add_argument("--dimensions", type=Path, default=Path("data/dimensions/全尺码全量.csv"))
    parser.add_argument("--projects", type=Path, default=Path("data/projects"))
    parser.add_argument("--apply", action="store_true", help="write generated project CSV files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path.cwd().resolve()
    images_dir = (root / args.images).resolve()
    dimensions_path = (root / args.dimensions).resolve()
    projects_dir = (root / args.projects).resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(images_dir)
    if not dimensions_path.is_file():
        raise FileNotFoundError(dimensions_path)

    dimensions, by_name = read_dimensions(dimensions_path)
    aliases = existing_aliases(projects_dir) if projects_dir.is_dir() else {}
    generated: dict[str, list[dict[str, str]]] = {}
    warnings: list[str] = []
    images = sorted(
        (path for path in images_dir.glob("*/*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda path: path.as_posix().lower(),
    )
    for image in images:
        size = image.parent.name
        project_file = project_filename(size)
        try:
            dimension = match_image(image, root, aliases, dimensions, by_name)
        except ValueError as exc:
            warnings.append(f"{image.relative_to(root)}: {exc}")
            generated.setdefault(project_file, []).append({
                "name": image.stem,
                "Size": size,
                "length_mm": "",
                "width_mm": "",
                "height_mm": "",
                "image_path": image.relative_to(root).as_posix(),
            })
            continue
        generated.setdefault(project_file, []).append({
            "name": dimension.name,
            "Size": size,
            "length_mm": dimension.length_mm,
            "width_mm": dimension.width_mm,
            "height_mm": dimension.height_mm,
            "image_path": image.relative_to(root).as_posix(),
        })

    print(f"images: {len(images)}; matched: {len(images) - len(warnings)}; incomplete: {len(warnings)}")
    for warning in warnings:
        print(f"incomplete: {warning}")

    changes: list[tuple[Path, str]] = []
    for filename, rows in generated.items():
        target = projects_dir / filename
        content = render_csv(rows)
        old = target.read_text(encoding="utf-8-sig") if target.is_file() else None
        if old != content:
            changes.append((target, content))
    stale = sorted(set(projects_dir.glob("*.csv")) - {path for path, _ in changes} - {
        projects_dir / filename for filename in generated
    }) if projects_dir.is_dir() else []
    print(f"project CSV changes: {len(changes)}; stale project CSV files: {len(stale)}")
    if not args.apply:
        print("Dry run only. Re-run with --apply to write changes.")
        return 0
    projects_dir.mkdir(parents=True, exist_ok=True)
    for target, content in changes:
        target.write_text(content, encoding="utf-8", newline="")
        print(f"wrote: {target.relative_to(root)}")
    for target in stale:
        target.unlink()
        print(f"removed stale: {target.relative_to(root)}")
    if warnings:
        print("Incomplete rows were written with blank dimension fields.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
