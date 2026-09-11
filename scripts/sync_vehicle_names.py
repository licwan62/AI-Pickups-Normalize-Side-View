"""Match vehicle images, then synchronize CSV image_path and name fields.

Matching is limited to the image_path's existing Size directory. A candidate
must have the same non-year model tokens and, when both names contain years,
its year must be inside the record's year range. The candidate filename is then
authoritative for both name and image_path. Images are never moved or renamed.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


SUPPORTED_EXTENSIONS = {".avif", ".jpg", ".jpeg", ".png", ".webp"}
YEAR_PATTERN = re.compile(r"(?<!\d)(\d{4})(?:\s*-\s*(\d{4}))?(?!\d)")
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
OPTIONAL_VARIANT_TOKENS = {"maybach", "sedan"}


def model_tokens(name: str) -> tuple[str, ...]:
    without_years = YEAR_PATTERN.sub(" ", name.lower())
    return tuple(
        token for token in TOKEN_PATTERN.findall(without_years)
        if token not in OPTIONAL_VARIANT_TOKENS
    )


def year_range(name: str) -> tuple[int, int] | None:
    match = YEAR_PATTERN.search(name)
    if not match:
        return None
    start = int(match.group(1))
    return start, int(match.group(2) or start)


def matches_record(record_name: str, image_name: str) -> bool:
    if model_tokens(record_name) != model_tokens(image_name):
        return False
    record_years = year_range(record_name)
    image_years = year_range(image_name)
    if record_years is None or image_years is None:
        return True
    return record_years[0] <= image_years[0] <= record_years[1]


def find_unique_candidate(record_name: str, directory: Path) -> Path | None:
    if not directory.is_dir():
        return None
    candidates = [
        path for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and matches_record(record_name, path.stem)
    ]
    return candidates[0] if len(candidates) == 1 else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/vehicles.csv"))
    parser.add_argument("--apply", action="store_true", help="Write matched path and name changes")
    args = parser.parse_args()

    root = Path.cwd().resolve()
    csv_path = (root / args.csv).resolve()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or not {"name", "Size", "image_path"}.issubset(fieldnames):
        raise ValueError("CSV must contain name, Size, and image_path columns")

    changes = 0
    missing: list[str] = []
    for row in rows:
        if not row.get("name"):
            continue
        raw_image_path = (row.get("image_path") or "").strip()
        image_path = root / raw_image_path if raw_image_path else root / "img" / "input" / row["Size"]
        if not image_path.is_file():
            search_directory = image_path.parent if raw_image_path else image_path
            candidate = find_unique_candidate(row["name"], search_directory)
            if candidate is None:
                missing.append(raw_image_path or f"img/input/{row['Size']}/ (no image_path)")
                continue
            new_image_path = candidate.relative_to(root).as_posix()
            print(f"{row['image_path']} -> {new_image_path}")
            row["image_path"] = new_image_path
            image_path = candidate
            changes += 1
        expected_name = image_path.stem
        if row["name"] != expected_name:
            print(f"{row['name']} -> {expected_name}")
            row["name"] = expected_name
            changes += 1

    print(f"proposed CSV changes: {changes}; unresolved image paths: {len(missing)}")
    for path in missing:
        print(f"missing: {path}")
    if not args.apply:
        print("Dry run only. Re-run with --apply to write changes.")
        return 0
    if changes == 0:
        print("No CSV changes to write.")
        return 0

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
