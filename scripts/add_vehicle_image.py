"""Register an existing vehicle image, using its filename as the vehicle name.

The command performs no writes unless ``--apply`` is supplied. It validates the
CSV row and appends it with an ``image_path`` relative to the repository root.
Image files are never renamed or moved.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


SUPPORTED_EXTENSIONS = {".avif", ".jpg", ".jpeg", ".png", ".webp"}
UNSAFE_FILENAME = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
REQUIRED_COLUMNS = ("name", "Size", "length_mm", "width_mm", "height_mm", "image_path")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", required=True, help="Target Size group, for example 3XL+")
    parser.add_argument("--length-mm", required=True, type=int)
    parser.add_argument("--width-mm", required=True, type=int)
    parser.add_argument("--height-mm", required=True, type=int)
    parser.add_argument("--source", required=True, type=Path, help="Existing image to register")
    parser.add_argument("--csv", type=Path, default=Path("data/vehicles.csv"))
    parser.add_argument("--apply", action="store_true", help="Move the image and append the CSV row")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path.cwd().resolve()
    csv_path = (root / args.csv).resolve()
    source = (root / args.source).resolve()
    if not re.fullmatch(r"[A-Za-z0-9+_.-]+", args.size):
        raise ValueError("--size may contain only letters, numbers, +, _, . and -")
    if min(args.length_mm, args.width_mm, args.height_mm) <= 0:
        raise ValueError("all dimensions must be positive")
    if not source.is_file() or source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"--source must be an existing supported image: {source}")
    name = source.stem
    if not name.strip() or UNSAFE_FILENAME.search(name) or name.endswith((".", " ")):
        raise ValueError("image filename must have a non-empty Windows-safe stem")
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    image_path = source.relative_to(root).as_posix()
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
        if tuple(handle.seek(0) or csv.reader(handle).__next__()) != REQUIRED_COLUMNS:
            raise ValueError(f"{csv_path} must use columns: {', '.join(REQUIRED_COLUMNS)}")
    matching_rows = [row for row in rows if row["name"] == name and row["Size"] == args.size]
    if len(matching_rows) > 1:
        raise ValueError(f"duplicate vehicle records already exist: {name} ({args.size})")

    row = {
        "name": name,
        "Size": args.size,
        "length_mm": str(args.length_mm),
        "width_mm": str(args.width_mm),
        "height_mm": str(args.height_mm),
        "image_path": image_path,
    }
    if matching_rows and matching_rows[0] != row:
        raise ValueError(
            "vehicle already exists with different dimensions or image_path: "
            f"{matching_rows[0]}"
        )
    print(f"image: {image_path}")
    print(f"row: {row}")
    if not args.apply:
        print("Dry run only. Re-run with --apply to make changes.")
        return 0

    if not matching_rows:
        with csv_path.open("a", encoding="utf-8", newline="") as handle:
            csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS).writerow(row)
    print("Applied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
