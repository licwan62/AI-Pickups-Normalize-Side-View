import csv
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "sync_project_csvs.py"


def write_dimensions(path: Path) -> None:
    path.parent.mkdir(parents=True)
    path.write_text(
        "DIMENSION-ID,L-MM,W-MM,H-MM\n"
        "Jeep Wrangler 2dr JL Xtreme SUV 2026 US,4341,1877,1918\n"
        "Tesla Model X SUV 2016-2026 US,5057,2004,1679\n",
        encoding="utf-8",
    )


def run_sync(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args], cwd=root, text=True,
        capture_output=True, check=False,
    )


def test_generates_project_csv_from_exact_dimension_name(tmp_path):
    write_dimensions(tmp_path / "data/dimensions/all.csv")
    image = tmp_path / "img/input/custom/Tesla Model X SUV 2016-2026.jpg"
    image.parent.mkdir(parents=True)
    image.touch()

    result = run_sync(tmp_path, "--dimensions", "data/dimensions/all.csv", "--apply")

    assert result.returncode == 0
    with (tmp_path / "data/projects/vehicles_custom.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row == {
        "name": "Tesla Model X SUV 2016-2026", "Size": "custom",
        "length_mm": "5057", "width_mm": "2004", "height_mm": "1679",
        "image_path": "img/input/custom/Tesla Model X SUV 2016-2026.jpg",
    }


def test_existing_row_is_a_confirmed_alias(tmp_path):
    write_dimensions(tmp_path / "data/dimensions/all.csv")
    image = tmp_path / "img/input/custom/JL WRANGLER 2DR.webp"
    image.parent.mkdir(parents=True)
    image.touch()
    projects = tmp_path / "data/projects"
    projects.mkdir(parents=True)
    (projects / "vehicles_custom.csv").write_text(
        "name,Size,length_mm,width_mm,height_mm,image_path\n"
        "Jeep Wrangler 2dr JL Xtreme SUV 2026,custom,4341,1877,1918,img\\input\\custom\\JL WRANGLER 2DR.webp\n",
        encoding="utf-8",
    )

    result = run_sync(tmp_path, "--dimensions", "data/dimensions/all.csv", "--apply")

    assert result.returncode == 0
    assert "Jeep Wrangler 2dr JL Xtreme SUV 2026" in (projects / "vehicles_custom.csv").read_text()


def test_unmatched_image_writes_incomplete_row(tmp_path):
    write_dimensions(tmp_path / "data/dimensions/all.csv")
    image = tmp_path / "img/input/custom/unknown.png"
    image.parent.mkdir(parents=True)
    image.touch()

    result = run_sync(tmp_path, "--dimensions", "data/dimensions/all.csv", "--apply")

    assert result.returncode == 0
    assert "incomplete" in result.stdout
    with (tmp_path / "data/projects/vehicles_custom.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row == {
        "name": "unknown", "Size": "custom", "length_mm": "",
        "width_mm": "", "height_mm": "",
        "image_path": "img/input/custom/unknown.png",
    }


def test_matches_abbreviated_name_when_year_is_inside_master_range(tmp_path):
    write_dimensions(tmp_path / "data/dimensions/all.csv")
    image = tmp_path / "img/input/custom/2020 Tesla Model X.jpg"
    image.parent.mkdir(parents=True)
    image.touch()

    result = run_sync(tmp_path, "--dimensions", "data/dimensions/all.csv", "--apply")

    assert result.returncode == 0
    output = (tmp_path / "data/projects/vehicles_custom.csv").read_text()
    assert "Tesla Model X SUV 2016-2026" in output


def test_does_not_match_year_outside_master_range(tmp_path):
    write_dimensions(tmp_path / "data/dimensions/all.csv")
    image = tmp_path / "img/input/custom/2015 Tesla Model X.jpg"
    image.parent.mkdir(parents=True)
    image.touch()

    result = run_sync(tmp_path, "--dimensions", "data/dimensions/all.csv", "--apply")

    assert result.returncode == 0
    output = (tmp_path / "data/projects/vehicles_custom.csv").read_text()
    assert '"2015 Tesla Model X","custom","","","",' in output
