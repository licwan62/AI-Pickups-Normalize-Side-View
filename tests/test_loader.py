from PIL import Image

from pickup_measure.src.loader import load_records


def test_image_defaults_to_id_filename(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (10, 10)).save(images / "F150_01.png")
    table = tmp_path / "vehicles.tsv"
    table.write_text(
        "id\tname\tlength_mm\twidth_mm\theight_mm\n"
        "F150_01\tFord F150\t6195\t2029\t1971\n",
        encoding="utf-8",
    )

    records = load_records(table)

    assert records[0].image_path == (images / "F150_01.png").resolve()


def test_missing_explicit_path_falls_back_to_configured_images_dir(tmp_path):
    images = tmp_path / "input" / "images"
    images.mkdir(parents=True)
    Image.new("RGB", (10, 10)).save(images / "truck.png")
    table = tmp_path / "input" / "vehicles.tsv"
    table.write_text(
        "id\tname\timage_path\tlength_mm\twidth_mm\theight_mm\n"
        "F150_01\tFord F150\t..\\images\\truck.png\t6195\t2029\t1971\n",
        encoding="utf-8",
    )

    records = load_records(table, images)

    assert records[0].image_path == (images / "truck.png").resolve()
