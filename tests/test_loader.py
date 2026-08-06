from PIL import Image

from pickup_measure.src.loader import load_image, load_records


def test_csv_input_is_parsed_with_comma_separator(tmp_path):
    image_path = tmp_path / "truck.png"
    Image.new("RGB", (10, 10)).save(image_path)
    table = tmp_path / "vehicles.csv"
    table.write_text(
        "name,Size,image_path,length_mm,width_mm,height_mm\n"
        "Ford F150,PK-XL,truck.png,6195,2029,1971\n",
        encoding="utf-8",
    )

    records = load_records(table)

    assert len(records) == 1
    assert records[0].name == "Ford F150"
    assert records[0].image_path == image_path.resolve()


def test_id_is_generated_from_all_fields_except_image_path(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    generated_id = "Ford_F150_PK-XL_6195_2029_1971"
    Image.new("RGB", (10, 10)).save(images / f"{generated_id}.png")
    table = tmp_path / "vehicles.tsv"
    table.write_text(
        "name\tSize\tlength_mm\twidth_mm\theight_mm\n"
        "Ford F150\tPK-XL\t6195\t2029\t1971\n",
        encoding="utf-8",
    )

    records = load_records(table)

    assert records[0].id == generated_id
    assert records[0].image_path == (images / f"{generated_id}.png").resolve()


def test_missing_explicit_path_falls_back_to_configured_images_dir(tmp_path):
    images = tmp_path / "input" / "images"
    images.mkdir(parents=True)
    Image.new("RGB", (10, 10)).save(images / "truck.png")
    table = tmp_path / "input" / "vehicles.tsv"
    table.write_text(
        "name\tSize\timage_path\tlength_mm\twidth_mm\theight_mm\n"
        "Ford F150\tPK-XL\t..\\images\\truck.png\t6195\t2029\t1971\n",
        encoding="utf-8",
    )

    records = load_records(table, images)

    assert records[0].id == "Ford_F150_PK-XL_6195_2029_1971"
    assert records[0].image_path == (images / "truck.png").resolve()
    assert records[0].size == "PK-XL"


def test_rows_with_empty_image_path_are_skipped(tmp_path):
    image_path = tmp_path / "truck.png"
    Image.new("RGB", (10, 10)).save(image_path)
    table = tmp_path / "vehicles.tsv"
    table.write_text(
        "name\tSize\timage_path\tlength_mm\twidth_mm\theight_mm\n"
        "Missing Truck\tPK-XL\t   \t6000\t2000\t2000\n"
        "Available Truck\tPK-XL\ttruck.png\t6100\t2100\t2050\n",
        encoding="utf-8",
    )

    records = load_records(table)

    assert len(records) == 1
    assert records[0].name == "Available Truck"
    assert records[0].image_path == image_path.resolve()


def test_all_empty_image_paths_produce_no_records(tmp_path):
    table = tmp_path / "vehicles.tsv"
    table.write_text(
        "name\tSize\timage_path\tlength_mm\twidth_mm\theight_mm\n"
        "Missing Truck\tPK-XL\t\t6000\t2000\t2000\n",
        encoding="utf-8",
    )

    assert load_records(table) == []


def test_generated_id_normalizes_filename_unsafe_characters(tmp_path):
    image_path = tmp_path / "truck.png"
    Image.new("RGB", (10, 10)).save(image_path)
    table = tmp_path / "vehicles.tsv"
    table.write_text(
        "name\tSize\timage_path\tlength_mm\twidth_mm\theight_mm\n"
        "Ford / F150\tPK-XL\ttruck.png\t6195\t2029\t1971\n",
        encoding="utf-8",
    )

    records = load_records(table)

    assert records[0].id == "Ford___F150_PK-XL_6195_2029_1971"


def test_generated_ids_must_be_unique(tmp_path):
    table = tmp_path / "vehicles.tsv"
    table.write_text(
        "name\tSize\timage_path\tlength_mm\twidth_mm\theight_mm\n"
        "Ford F150\tPK-XL\tfirst.png\t6195\t2029\t1971\n"
        "Ford F150\tPK-XL\tsecond.png\t6195\t2029\t1971\n",
        encoding="utf-8",
    )

    try:
        load_records(table)
    except ValueError as exc:
        assert "Duplicate vehicle IDs" in str(exc)
    else:
        raise AssertionError("Expected duplicate generated IDs to be rejected")


def test_palette_transparency_is_composited_on_white(tmp_path):
    image_path = tmp_path / "transparent.png"
    image = Image.new("P", (2, 1), 0)
    palette = [0] * 768
    palette[3:6] = [255, 0, 0]
    image.putpalette(palette)
    image.putpixel((1, 0), 1)
    image.save(image_path, transparency=0)

    loaded = load_image(image_path)

    assert loaded.getpixel((0, 0)) == (255, 255, 255)
    assert loaded.getpixel((1, 0)) == (255, 0, 0)


def test_avif_is_discovered_and_converted_to_rgb(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    generated_id = "AVIF_Truck_TEST_6000_2000_2000"
    image_path = images / f"{generated_id}.avif"
    Image.new("RGB", (12, 8), (220, 30, 20)).save(image_path, format="AVIF")
    table = tmp_path / "vehicles.tsv"
    table.write_text(
        "name\tSize\tlength_mm\twidth_mm\theight_mm\n"
        "AVIF Truck\tTEST\t6000\t2000\t2000\n",
        encoding="utf-8",
    )

    record = load_records(table)[0]
    loaded = load_image(record.image_path)

    assert record.image_path == image_path.resolve()
    assert loaded.mode == "RGB"
    assert loaded.size == (12, 8)
