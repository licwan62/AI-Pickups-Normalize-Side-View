from PIL import Image, ImageDraw

from pickup_measure.src.annotation import measure_and_trace
from pickup_measure.src.renderer import render_annotated_svg
from pickup_measure.src.scaler import ScaleMapping


def test_annotation_geometry_and_svg(tmp_path):
    image = Image.new("RGB", (600, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 90, 570, 190), fill="gray")
    draw.rectangle((230, 40, 420, 100), fill="gray")
    draw.ellipse((100, 150, 200, 239), fill="black")
    draw.ellipse((400, 150, 500, 239), fill="black")
    mapping = ScaleMapping.from_dimensions(600, 240, 6000, 2000)

    geometry = measure_and_trace(image, mapping, 2000)
    output = tmp_path / "annotated.svg"
    render_annotated_svg(image, geometry, output, 6000, 2000)

    svg = output.read_text(encoding="utf-8")
    assert "#C8242A" in svg
    assert "LENGTH  6000" in svg
    assert "6000 mm" not in svg
    assert geometry.cab_height_mm > geometry.bed_height_mm
    assert len(geometry.outline_segments_mm) == 3
    assert geometry.cab_start_x_mm < geometry.door_seam_x_mm < geometry.neck_x_mm
    assert geometry.door_to_front_mm > 0
