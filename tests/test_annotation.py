import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw

from pickup_measure.src.annotation import (
    _chassis_line,
    measure_and_trace,
)
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
    render_annotated_svg(image, geometry, output, 6000, 2000, model_name="Test Vehicle")

    svg = output.read_text(encoding="utf-8")
    assert "#C8242A" in svg
    assert "LENGTH  6000" in svg
    assert "6000 mm" not in svg
    assert "Test Vehicle" in svg
    assert 'viewBox="0 0 7750 2840"' in svg
    root = ET.fromstring(svg)
    embedded_image = root.find("{http://www.w3.org/2000/svg}image")
    assert embedded_image is not None
    assert float(embedded_image.attrib["x"]) == 750.0
    assert float(embedded_image.attrib["y"]) == 220.0
    assert float(embedded_image.attrib["width"]) == 6000.0
    assert float(embedded_image.attrib["height"]) == 2000.0
    model_label = next(
        node for node in root.iter("{http://www.w3.org/2000/svg}text")
        if node.text == "Test Vehicle"
    )
    assert float(model_label.attrib["font-size"]) == 82.0
    assert geometry.cab_height_mm > geometry.bed_height_mm
    assert len(geometry.outline_segments_mm) == 3


def test_chassis_line_preserves_a_sloped_lower_body_edge():
    image = Image.new("RGB", (600, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.polygon(
        [(30, 75), (570, 90), (570, 187), (30, 165)],
        fill="#777777",
    )

    left_y, right_y = _chassis_line(image, fallback_y=210)

    assert 158 <= left_y <= 170
    assert 180 <= right_y <= 192
    assert right_y - left_y >= 15


def test_chassis_line_ignores_a_short_ground_shadow():
    image = Image.new("RGB", (400, 140), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((20, 35, 380, 105), fill="#888888")
    draw.line((25, 105, 375, 105), fill="#222222", width=2)
    draw.ellipse((155, 116, 245, 130), fill="#999999")

    left_y, right_y = _chassis_line(image, fallback_y=130)

    assert 102 <= left_y <= 108
    assert left_y == right_y
