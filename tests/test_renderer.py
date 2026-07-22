import xml.etree.ElementTree as ET

from PIL import Image

from pickup_measure.src.renderer import render_vehicle_svg


def test_svg_has_true_physical_dimensions(tmp_path):
    output = tmp_path / "vehicle.svg"
    render_vehicle_svg(Image.new("RGB", (30, 10), "white"), output, 6195, 1971)
    root = ET.parse(output).getroot()
    assert root.attrib["width"] == "6195mm"
    assert root.attrib["height"] == "1971mm"
    assert root.attrib["viewBox"] == "0 0 6195 1971"
