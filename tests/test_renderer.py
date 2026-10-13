import xml.etree.ElementTree as ET

from PIL import Image

from pickup_measure.src.renderer import render_dimensioned_vehicle_svg


def test_vehicle_image_keeps_true_physical_dimensions(tmp_path):
    output = tmp_path / "vehicle.svg"
    render_dimensioned_vehicle_svg(
        Image.new("RGB", (30, 10), "white"), output, 6195, 1971
    )
    root = ET.parse(output).getroot()
    assert root.attrib["width"].endswith("mm")
    image = next(element for element in root.iter() if element.tag.endswith("image"))
    assert image.attrib["width"] == "6195"
    assert image.attrib["height"] == "1971"
