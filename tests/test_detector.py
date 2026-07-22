from PIL import Image, ImageDraw

from pickup_measure.src.detector import OpenCVVehicleDetector


def test_detects_vehicle_on_clean_background():
    image = Image.new("RGB", (600, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((90, 80, 510, 180), fill=(45, 65, 85))
    draw.rectangle((250, 45, 410, 100), fill=(45, 65, 85))
    draw.ellipse((135, 155, 205, 225), fill=(20, 20, 20))
    draw.ellipse((405, 155, 475, 225), fill=(20, 20, 20))

    bounds = OpenCVVehicleDetector().detect(image)

    assert bounds.left <= 95
    assert bounds.right >= 505
    assert bounds.roof <= 50
    assert bounds.ground >= 220
    assert bounds.ground <= 226
