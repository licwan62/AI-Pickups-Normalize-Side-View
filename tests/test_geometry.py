from PIL import Image

from pickup_measure.src.geometry import Bounds, rectify_perspective


def test_bounds_dimensions_and_crop_box():
    bounds = Bounds(left=10, right=1510, roof=20, ground=520)
    bounds.validate(2000, 1000)
    assert bounds.pixel_width == 1500
    assert bounds.pixel_height == 500
    assert bounds.as_pillow_box() == (10, 20, 1510, 520)


def test_rectify_perspective_maps_wheel_line_to_horizontal_rectangle():
    image = Image.new("RGB", (200, 120), "white")
    quad = (
        (30.0, 20.0),
        (160.0, 10.0),
        (180.0, 100.0),
        (20.0, 110.0),
    )

    rectified = rectify_perspective(image, quad)

    assert rectified.width == 160
    assert rectified.height == 92
