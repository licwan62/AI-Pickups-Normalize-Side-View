from pickup_measure.src.geometry import Bounds


def test_bounds_dimensions_and_crop_box():
    bounds = Bounds(left=10, right=1510, roof=20, ground=520)
    bounds.validate(2000, 1000)
    assert bounds.pixel_width == 1500
    assert bounds.pixel_height == 500
    assert bounds.as_pillow_box() == (10, 20, 1510, 520)
