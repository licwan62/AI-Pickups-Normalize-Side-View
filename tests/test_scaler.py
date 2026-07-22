from pickup_measure.src.scaler import ScaleMapping, millimetres_to_pixels, png_effective_ppi


def test_independent_axis_mapping():
    mapping = ScaleMapping.from_dimensions(1500, 500, 6195, 1971)
    assert mapping.pixel_to_mm(1500, 500) == (6195, 1971)


def test_mm_to_pixels_at_72_ppi():
    assert millimetres_to_pixels(254, 72) == 720


def test_png_effective_ppi_is_nearest_pixels_per_metre():
    assert png_effective_ppi(72) == 72.009
