import xml.etree.ElementTree as ET

from PIL import Image, ImageDraw

from pickup_measure.src.annotation import (
    _chassis_line,
    _windshield_edge,
    classify_background,
    detect_body_chassis_line,
    geometry_from_api_parts,
    geometry_from_api_structure,
    measure_and_trace,
)
from pickup_measure.src.renderer import render_annotated_svg
from pickup_measure.src.scaler import ScaleMapping


def test_background_classification_separates_white_transparent_and_environment():
    assert classify_background(Image.new("RGB", (200, 100), "white")) == "white"

    transparent = Image.new("RGBA", (200, 100), (0, 0, 0, 0))
    assert classify_background(transparent) == "transparent"

    environment = Image.new("RGB", (200, 100), (70, 110, 55))
    draw = ImageDraw.Draw(environment)
    draw.rectangle((0, 60, 199, 99), fill=(80, 75, 65))
    assert classify_background(environment) == "environment"
    assert classify_background(environment, ai_hint="white") == "environment"


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


def test_windshield_edge_follows_visible_a_pillar():
    image = Image.new("RGB", (600, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.line((370, 35, 480, 145), fill="black", width=5)

    edge = _windshield_edge(image, cab_roof_y=35)

    assert edge is not None
    top, bottom = edge
    assert abs(top[0] - 370) <= 6
    assert abs(top[1] - 35) <= 6
    assert abs(bottom[0] - 480) <= 6
    assert abs(bottom[1] - 145) <= 6


def test_windshield_edge_retry_rejects_a_bottom_below_hood():
    image = Image.new("RGB", (600, 240), "white")
    draw = ImageDraw.Draw(image)
    # Longer false continuation below the hood topology.
    draw.line((355, 20, 500, 150), fill="black", width=5)
    # Real A-pillar ending above the hood.
    draw.line((370, 20, 455, 100), fill="black", width=5)

    edge = _windshield_edge(
        image,
        cab_roof_y=20,
        maximum_bottom_y=110,
    )

    assert edge is not None
    _, bottom = edge
    assert bottom[1] <= 110


def test_ai_chassis_hint_selects_the_matching_real_edge():
    image = Image.new("RGB", (600, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.line((30, 160, 570, 160), fill="#555555", width=3)
    draw.line((30, 195, 570, 195), fill="#222222", width=3)

    detected = detect_body_chassis_line(
        image,
        ai_hint_line=[(150, 194), (450, 196)],
    )

    assert 192 <= detected[0][1] <= 198
    assert 192 <= detected[1][1] <= 198


def test_ai_chassis_hint_prefers_lower_edge_of_rocker_band():
    image = Image.new("RGB", (600, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.line((20, 160, 580, 160), fill="#777777", width=3)
    draw.line((100, 174, 500, 174), fill="#222222", width=3)

    detected = detect_body_chassis_line(
        image,
        ai_hint_line=[(100, 164), (500, 164)],
    )

    assert 171 <= detected[0][1] <= 177
    assert 171 <= detected[1][1] <= 177


def test_ai_chassis_hint_beats_a_longer_door_moulding():
    image = Image.new("RGB", (600, 240), "white")
    draw = ImageDraw.Draw(image)
    draw.line((20, 160, 580, 160), fill="#555555", width=3)
    draw.line((155, 190, 475, 190), fill="#222222", width=3)

    detected = detect_body_chassis_line(
        image,
        ai_hint_line=[(170, 189), (460, 191)],
    )

    assert 187 <= detected[0][1] <= 193
    assert 187 <= detected[1][1] <= 193


def test_api_non_pickup_geometry_omits_bed_frame_and_dimension(tmp_path):
    image = Image.new("RGB", (600, 240), "white")
    mapping = ScaleMapping.from_dimensions(600, 240, 6000, 2000)
    parts = {
        "vehicle_type": "non_pickup",
        "bed_quad": None,
        "cab_quad": [
            (120.0, 55.0),
            (390.0, 45.0),
            (410.0, 185.0),
            (110.0, 180.0),
        ],
        "hood_quad": [
            (390.0, 100.0),
            (570.0, 125.0),
            (575.0, 190.0),
            (410.0, 185.0),
        ],
        "chassis_line": [(30.0, 180.0), (575.0, 190.0)],
    }

    geometry = geometry_from_api_parts(parts, mapping, 600, 2000)
    output = tmp_path / "non-pickup.svg"
    render_annotated_svg(image, geometry, output, 6000, 2000)

    svg = output.read_text(encoding="utf-8")
    assert geometry.is_pickup is False
    assert geometry.bed_height_mm == 0
    assert len(geometry.outline_segments_mm) == 2
    assert "BED" not in svg
    cab_segment = geometry.outline_segments_mm[0]
    assert cab_segment[0][0] != cab_segment[1][0]
    assert cab_segment[2][0] != cab_segment[3][0]
    for segment in geometry.outline_segments_mm:
        for x_mm, y_mm in (segment[0], segment[3]):
            expected_y = (
                geometry.chassis_left_y_mm
                + (geometry.chassis_right_y_mm - geometry.chassis_left_y_mm)
                * (x_mm / 6000)
            )
            assert abs(y_mm - expected_y) < 1


def test_semantic_structure_generates_regions_from_points_and_polylines():
    mapping = ScaleMapping.from_dimensions(1000, 500, 6000, 2000)
    structure = {
        "schema": "semantic_structure_v1",
        "vehicle_type": "pickup",
        "direction": "front_right",
        "keypoints": {
            "rear_bumper_outermost": (20, 360),
            "tailgate_top_rear": (30, 150),
            "bed_rail_rear": (40, 140),
            "bed_rail_front": (350, 140),
            "bed_cab_gap_top": (355, 140),
            "bed_cab_gap_bottom": (360, 360),
            "cab_rear_roof": (370, 90),
            "roof_highest": (500, 65),
            "cab_roof_front": (650, 90),
            "windshield_base": (730, 195),
            "hood_rear_top": (735, 200),
            "hood_front_top": (930, 225),
            "grille_front_top": (960, 250),
            "front_bumper_outermost": (980, 350),
            "rocker_rear": (370, 360),
            "rocker_front": (730, 355),
            "rear_tire_contact": (220, 475),
            "front_tire_contact": (800, 475),
            "rear_wheel_center": (220, 380),
            "front_wheel_center": (800, 380),
        },
        "polylines": {
            "bed_top_line": [(40, 140), (200, 139), (350, 140)],
            "cab_roof_line": [(370, 90), (500, 65), (650, 90)],
            "cab_rear_line": [(360, 360), (365, 200), (370, 90)],
            "windshield_line": [(650, 90), (690, 140), (730, 195)],
            "hood_top_line": [(735, 200), (830, 210), (930, 225)],
            "front_profile_line": [(930, 225), (960, 250), (980, 350)],
            "rocker_line": [(370, 360), (550, 358), (730, 355)],
            "vehicle_lower_body_line": [
                (20, 360), (360, 360), (730, 355), (980, 350)
            ],
            "ground_line": [(220, 475), (800, 475)],
            "wheel_center_line": [(220, 380), (800, 380)],
        },
    }

    geometry = geometry_from_api_structure(structure, mapping, 1000, 2000)

    assert geometry.is_pickup is True
    assert len(geometry.outline_segments_mm) == 3
    assert all(len(segment) == 5 for segment in geometry.outline_segments_mm)
    assert geometry.region_types == {
        "bed": "rectangle",
        "cab": "trapezoid",
        "hood": "trapezoid",
    }
    assert (
        geometry.dimension_definitions["cab_height"]["baseline"]
        == "python_chassis_line"
    )
    assert geometry.dimension_definitions["bed_height"]["start_point"]
