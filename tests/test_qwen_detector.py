import json

from PIL import Image, ImageDraw
import pytest

from pickup_measure.src.geometry import Bounds, transform_perspective_points
from pickup_measure.src.qwen_detector import QwenVehicleDetector


class _FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def test_qwen_detector_converts_normalized_box(monkeypatch, tmp_path):
    captured = {}
    call_count = 0
    prompt_file = tmp_path / "promting.md"
    prompt_file.write_text("Return the vehicle box.", encoding="utf-8")

    def fake_urlopen(request, timeout):
        nonlocal call_count
        call_count += 1
        captured["request"] = request
        captured["timeout"] = timeout
        if call_count == 1:
            content = (
                '{"bbox_1000":[50,100,950,900],'
                '"boundary_touch_points_1000":{'
                '"leftmost":[50,600],"rightmost":[950,610],'
                '"topmost":[500,100]},'
                '"perspective_quad_1000":'
                '[[50,120],[940,100],[950,900],[60,880]],'
                '"wheel_centers_1000":[[220,760],[780,760]],'
                '"front":"right"}'
            )
        else:
            content = json.dumps({
                "schema": "semantic_structure_v1",
                "vehicle_type": "pickup",
                "direction": "front_right",
                "keypoints_1000": {
                    "rear_bumper_outermost": [20, 700],
                    "tailgate_top_rear": [30, 300],
                    "bed_rail_rear": [40, 280],
                    "bed_rail_front": [350, 280],
                    "bed_cab_gap_top": [355, 280],
                    "bed_cab_gap_bottom": [360, 720],
                    "cab_rear_roof": [370, 180],
                    "roof_highest": [500, 130],
                    "cab_roof_front": [650, 180],
                    "windshield_base": [730, 390],
                    "hood_rear_top": [735, 400],
                    "hood_front_top": [930, 450],
                    "grille_front_top": [960, 500],
                    "front_bumper_outermost": [980, 700],
                    "rocker_rear": [370, 720],
                    "rocker_front": [730, 710],
                    "rear_tire_contact": [220, 950],
                    "front_tire_contact": [800, 950],
                    "rear_wheel_center": [220, 760],
                    "front_wheel_center": [800, 760],
                },
                "polylines_1000": {
                    "bed_top_line": [[40, 280], [200, 278], [350, 280]],
                    "cab_roof_line": [[370, 180], [500, 130], [650, 180]],
                    "cab_rear_line": [[360, 720], [365, 400], [370, 180]],
                    "windshield_line": [[650, 180], [690, 280], [730, 390]],
                    "hood_top_line": [[735, 400], [830, 420], [930, 450]],
                    "front_profile_line": [[930, 450], [960, 500], [980, 700]],
                    "rocker_line": [[370, 720], [550, 715], [730, 710]],
                    "vehicle_lower_body_line": [
                        [20, 720], [360, 720], [730, 710], [980, 700]
                    ],
                    "ground_line": [[220, 950], [800, 950]],
                    "wheel_center_line": [[220, 760], [800, 760]],
                },
            })
        return _FakeResponse({
            "choices": [{
                "message": {
                    "content": content
                }
            }]
        })

    monkeypatch.setattr(
        "pickup_measure.src.qwen_detector.urlopen",
        fake_urlopen,
    )
    detector = QwenVehicleDetector(
        api_key="secret",
        prompt_file=prompt_file,
        endpoint="https://example.test/chat/completions",
        timeout_seconds=12,
        perspective_correction=True,
    )

    bounds = detector.detect(Image.new("RGB", (1000, 500), "white"))

    assert bounds.left == 50
    assert bounds.right == 950
    assert bounds.roof == 50
    assert bounds.ground == 450
    assert detector.last_front == "right"
    assert detector.last_boundary_touch_points == {
        "leftmost": (50.0, 300.0),
        "rightmost": (950.0, 305.0),
        "topmost": (500.0, 50.0),
    }
    assert detector.last_perspective_quad == (
        (50.0, 60.0),
        (940.0, 50.0),
        (950.0, 450.0),
        (60.0, 440.0),
    )
    assert captured["timeout"] == 12
    request_payload = json.loads(captured["request"].data.decode("utf-8"))
    image_url = request_payload["messages"][0]["content"][0]["image_url"]["url"]
    assert image_url.startswith("data:image/jpeg;base64,")
    localization_prompt = request_payload["messages"][0]["content"][1]["text"]
    assert localization_prompt == "Return the vehicle box."
    assert captured["request"].headers["Authorization"] == "Bearer secret"


def test_qwen_detector_requires_api_key():
    detector = QwenVehicleDetector(api_key="")

    with pytest.raises(RuntimeError, match="detection.api_key"):
        detector.detect(Image.new("RGB", (100, 100), "white"))


def test_qwen_body_chassis_line_is_a_long_near_horizontal_hint():
    line = QwenVehicleDetector._body_chassis_line_from_payload(
        {"body_chassis_line_1000": [[180, 720], [820, 725]]},
        1000,
        500,
        required=True,
    )

    assert line == [(180.0, 360.0), (820.0, 362.5)]


def test_qwen_detector_rejects_legacy_prompt_template_copy():
    payload = {
        "bbox_1000": [40, 120, 960, 900],
        "perspective_quad_1000": [
            [300, 260], [700, 250], [710, 720], [290, 730]
        ],
        "wheel_centers_1000": [[250, 760], [800, 750]],
    }

    with pytest.raises(RuntimeError, match="placeholder coordinates"):
        QwenVehicleDetector._reject_prompt_template_copy(payload)


def test_qwen_body_chassis_line_parser():
    parsed = QwenVehicleDetector._body_chassis_line_from_payload(
        {"body_chassis_line_1000": [[800, 700], [250, 710]]},
        image_width=1200,
        image_height=600,
    )

    assert parsed == [(300.0, 426.0), (960.0, 420.0)]


def test_perspective_quad_uses_real_wheel_axis_slope():
    quad = (
        (250.0, 180.0),
        (750.0, 170.0),
        (750.0, 420.0),
        (250.0, 430.0),
    )

    aligned = QwenVehicleDetector._align_quad_to_wheel_axis(
        quad,
        [(200.0, 350.0), (800.0, 380.0)],
        [(200.0, 500.0), (800.0, 512.0)],
    )

    center_slope = 30.0 / 600.0
    contact_slope = 12.0 / 600.0
    top_slope = (aligned[1][1] - aligned[0][1]) / (
        aligned[1][0] - aligned[0][0]
    )
    bottom_slope = (aligned[2][1] - aligned[3][1]) / (
        aligned[2][0] - aligned[3][0]
    )
    assert top_slope == pytest.approx(center_slope)
    assert bottom_slope == pytest.approx(contact_slope)

    bounds = Bounds(left=0, right=1000, roof=0, ground=700)
    mapped_centers = transform_perspective_points(
        [(200.0, 350.0), (800.0, 380.0)],
        aligned,
        bounds,
    )
    mapped_contacts = transform_perspective_points(
        [(200.0, 500.0), (800.0, 512.0)],
        aligned,
        bounds,
    )
    assert mapped_centers[0][1] == pytest.approx(
        mapped_centers[1][1],
        abs=0.01,
    )
    assert mapped_contacts[0][1] == pytest.approx(
        mapped_contacts[1][1],
        abs=0.01,
    )


def test_wheel_contacts_do_not_expand_to_a_lower_road_edge():
    image = Image.new("RGB", (400, 220), "white")
    draw = ImageDraw.Draw(image)
    draw.ellipse((62, 92, 138, 168), outline="black", width=5)
    draw.ellipse((262, 97, 338, 173), outline="black", width=5)
    draw.line((0, 205, 399, 205), fill="black", width=5)
    bounds = Bounds(left=0, right=400, roof=20, ground=210)

    contacts = QwenVehicleDetector._image_wheel_contacts(
        image,
        bounds,
        [(100.0, 130.0), (300.0, 135.0)],
    )

    assert contacts is not None
    assert contacts[0][1] == pytest.approx(168, abs=3)
    assert contacts[1][1] == pytest.approx(173, abs=3)


def test_qwen_detector_rejects_vehicle_outline_as_perspective_quad():
    payload = {
        "bbox_1000": [6, 102, 994, 978],
        "perspective_quad_1000": [
            [412, 102], [994, 102], [994, 978], [6, 978]
        ],
    }

    with pytest.raises(RuntimeError, match="implausibly strong keystone"):
        QwenVehicleDetector._quad_from_payload(payload, 400, 148)


def test_qwen_detector_rejects_invalid_box(monkeypatch, tmp_path):
    prompt_file = tmp_path / "promting.md"
    prompt_file.write_text("Return the vehicle box.", encoding="utf-8")
    monkeypatch.setattr(
        "pickup_measure.src.qwen_detector.urlopen",
        lambda request, timeout: _FakeResponse({
            "choices": [{
                "message": {
                    "content": '{"bbox_1000":[900,100,100,900]}'
                }
            }]
        }),
    )
    detector = QwenVehicleDetector(api_key="secret", prompt_file=prompt_file)

    with pytest.raises(RuntimeError, match="invalid"):
        detector.detect(Image.new("RGB", (1000, 500), "white"))


def test_qwen_detector_requires_perspective_quad(monkeypatch, tmp_path):
    prompt_file = tmp_path / "promting.md"
    prompt_file.write_text("Return the vehicle box.", encoding="utf-8")
    monkeypatch.setattr(
        "pickup_measure.src.qwen_detector.urlopen",
        lambda request, timeout: _FakeResponse({
            "choices": [{
                "message": {
                    "content": '{"bbox_1000":[50,100,950,900]}'
                }
            }]
        }),
    )
    detector = QwenVehicleDetector(
        api_key="secret",
        prompt_file=prompt_file,
        perspective_correction=True,
    )

    with pytest.raises(RuntimeError, match="perspective_quad_1000"):
        detector.detect(Image.new("RGB", (1000, 500), "white"))


def test_qwen_detector_repairs_incomplete_first_response(monkeypatch, tmp_path):
    prompt_file = tmp_path / "promting.md"
    prompt_file.write_text("Return all required vehicle fields.", encoding="utf-8")
    responses = iter([
        _FakeResponse({
            "choices": [{
                "message": {
                    "content": (
                        '{"vehicle_type":"pickup",'
                        '"bbox_1000":[50,100,950,900],'
                        '"perspective_quad_1000":'
                        '[[50,100],[950,100],[950,900],[50,900]]}'
                    )
                }
            }]
        }),
        _FakeResponse({
            "choices": [{
                "message": {
                    "content": (
                        '{"vehicle_type":"pickup",'
                        '"bbox_1000":[50,100,950,900],'
                        '"perspective_quad_1000":'
                        '[[250,200],[750,180],[760,760],[240,780]],'
                        '"parts":{'
                        '"bed":{"points":[{"x":500,"y":300},{"x":900,"y":300},'
                        '{"x":900,"y":750},{"x":500,"y":750}]},'
                        '"cab":[250,180,500,200,500,750,250,750],'
                        '"hood":[[60,400],[250,300],[250,750],[60,750]],'
                        '"chassis":{"left":[60,750],"right":[900,750]},'
                        '"wheel_centers":[[220,771],[780,749]]},'
                        '"front":"left"}'
                    )
                }
            }]
        }),
    ])
    monkeypatch.setattr(
        "pickup_measure.src.qwen_detector.urlopen",
        lambda request, timeout: next(responses),
    )
    detector = QwenVehicleDetector(api_key="secret", prompt_file=prompt_file)

    bounds = detector.detect(Image.new("RGB", (1000, 500), "white"))

    assert bounds.left == 50
    assert len(detector.last_response_contents) == 2


def test_qwen_detector_skips_perspective_quad_by_default(monkeypatch, tmp_path):
    prompt_file = tmp_path / "promting.md"
    prompt_file.write_text("Return the vehicle box.", encoding="utf-8")
    monkeypatch.setattr(
        "pickup_measure.src.qwen_detector.urlopen",
        lambda request, timeout: _FakeResponse({
            "choices": [{
                "message": {
                    "content": (
                        '{"bbox_1000":[50,100,950,900],'
                        '"boundary_touch_points_1000":{'
                        '"leftmost":[50,600],"rightmost":[950,610],'
                        '"topmost":[500,100]},'
                        '"wheel_centers_1000":[[220,760],[780,760]],'
                        '"front":"right"}'
                    )
                }
            }]
        }),
    )
    detector = QwenVehicleDetector(api_key="secret", prompt_file=prompt_file)

    bounds = detector.detect(Image.new("RGB", (1000, 500), "white"))

    assert bounds.left == 50
    assert bounds.right == 950
    assert detector.last_perspective_quad is None
    assert detector.last_image_wheel_centers is None
    assert detector.last_image_wheel_contacts is None


def test_trim_antenna_from_bounds_detects_thin_protrusion():
    """A narrow vertical strip at the top of the bbox should be trimmed."""
    img = Image.new("RGB", (200, 100), "white")
    draw = ImageDraw.Draw(img)
    # Vehicle body: wide rectangle from y=20 to y=90
    draw.rectangle([10, 20, 190, 90], fill="gray")
    # Antenna: thin vertical line from y=0 to y=20 at x=100
    draw.line([100, 0, 100, 20], fill="black", width=2)

    bounds = Bounds(left=10, right=190, roof=0, ground=90)
    trimmed = QwenVehicleDetector._trim_antenna_from_bounds(img, bounds)

    assert trimmed.roof == 20
    assert trimmed.left == 10
    assert trimmed.right == 190
    assert trimmed.ground == 90


def test_trim_antenna_from_bounds_no_antenna():
    """When the bbox is already tight, bounds should be unchanged."""
    img = Image.new("RGB", (200, 100), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 190, 90], fill="gray")

    bounds = Bounds(left=10, right=190, roof=10, ground=90)
    trimmed = QwenVehicleDetector._trim_antenna_from_bounds(img, bounds)

    assert trimmed.roof == 10
    assert trimmed == bounds


def test_trim_antenna_preserves_short_sloping_roof_top():
    """The crop starts at the roof, not where the curved roof becomes wide."""
    img = Image.new("RGB", (220, 120), "white")
    draw = ImageDraw.Draw(img)
    # The roof begins with a short plateau and then widens gradually.  Its first
    # row is intentionally much narrower than 15% of the vehicle bbox.
    draw.polygon(
        [(90, 20), (102, 20), (150, 38), (205, 105), (15, 105), (75, 38)],
        fill=(220, 220, 220),
    )
    draw.line([45, 0, 45, 62], fill="black", width=2)

    bounds = Bounds(left=10, right=210, roof=0, ground=105)
    trimmed = QwenVehicleDetector._trim_antenna_from_bounds(img, bounds)

    assert trimmed.roof == 20


def test_trim_antenna_skips_environment_background():
    img = Image.new("RGB", (200, 100), "gray")
    bounds = Bounds(left=10, right=190, roof=5, ground=90)

    trimmed = QwenVehicleDetector._trim_antenna_from_bounds(
        img,
        bounds,
        background_type="environment",
    )

    assert trimmed == bounds
