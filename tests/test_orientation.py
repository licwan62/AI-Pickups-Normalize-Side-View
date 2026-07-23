from PIL import Image, ImageDraw

from pickup_measure.src.orientation import normalize_front_to_right


def _vehicle(front: str) -> Image.Image:
    image = Image.new("RGB", (600, 260), "white")
    draw = ImageDraw.Draw(image)
    # Canonical test vehicle faces left: low hood at left, sloped windshield,
    # high tail at right, two wheels, a mirror at the A-pillar and a red tail lamp.
    draw.polygon(
        [(20, 125), (135, 105), (220, 45), (510, 45), (575, 70),
         (580, 190), (20, 190)],
        fill=(95, 115, 135),
    )
    draw.ellipse((105, 155, 195, 245), fill=(25, 25, 25))
    draw.ellipse((430, 155, 520, 245), fill=(25, 25, 25))
    draw.rectangle((187, 105, 228, 125), fill=(35, 35, 35))
    draw.rectangle((560, 85, 582, 115), fill=(210, 20, 25))
    return image if front == "left" else image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)


def test_left_facing_vehicle_is_flipped_to_the_right():
    normalized, result = normalize_front_to_right(_vehicle("left"))

    assert result.detected_front == "left"
    assert result.normalized_front == "right"
    assert result.flipped is True
    # The red tail lamp moves to the left after normalization.
    assert normalized.getpixel((28, 95))[0] > 150


def test_right_facing_vehicle_is_not_flipped():
    _, result = normalize_front_to_right(_vehicle("right"))

    assert result.detected_front == "right"
    assert result.normalized_front == "right"
    assert result.flipped is False
