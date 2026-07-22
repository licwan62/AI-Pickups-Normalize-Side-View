from __future__ import annotations

import base64
import io
from pathlib import Path

import svgwrite
from PIL import Image, ImageDraw, ImageFont

from .annotation import AnnotationGeometry
from .settings import AnnotationStyle


def _png_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def render_vehicle_svg(
    image: Image.Image,
    output_path: Path,
    width_mm: float,
    height_mm: float,
) -> Path:
    """Embed the crop into an SVG whose user units and physical dimensions are millimetres."""
    if width_mm <= 0 or height_mm <= 0:
        raise ValueError("SVG dimensions must be positive")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    drawing = svgwrite.Drawing(
        filename=str(output_path),
        size=(f"{width_mm:g}mm", f"{height_mm:g}mm"),
        viewBox=f"0 0 {width_mm:g} {height_mm:g}",
        profile="full",
    )
    drawing.add(
        drawing.image(
            href=_png_data_uri(image),
            insert=(0, 0),
            size=(width_mm, height_mm),
            preserveAspectRatio="none",
        )
    )
    drawing.save(pretty=True)
    return output_path


def _arrow_dimension(
    drawing: svgwrite.Drawing,
    start: tuple[float, float],
    end: tuple[float, float],
    label: str,
    vertical: bool = False,
    colour: str = "#202124",
    stroke_width: float = 4,
    font_size: float = 64,
    font_family: str = "Arial",
) -> svgwrite.container.Group:
    group = drawing.g(stroke=colour, fill=colour)
    group.add(drawing.line(start=start, end=end, stroke_width=stroke_width))
    arrow = 28
    if vertical:
        group.add(drawing.polygon([(start[0], start[1]), (start[0] - arrow, start[1] + arrow * 1.7), (start[0] + arrow, start[1] + arrow * 1.7)]))
        group.add(drawing.polygon([(end[0], end[1]), (end[0] - arrow, end[1] - arrow * 1.7), (end[0] + arrow, end[1] - arrow * 1.7)]))
        text_x = start[0] + 105
        text_y = (start[1] + end[1]) / 2
        group.add(drawing.text(
            label, insert=(text_x, text_y), text_anchor="middle", font_size=font_size,
            font_family=font_family, stroke="none",
            transform=f"rotate(-90 {text_x} {text_y})",
        ))
    else:
        group.add(drawing.polygon([(start[0], start[1]), (start[0] + arrow * 1.7, start[1] - arrow), (start[0] + arrow * 1.7, start[1] + arrow)]))
        group.add(drawing.polygon([(end[0], end[1]), (end[0] - arrow * 1.7, end[1] - arrow), (end[0] - arrow * 1.7, end[1] + arrow)]))
        group.add(drawing.text(label, insert=((start[0] + end[0]) / 2, start[1] - 45), text_anchor="middle", font_size=font_size, font_family=font_family, stroke="none"))
    return group


def render_annotated_svg(
    image: Image.Image,
    geometry: AnnotationGeometry,
    output_path: Path,
    width_mm: float,
    height_mm: float,
    style: AnnotationStyle = AnnotationStyle(),
) -> Path:
    margin_left, margin_top, margin_right, margin_bottom = 750.0, 220.0, 1050.0, 620.0
    canvas_width = margin_left + width_mm + margin_right
    canvas_height = margin_top + height_mm + margin_bottom
    drawing = svgwrite.Drawing(
        filename=str(output_path),
        size=(f"{canvas_width:g}mm", f"{canvas_height:g}mm"),
        viewBox=f"0 0 {canvas_width:g} {canvas_height:g}",
        profile="full",
    )
    origin_x, origin_y = margin_left, margin_top
    drawing.add(drawing.rect(insert=(0, 0), size=(canvas_width, canvas_height), fill=style.background_color))
    drawing.add(drawing.image(
        href=_png_data_uri(image), insert=(origin_x, origin_y), size=(width_mm, height_mm),
        preserveAspectRatio="none", opacity=style.image_opacity,
    ))
    for segment in geometry.outline_segments_mm:
        drawing.add(drawing.polyline(
            points=[(origin_x + x, origin_y + y) for x, y in segment],
            fill="none", stroke=style.outline_color, stroke_width=style.outline_width_mm,
            stroke_linejoin="miter", stroke_linecap="square",
        ))

    door_y = origin_y + height_mm + 160
    door_x = origin_x + geometry.door_seam_x_mm
    front_x = origin_x + width_mm
    drawing.add(drawing.line((door_x, origin_y + geometry.cab_roof_y_mm), (door_x, origin_y + geometry.chassis_y_mm), stroke=style.outline_color, stroke_width=style.outline_width_mm))
    drawing.add(drawing.line((door_x, origin_y + geometry.chassis_y_mm), (door_x, door_y + 45), stroke="#777", stroke_width=2))
    drawing.add(drawing.line((front_x, origin_y + geometry.chassis_y_mm), (front_x, door_y + 45), stroke="#777", stroke_width=2))
    drawing.add(_arrow_dimension(
        drawing, (door_x, door_y), (front_x, door_y),
        f"DOOR TO FRONT  {geometry.door_to_front_mm:.0f}",
        colour=style.dimension_color, stroke_width=style.dimension_width_mm,
        font_size=style.font_size_mm,
        font_family=style.font_family,
    ))

    length_y = origin_y + height_mm + 430
    drawing.add(drawing.line((origin_x, origin_y + height_mm), (origin_x, length_y + 55), stroke="#777", stroke_width=2))
    drawing.add(drawing.line((origin_x + width_mm, origin_y + height_mm), (origin_x + width_mm, length_y + 55), stroke="#777", stroke_width=2))
    drawing.add(_arrow_dimension(drawing, (origin_x, length_y), (origin_x + width_mm, length_y), f"LENGTH  {width_mm:.0f}", colour=style.dimension_color, stroke_width=style.dimension_width_mm, font_size=style.font_size_mm, font_family=style.font_family))

    height_x = origin_x + width_mm + 750
    drawing.add(drawing.line((origin_x + width_mm, origin_y), (height_x + 55, origin_y), stroke="#777", stroke_width=2))
    drawing.add(drawing.line((origin_x + width_mm, origin_y + height_mm), (height_x + 55, origin_y + height_mm), stroke="#777", stroke_width=2))
    drawing.add(_arrow_dimension(drawing, (height_x, origin_y), (height_x, origin_y + height_mm), f"HEIGHT  {height_mm:.0f}", vertical=True, colour=style.dimension_color, stroke_width=style.dimension_width_mm, font_size=style.font_size_mm, font_family=style.font_family))

    inner_dimensions = (
        (-420.0, 0.0, geometry.bed_top_y_mm, geometry.bed_height_mm, "BED"),
        (-150.0, geometry.cab_start_x_mm, geometry.cab_roof_y_mm, geometry.cab_height_mm, "CAB"),
        (width_mm + 150, geometry.neck_x_mm, geometry.neck_y_mm, geometry.neck_height_mm, "NECK"),
        (width_mm + 420, width_mm, geometry.hood_front_y_mm, geometry.hood_height_mm, "HOOD"),
    )
    for x, reference_x, feature_y, measured_mm, label in inner_dimensions:
        extension_end_x = x + (55 if x < reference_x else -55)
        drawing.add(drawing.line(
            (origin_x + reference_x, origin_y + feature_y),
            (origin_x + extension_end_x, origin_y + feature_y),
            stroke="#777", stroke_width=2,
        ))
        drawing.add(drawing.line(
            (origin_x + reference_x, origin_y + geometry.chassis_y_mm),
            (origin_x + extension_end_x, origin_y + geometry.chassis_y_mm),
            stroke="#777", stroke_width=2,
        ))
        drawing.add(_arrow_dimension(
            drawing,
            (origin_x + x, origin_y + feature_y),
            (origin_x + x, origin_y + geometry.chassis_y_mm),
            f"{label}  {measured_mm:.0f}",
            vertical=True,
            colour=style.dimension_color,
            stroke_width=style.dimension_width_mm,
            font_size=style.font_size_mm,
            font_family=style.font_family,
        ))
    drawing.save(pretty=True)
    return output_path


def render_annotated_preview(
    image: Image.Image,
    geometry: AnnotationGeometry,
    output_path: Path,
    width_mm: float,
    height_mm: float,
    preview_width_px: int = 2400,
    style: AnnotationStyle = AnnotationStyle(),
) -> Path:
    margin_left, margin_top, margin_right, margin_bottom = 750.0, 220.0, 1050.0, 620.0
    canvas_width_mm = margin_left + width_mm + margin_right
    canvas_height_mm = margin_top + height_mm + margin_bottom
    scale = preview_width_px / canvas_width_mm
    preview_height_px = round(canvas_height_mm * scale)
    preview = Image.new("RGB", (preview_width_px, preview_height_px), style.background_color)
    draw = ImageDraw.Draw(preview)

    def point(x_mm: float, y_mm: float) -> tuple[int, int]:
        return round(x_mm * scale), round(y_mm * scale)

    origin_x, origin_y = margin_left, margin_top
    placed = image.resize(
        (round(width_mm * scale), round(height_mm * scale)), Image.Resampling.LANCZOS
    )
    placed = Image.blend(
        Image.new("RGB", placed.size, style.background_color), placed, style.image_opacity
    )
    preview.paste(placed, point(origin_x, origin_y))
    red = style.outline_color
    line_width = max(2, round(style.outline_width_mm * scale))
    for segment in geometry.outline_segments_mm:
        draw.line(
            [point(origin_x + x, origin_y + y) for x, y in segment],
            fill=red, width=line_width, joint="curve",
        )

    try:
        font = ImageFont.truetype(
            str(Path("C:/Windows/Fonts") / f"{style.font_family.replace(' ', '').lower()}.ttf"),
            max(16, round(style.font_size_mm * scale)),
        )
    except OSError:
        try:
            font = ImageFont.truetype("arial.ttf", max(16, round(style.font_size_mm * scale)))
        except OSError:
            font = ImageFont.load_default()

    def paste_rotated_text(label: str, center: tuple[int, int], colour: str) -> None:
        box = draw.textbbox((0, 0), label, font=font)
        layer = Image.new(
            "RGBA", (box[2] - box[0] + 8, box[3] - box[1] + 8), (255, 255, 255, 0)
        )
        ImageDraw.Draw(layer).text((4, 4), label, fill=colour, font=font)
        rotated = layer.rotate(90, expand=True)
        preview.paste(
            rotated,
            (center[0] - rotated.width // 2, center[1] - rotated.height // 2),
            rotated,
        )
    door_y = origin_y + height_mm + 160
    door_x = origin_x + geometry.door_seam_x_mm
    front_x = origin_x + width_mm
    draw.line(
        [point(door_x, origin_y + geometry.cab_roof_y_mm), point(door_x, origin_y + geometry.chassis_y_mm)],
        fill=red, width=line_width,
    )
    draw.line([point(door_x, origin_y + geometry.chassis_y_mm), point(door_x, door_y + 40)], fill="#777", width=1)
    draw.line([point(front_x, origin_y + geometry.chassis_y_mm), point(front_x, door_y + 40)], fill="#777", width=1)
    draw.line([point(door_x, door_y), point(front_x, door_y)], fill=style.dimension_color, width=max(1, round(style.dimension_width_mm * scale)))
    draw.text(
        point((door_x + front_x) / 2, door_y - 40),
        f"DOOR TO FRONT  {geometry.door_to_front_mm:.0f}",
        fill=style.dimension_color, font=font, anchor="ms",
    )

    length_y = origin_y + height_mm + 430
    draw.line([point(origin_x, origin_y + height_mm), point(origin_x, length_y + 40)], fill="#777", width=1)
    draw.line([point(origin_x + width_mm, origin_y + height_mm), point(origin_x + width_mm, length_y + 40)], fill="#777", width=1)
    draw.line([point(origin_x, length_y), point(origin_x + width_mm, length_y)], fill=style.dimension_color, width=max(1, round(style.dimension_width_mm * scale)))
    draw.text(point(origin_x + width_mm / 2, length_y - 50), f"LENGTH  {width_mm:.0f}", fill=style.dimension_color, font=font, anchor="ms")
    height_x = origin_x + width_mm + 750
    draw.line([point(origin_x + width_mm, origin_y), point(height_x + 40, origin_y)], fill="#777", width=1)
    draw.line([point(origin_x + width_mm, origin_y + height_mm), point(height_x + 40, origin_y + height_mm)], fill="#777", width=1)
    draw.line([point(height_x, origin_y), point(height_x, origin_y + height_mm)], fill=style.dimension_color, width=max(1, round(style.dimension_width_mm * scale)))
    paste_rotated_text(
        f"HEIGHT  {height_mm:.0f}",
        point(height_x + 45, origin_y + height_mm / 2),
        style.dimension_color,
    )

    dimensions = (
        (-420.0, 0.0, geometry.bed_top_y_mm, geometry.bed_height_mm, "BED"),
        (-150.0, geometry.cab_start_x_mm, geometry.cab_roof_y_mm, geometry.cab_height_mm, "CAB"),
        (width_mm + 150, geometry.neck_x_mm, geometry.neck_y_mm, geometry.neck_height_mm, "NECK"),
        (width_mm + 420, width_mm, geometry.hood_front_y_mm, geometry.hood_height_mm, "HOOD"),
    )
    for x, reference_x, top_y, measured, label in dimensions:
        end_x = x + (40 if x < reference_x else -40)
        draw.line([point(origin_x + reference_x, origin_y + top_y), point(origin_x + end_x, origin_y + top_y)], fill="#777", width=1)
        draw.line([point(origin_x + reference_x, origin_y + geometry.chassis_y_mm), point(origin_x + end_x, origin_y + geometry.chassis_y_mm)], fill="#777", width=1)
        draw.line(
            [point(origin_x + x, origin_y + top_y), point(origin_x + x, origin_y + geometry.chassis_y_mm)],
            fill=style.dimension_color, width=max(1, round(style.dimension_width_mm * scale)),
        )
        paste_rotated_text(
            f"{label}  {measured:.0f}",
            point(origin_x + x + 40, origin_y + (top_y + geometry.chassis_y_mm) / 2),
            style.dimension_color,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(output_path, format="PNG", optimize=True)
    return output_path
