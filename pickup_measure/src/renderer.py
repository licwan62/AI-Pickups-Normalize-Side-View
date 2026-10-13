from __future__ import annotations

import base64
import io
from pathlib import Path

import svgwrite
from PIL import Image

from .settings import AnnotationStyle


ANNOTATED_MARGIN_LEFT_MM = 750.0
ANNOTATED_MARGIN_TOP_MM = 220.0
ANNOTATED_PADDING_RIGHT_MM = 1000.0
ANNOTATED_MARGIN_BOTTOM_MM = 620.0


def _png_data_uri(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


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


def render_dimensioned_vehicle_svg(
    image: Image.Image,
    output_path: Path,
    width_mm: float,
    height_mm: float,
    style: AnnotationStyle = AnnotationStyle(),
    model_name: str = "",
) -> Path:
    """Render the vehicle name and overall LENGTH/HEIGHT dimensions."""
    if width_mm <= 0 or height_mm <= 0:
        raise ValueError("SVG dimensions must be positive")
    margin_left = ANNOTATED_MARGIN_LEFT_MM
    margin_top = ANNOTATED_MARGIN_TOP_MM
    margin_right = ANNOTATED_PADDING_RIGHT_MM
    margin_bottom = ANNOTATED_MARGIN_BOTTOM_MM
    canvas_width = margin_left + width_mm + margin_right
    canvas_height = margin_top + height_mm + margin_bottom
    origin_x, origin_y = margin_left, margin_top
    output_path.parent.mkdir(parents=True, exist_ok=True)
    drawing = svgwrite.Drawing(
        filename=str(output_path),
        size=(f"{canvas_width:g}mm", f"{canvas_height:g}mm"),
        viewBox=f"0 0 {canvas_width:g} {canvas_height:g}",
        profile="full",
    )
    drawing.add(drawing.rect(
        insert=(0, 0),
        size=(canvas_width, canvas_height),
        fill=style.background_color,
    ))
    drawing.add(drawing.image(
        href=_png_data_uri(image),
        insert=(origin_x, origin_y),
        size=(width_mm, height_mm),
        preserveAspectRatio="none",
    ))

    length_y = origin_y + height_mm + 430
    drawing.add(drawing.line(
        (origin_x, origin_y + height_mm),
        (origin_x, length_y + 55),
        stroke="#777",
        stroke_width=2,
    ))
    drawing.add(drawing.line(
        (origin_x + width_mm, origin_y + height_mm),
        (origin_x + width_mm, length_y + 55),
        stroke="#777",
        stroke_width=2,
    ))
    drawing.add(_arrow_dimension(
        drawing,
        (origin_x, length_y),
        (origin_x + width_mm, length_y),
        f"LENGTH  {width_mm:.0f}",
        colour=style.dimension_color,
        stroke_width=style.dimension_width_mm,
        font_size=style.font_size_mm,
        font_family=style.font_family,
    ))

    height_x = origin_x + width_mm + 750
    drawing.add(drawing.line(
        (origin_x + width_mm, origin_y),
        (height_x + 55, origin_y),
        stroke="#777",
        stroke_width=2,
    ))
    drawing.add(drawing.line(
        (origin_x + width_mm, origin_y + height_mm),
        (height_x + 55, origin_y + height_mm),
        stroke="#777",
        stroke_width=2,
    ))
    drawing.add(_arrow_dimension(
        drawing,
        (height_x, origin_y),
        (height_x, origin_y + height_mm),
        f"HEIGHT  {height_mm:.0f}",
        vertical=True,
        colour=style.dimension_color,
        stroke_width=style.dimension_width_mm,
        font_size=style.font_size_mm,
        font_family=style.font_family,
    ))
    if model_name:
        drawing.add(drawing.text(
            model_name,
            insert=(origin_x + width_mm / 2, canvas_height - 45),
            text_anchor="middle",
            font_size=style.font_size_mm,
            font_family=style.font_family,
            fill=style.dimension_color,
        ))
    drawing.save(pretty=True)
    return output_path
