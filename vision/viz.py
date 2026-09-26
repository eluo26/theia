"""Draw vision-model boxes and labels on an image for debugging."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()

from vision.schemas import Detection, IndexedObject, norm_box_to_pixels

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    r"C:\Windows\Fonts\arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
)


def legend_name(provider: str) -> str:
    """Short label for the green boxes. It follows config.yaml provider."""
    return {"openai": "OpenAI", "xai": "xAI"}.get(provider, provider)


GROK_COLOR = (32, 140, 64)
DETECTOR_COLOR = (37, 99, 235)
TEXT_COLOR = (255, 255, 255)


def annotate_image(
    source: Path,
    objects: list[IndexedObject],
    dest: Path,
    *,
    legend: str = "OpenAI",
) -> Path:
    """Save a copy of source with a green box and label for each vision-model object."""
    with Image.open(source) as opened:
        transposed = ImageOps.exif_transpose(opened)
        image = (transposed if transposed is not None else opened).convert("RGB")
        image.load()
    draw = ImageDraw.Draw(image)
    font = _font(16)
    width, height = image.size

    for obj in objects:
        x1, y1, x2, y2 = norm_box_to_pixels(obj.box, width, height)
        draw.rectangle([x1, y1, x2, y2], outline=GROK_COLOR, width=3)
        caption = _caption(obj)
        text_box = draw.textbbox((0, 0), caption, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        top = y1 - text_height - 6
        if top < 0:
            top = y1
        draw.rectangle(
            [x1, top, min(width - 1, x1 + text_width + 8), top + text_height + 4],
            fill=GROK_COLOR,
        )
        draw.text((x1 + 4, top + 2), caption, fill=TEXT_COLOR, font=font)

    legend_box = draw.textbbox((0, 0), legend, font=font)
    legend_w = legend_box[2] - legend_box[0]
    legend_h = legend_box[3] - legend_box[1]
    draw.rectangle([8, 8, 16 + legend_w, 14 + legend_h], fill=GROK_COLOR)
    draw.text((12, 10), legend, fill=TEXT_COLOR, font=font)

    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest)
    return dest


def _caption(obj: IndexedObject) -> str:
    parts = [obj.label]
    if obj.count != 1:
        parts.append(f"x{obj.count}")
    if obj.drug_name:
        parts.append(obj.drug_name)
    text = " · ".join(parts)
    if len(text) > 80:
        return text[:77] + "..."
    return text


def annotate_detections(
    image,
    detections: list[Detection],
    dest: Path,
    ids: dict[tuple[int, int, int, int], str] | None = None,
    *,
    legend: str = "OpenAI",
) -> Path:
    """Save a debug image. Vision-model boxes are green. Detector boxes are blue."""
    if isinstance(image, np.ndarray):
        canvas = Image.fromarray(np.ascontiguousarray(image)).convert("RGB")
    else:
        canvas = Image.open(image).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = _font(16)
    width, height = canvas.size

    for det in detections:
        if (
            det.box_source == "detector"
            and det.grok_bbox_px is not None
            and list(det.grok_bbox_px) != list(det.bbox_px)
        ):
            _draw_box(
                draw,
                det.grok_bbox_px,
                GROK_COLOR,
                font,
                _detection_caption(det, ids, prefix_id=False),
                width,
            )
        color = DETECTOR_COLOR if det.box_source == "detector" else GROK_COLOR
        _draw_box(
            draw,
            det.bbox_px,
            color,
            font,
            _detection_caption(det, ids, prefix_id=True),
            width,
        )

    _legend(draw, font, legend)
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest)
    return dest


def _detection_caption(
    det: Detection,
    ids: dict[tuple[int, int, int, int], str] | None,
    *,
    prefix_id: bool,
) -> str:
    label = det.label
    if prefix_id and ids is not None:
        object_id = ids.get(tuple(int(v) for v in det.bbox_px))
        if object_id:
            label = f"{object_id} {label}"
    if det.count != 1:
        label = f"{label} x{det.count}"
    if len(label) > 80:
        return label[:77] + "..."
    return label


def _draw_box(draw, box, color, font, caption, width) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    text_box = draw.textbbox((0, 0), caption, font=font)
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    top = y1 - text_height - 6
    if top < 0:
        top = y1
    draw.rectangle(
        [x1, top, min(width - 1, x1 + text_width + 8), top + text_height + 4],
        fill=color,
    )
    draw.text((x1 + 4, top + 2), caption, fill=TEXT_COLOR, font=font)


def _legend(draw, font, vision_label: str) -> None:
    grok_box = draw.textbbox((0, 0), vision_label, font=font)
    det_box = draw.textbbox((0, 0), "detector", font=font)
    grok_w = grok_box[2] - grok_box[0]
    det_w = det_box[2] - det_box[0]
    height = max(grok_box[3] - grok_box[1], det_box[3] - det_box[1])
    draw.rectangle([8, 8, 16 + grok_w, 14 + height], fill=GROK_COLOR)
    draw.text((12, 10), vision_label, fill=TEXT_COLOR, font=font)
    left = 24 + grok_w
    draw.rectangle([left, 8, left + 8 + det_w, 14 + height], fill=DETECTOR_COLOR)
    draw.text((left + 4, 10), "detector", fill=TEXT_COLOR, font=font)


def _font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()
