from __future__ import annotations

import os
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from sign_language_app.classifier import ALPHABET_LABELS, WORD_LABELS


def _safe_font(size: int):
    try:
        return ImageFont.truetype("Helvetica.ttc", size)
    except OSError:
        return ImageFont.load_default()


def _draw_tile(draw: ImageDraw.ImageDraw, rect, text: str) -> None:
    draw.rounded_rectangle(rect, radius=12, fill=(38, 45, 58), outline=(240, 208, 84), width=2)
    font = _safe_font(26)
    x0, y0, x1, y1 = rect
    tw, th = draw.textbbox((0, 0), text, font=font)[2:]
    tx = x0 + ((x1 - x0) - tw) // 2
    ty = y0 + ((y1 - y0) - th) // 2
    draw.text((tx, ty), text, font=font, fill=(245, 245, 245))


def ensure_assets(assets_root: str) -> None:
    os.makedirs(assets_root, exist_ok=True)
    icon_dir = os.path.join(assets_root, "gesture_icons")
    os.makedirs(icon_dir, exist_ok=True)

    for label in _iter_labels():
        filename = f"{label}.png" if len(label) == 1 else f"{label.replace(' ', '_')}.png"
        path = os.path.join(icon_dir, filename)
        if os.path.exists(path):
            continue

        img = Image.new("RGB", (256, 256), color=(23, 28, 34))
        draw = ImageDraw.Draw(img)
        _draw_tile(draw, (20, 20, 236, 236), label)
        draw.text((26, 218), "reference", font=_safe_font(16), fill=(195, 195, 195))
        img.save(path)

    chart_path = os.path.join(assets_root, "asl_chart.png")
    if not os.path.exists(chart_path):
        _generate_chart(chart_path)


def _generate_chart(chart_path: str) -> None:
    tile_w = 150
    tile_h = 120
    cols = 6
    rows = 5
    width = cols * tile_w + 30
    height = rows * tile_h + 60

    img = Image.new("RGB", (width, height), color=(16, 22, 28))
    draw = ImageDraw.Draw(img)
    draw.text((20, 16), "ASL Alphabet Reference", font=_safe_font(28), fill=(248, 248, 248))

    for idx, label in enumerate(ALPHABET_LABELS):
        row = idx // cols
        col = idx % cols
        x0 = 15 + col * tile_w
        y0 = 50 + row * tile_h
        _draw_tile(draw, (x0, y0, x0 + tile_w - 12, y0 + tile_h - 12), label)

    img.save(chart_path)


def _iter_labels() -> Iterable[str]:
    for letter in ALPHABET_LABELS:
        yield letter
    for word in WORD_LABELS:
        yield word
