"""Download Pexels background video and create lo-fi aesthetic overlay."""

import io
import requests
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from config import PEXELS_API_KEY, FFMPEG

W, H = 1280, 720

_FONT_BOLD = Path("C:/Windows/Fonts/arialbd.ttf")
_FONT_REG  = Path("C:/Windows/Fonts/arial.ttf")
FONT_PATH  = str(_FONT_BOLD) if _FONT_BOLD.exists() else str(_FONT_REG)

_FALLBACK_KEYWORDS = [
    "cozy room night lamp",
    "rain window bokeh",
    "coffee shop aesthetic",
    "misty forest morning",
]


def fetch_background_video(keyword: str, dest: Path) -> Path | None:
    """Download a landscape video from Pexels. Returns path or None."""
    headers = {"Authorization": PEXELS_API_KEY}

    for kw in [keyword] + _FALLBACK_KEYWORDS:
        resp = requests.get(
            "https://api.pexels.com/videos/search",
            headers=headers,
            params={"query": kw, "per_page": 5, "orientation": "landscape"},
            timeout=15,
        )
        if resp.status_code != 200:
            continue

        videos = resp.json().get("videos", [])
        for video in videos:
            # Pick HD file closest to 1280x720
            files = sorted(
                [f for f in video["video_files"] if f.get("quality") in ("hd", "sd")],
                key=lambda f: abs(f.get("width", 0) - W)
            )
            if not files:
                continue
            url = files[0]["link"]
            try:
                r = requests.get(url, timeout=120, stream=True)
                r.raise_for_status()
                dest.write_bytes(r.content)
                print(f"   Video: '{kw}' -> {dest.name} ({dest.stat().st_size // 1024} KB)")
                return dest
            except Exception:
                continue

    return None


def create_overlay(title: str, output_path: Path) -> Path:
    """
    Create a transparent PNG overlay with:
    - Vignette (dark edges, clear centre)
    - Channel name top-left
    - Song title bottom-left
    - Subtle grain texture for lo-fi film feel
    """
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    # --- Vignette ---
    vignette = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    vd = ImageDraw.Draw(vignette)
    steps = 80
    for i in range(steps):
        t     = i / steps
        alpha = int(200 * t ** 2.2)
        margin = int(i * (min(W, H) / 2) / steps)
        vd.rectangle([(margin, margin), (W - margin, H - margin)],
                     outline=(0, 0, 0, 0), fill=None)
    # Simpler solid vignette via radial gradient approximation
    for i in range(60):
        a = int(180 * (i / 60) ** 1.8)
        vignette.paste(Image.new("RGBA", (W - i*2, H - i*2), (0, 0, 0, 0)),
                       (i, i))
    # Just use a gradient rectangle overlay on all 4 edges
    vd2 = ImageDraw.Draw(img)
    for i in range(120):
        alpha = int(160 * ((120 - i) / 120) ** 2)
        vd2.rectangle([(0, 0), (W, i)],           fill=(0, 0, 0, alpha))
        vd2.rectangle([(0, H - i), (W, H)],       fill=(0, 0, 0, alpha))
        vd2.rectangle([(0, 0), (i, H)],            fill=(0, 0, 0, alpha))
        vd2.rectangle([(W - i, 0), (W, H)],        fill=(0, 0, 0, alpha))

    # --- Grain (film noise) ---
    noise = np.random.randint(0, 18, (H, W), dtype=np.uint8)
    grain = Image.fromarray(noise, mode="L").convert("RGBA")
    grain_data = grain.getdata()
    grain = Image.new("RGBA", (W, H))
    grain.putdata([(v, v, v, min(v * 2, 40)) for v, _, _, _ in grain_data])
    img = Image.alpha_composite(img, grain)

    draw = ImageDraw.Draw(img)

    # --- Channel name (top-left) ---
    brand_font = ImageFont.truetype(FONT_PATH, 18)
    draw.text((28, 26), "DAILY LOFI BEATS", font=brand_font,
              fill=(255, 255, 255, 160))

    # --- Song title (bottom-left) ---
    title_font = ImageFont.truetype(FONT_PATH, 32)
    title_upper = title.title()
    # Shadow
    draw.text((30, H - 62), title_upper, font=title_font, fill=(0, 0, 0, 180))
    draw.text((28, H - 64), title_upper, font=title_font, fill=(255, 255, 255, 220))

    # --- Subtle bottom fade ---
    for i in range(80):
        alpha = int(120 * (i / 80))
        draw.rectangle([(0, H - i - 1), (W, H - i)], fill=(0, 0, 0, alpha))

    output_path = Path(output_path)
    img.save(str(output_path), "PNG")
    return output_path
