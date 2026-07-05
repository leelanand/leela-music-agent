"""
Lo-fi music generator using MusicGen (Meta AI).
Generates genuinely creative, varied music from a text prompt.
First run downloads ~1.7GB model — cached afterwards.
"""

import subprocess
import numpy as np
import scipy.io.wavfile
import torch
from pathlib import Path
from datetime import date
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from config import FFMPEG, LOOP_SECONDS

MODEL_ID = "facebook/musicgen-small"

MOOD_MAP = {
    "slowjazz":  ["slow jazz", "smoky", "quiet souls", "jazz club"],
    "nocturne":  ["night", "city", "neon", "urban", "focus", "study", "late"],
    "rain":      ["rain", "midnight", "dark", "moody", "melancholic"],
    "dusk":      ["autumn", "leaves", "nostalgic", "vintage", "evening", "dusk"],
    "midnight":  ["dream", "ethereal", "ambient", "floating", "deep"],
    "nostalgia": ["morning", "cafe", "warm", "cozy", "peaceful", "bossa"],
}

# Three prompt variations per mood — rotate by day so same mood sounds different each time
MOOD_PROMPTS = {
    "nostalgia": [
        "lofi hip hop, warm jazz piano, bossa nova feel, soft boom bap drums, vinyl crackle, cozy cafe afternoon, upbeat and bright, 78 bpm, acoustic bass, jazzy chord changes",
        "lofi hip hop, solo acoustic piano, gentle guitar strumming, warm sunlight, hopeful and relaxed, 76 bpm, nostalgic, easy listening, afternoon coffee",
        "lofi hip hop, jazz piano comping, upright bass walk, light brushed snare, 78 bpm, smooth and intimate, vintage record player warmth",
    ],
    "midnight": [
        "lofi ambient, dark atmospheric rhodes piano, very sparse slow drums, heavy plate reverb, gentle rain, dreamy and emotional, slow 68 bpm, spacious, melancholic, floating",
        "lofi ambient, solo piano midnight, deep reverb, introspective and lonely, 65 bpm, very slow, cinematic, rain on glass, beautiful sadness",
        "lofi ambient, electric piano pads, soft synth texture, distant muffled drums, hazy dreamlike, 68 bpm, ethereal, late night thoughts, stars",
    ],
    "nocturne": [
        "lofi hip hop, late night jazz piano, muted trumpet stabs, urban night atmosphere, soft boom bap drums, vinyl crackle, 72 bpm, moody, sophisticated",
        "lofi hip hop, jazz piano trio, walking bass, brushed drums, city rain outside, 70 bpm, blue note feel, cool and moody, late night session",
        "lofi hip hop, piano and saxophone melody, boom bap groove, 73 bpm, smoky jazz club, dim lights, vinyl warmth, cinematic night",
    ],
    "rain": [
        "lofi ambient piano, melancholic, heavy rain sounds, slow cinematic, sad and reflective, 70 bpm, reverb drenched, minimal drums, emotional solo piano",
        "lofi ambient, piano and cello, rain storm, very slow 67 bpm, deeply emotional, romantic sadness, lush reverb, cinematic sorrow",
        "lofi ambient, sparse piano notes, thunderstorm background, 70 bpm, contemplative, minimal, each note hangs in air, meditative",
    ],
    "dusk": [
        "lofi hip hop, nostalgic jazz piano, acoustic guitar texture, warm autumn evening, soft brushed drums, 74 bpm, sentimental, flowing, chill",
        "lofi hip hop, piano and guitar duo, golden hour warmth, falling leaves feeling, 72 bpm, bittersweet, end of day, peaceful melancholy",
        "lofi hip hop, vintage jazz piano, light percussion, warm analog tape, 75 bpm, nostalgic afternoon, fading light, gentle and reflective",
    ],
    "slowjazz": [
        "slow smooth jazz, mellow tenor saxophone melody, soft brushed drums, warm upright bass, gentle jazz piano comping, smoky late night jazz club, 58 bpm, intimate and relaxing, music for quiet souls",
        "slow jazz ballad, expressive muted trumpet, warm rhodes piano, brushed snare, walking double bass, candlelit lounge, 60 bpm, tender and peaceful, deeply calming",
        "soft slow jazz trio, gentle piano, light brushed drums, mellow upright bass, occasional saxophone, vinyl warmth, 62 bpm, sophisticated and serene, peaceful moments, easy listening",
    ],
}

GEN_SECONDS = 32   # MusicGen Small max is ~38s; use 32s clips with smooth crossfades


def _detect_mood(prompt: str) -> str:
    p = prompt.lower()
    for key, keywords in MOOD_MAP.items():
        if any(kw in p for kw in keywords):
            return key
    return "nocturne"


def generate_lofi_track(prompt: str, output_path: Path) -> Path:
    print(f"[2/5] Generating lo-fi music with MusicGen AI ({LOOP_SECONDS}s seamless loop)...")

    mood_key    = _detect_mood(prompt)
    variations  = MOOD_PROMPTS.get(mood_key, MOOD_PROMPTS["nocturne"])

    # Daily seed — changes every day, different per mood
    day_seed    = int(date.today().strftime("%Y%m%d"))
    mood_offset = list(MOOD_PROMPTS.keys()).index(mood_key) * 1000
    combined    = (day_seed + mood_offset) % (2 ** 31)
    torch.manual_seed(combined)

    # Rotate prompt variation by day — same mood sounds different each time it recurs
    music_desc = variations[day_seed % len(variations)]

    print(f"   Mood    : {mood_key}")
    print(f"   Prompt  : {music_desc[:80]}...")
    print(f"   Loading MusicGen model (first run downloads ~1.7GB, then cached)...")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model     = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID)
    model.eval()

    frame_rate = model.config.audio_encoder.frame_rate   # tokens per second (~50)
    sr         = model.config.audio_encoder.sampling_rate  # 32000 Hz
    max_tokens = int(GEN_SECONDS * frame_rate)

    inputs = processor(text=[music_desc], padding=True, return_tensors="pt")

    print(f"   Generating {GEN_SECONDS}s clip (5-15 min on CPU — runs in background daily)...")
    with torch.no_grad():
        audio_values = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            guidance_scale=3.0,
        )

    audio_np = audio_values[0, 0].numpy().astype(np.float32)
    actual_s = len(audio_np) / sr
    print(f"   Clip done ({actual_s:.1f}s). Building {LOOP_SECONDS}s seamless loop segment...")

    # Equal-power overlap-add loop — no FFmpeg crossfade quirks, no silence.
    # We only build a short LOOP_SECONDS segment in RAM; the editor repeats it
    # with FFmpeg -stream_loop to reach the full hour without a huge allocation.
    target      = int(sr * LOOP_SECONDS)
    cf_samples  = int(sr * 12)                         # 12s crossfade (eliminate loop clicks)
    step        = len(audio_np) - cf_samples           # advance per copy
    t           = np.linspace(0, np.pi / 2, cf_samples)
    fade_out_c  = np.cos(t).astype(np.float32)         # 1 → 0
    fade_in_c   = np.sin(t).astype(np.float32)         # 0 → 1

    result = np.zeros(target, dtype=np.float32)
    pos    = 0
    first  = True
    while pos < target:
        end   = min(pos + len(audio_np), target)
        chunk = audio_np[: end - pos].copy()

        if not first:                                  # fade in at start
            fi = min(cf_samples, len(chunk))
            chunk[:fi] *= fade_in_c[:fi]

        if pos + len(audio_np) <= target:              # fade out at end
            fo_s = len(audio_np) - cf_samples
            fo_e = fo_s + cf_samples
            if fo_e <= len(chunk):
                chunk[fo_s:fo_e] *= fade_out_c

        result[pos:end] += chunk
        pos  += step
        first = False

    # No final fade here — this segment is looped by the editor, so it must
    # start and end at full level. The overall fade-out is applied once at the
    # very end of the full-length video in editor.assemble_video().

    # Write looped wav then encode to mp3
    clip_path = Path(str(output_path) + "_loop.wav")
    scipy.io.wavfile.write(str(clip_path), rate=sr, data=result)

    mp3_path = Path(output_path).with_suffix(".mp3")
    r = subprocess.run([
        str(FFMPEG), "-y", "-i", str(clip_path),
        "-codec:a", "libmp3lame", "-b:a", "320k",
        str(mp3_path),
    ], capture_output=True, text=True)

    clip_path.unlink(missing_ok=True)

    if r.returncode != 0:
        raise RuntimeError(f"Audio encode failed:\n{r.stderr[-1000:]}")

    print(f"   Saved: {mp3_path.name} ({mp3_path.stat().st_size // 1024} KB)")
    return mp3_path
