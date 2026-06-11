"""Claude generates the song title, music prompt, visual concept, and YouTube metadata."""

import json
import re
from datetime import date
import anthropic
from config import ANTHROPIC_API_KEY, LOFI_MOODS, VISUAL_KEYWORDS, OUTPUT_DIR

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# Don't reuse a mood that was used in the last N renders — prevents near-duplicate
# videos (same mood + title) that YouTube removes as reused/duplicate content.
RECENT_MOODS_LOG = OUTPUT_DIR / "recent_moods.log"
RECENT_WINDOW    = 5


def _recent_moods() -> list[str]:
    if RECENT_MOODS_LOG.exists():
        lines = RECENT_MOODS_LOG.read_text(encoding="utf-8").splitlines()
        return [ln.split("\t")[-1].strip() for ln in lines if ln.strip()][-RECENT_WINDOW:]
    return []


def _record_mood(mood_name: str) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    with open(RECENT_MOODS_LOG, "a", encoding="utf-8") as f:
        f.write(f"{date.today().isoformat()}\t{mood_name}\n")


def _pick_mood(slot: int) -> int:
    """Pick a mood index, skipping any used in the last RECENT_WINDOW renders."""
    day    = date.today().timetuple().tm_yday + slot
    start  = day % len(LOFI_MOODS)
    recent = _recent_moods()
    for step in range(len(LOFI_MOODS)):
        idx = (start + step) % len(LOFI_MOODS)
        if LOFI_MOODS[idx][0] not in recent:
            return idx
    return start   # all moods used recently (window >= mood count) — fall back


def generate_creative_brief(slot: int = 0) -> dict:
    """Returns {title, mood_name, music_prompt, visual_keyword, description, tags}"""
    print("[1/5] Generating creative brief...")

    idx         = _pick_mood(slot)
    mood_name, music_prompt = LOFI_MOODS[idx]
    visual_kw   = VISUAL_KEYWORDS[idx % len(VISUAL_KEYWORDS)]
    _record_mood(mood_name)

    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1200,
        messages=[{
            "role": "user",
            "content": f"""You are a creative director for "Daily Lofi Beats" — a YouTube lo-fi music channel.

Today's mood: {mood_name}
Music style: {music_prompt}
Visual theme: {visual_kw}

Generate SEO-optimised metadata for today's lo-fi music video.

Return ONLY valid JSON:
{{
  "title": "Poetic lo-fi track title — 4-7 words, evocative, no generic words like 'beats' or 'vibes'. Examples: 'Rain on Glass at 2am', 'The Last Café Before Dawn', 'Fog Over the Old Library'. This is the on-screen overlay title, NOT the YouTube title.",
  "youtube_title": "YouTube search title. RULES: include '1 Hour' near the front (the video is a 1-hour mix — this is a top lofi search term). Lead with '1 Hour' + the primary search keyword in the first 5-6 words (e.g. 'Lofi Study Music', 'Relaxing Jazz Music', 'Chill Beats to Study/Relax'), then an em dash, then the poetic mood/scene. Match the keyword to today's mood/style — if the style is jazz, use 'Jazz' keywords not generic 'lofi hip hop'. Include a use-case phrase ('to Study and Relax', 'for Sleep and Focus', 'for Deep Work'). 50-70 characters. Example: 'Relaxing Jazz Music — Smoke & Candlelight to Unwind'.",
  "description": "MINIMUM 220 words, target 300-400 (YouTube needs 200+ words to index). Line 1-2: front-load the primary keyword and what this is + use case (the first sentence is what search indexes — e.g. 'Slow jazz music for relaxing, studying and peaceful evenings.'). Then a short paragraph setting the mood/scene. Then a practical paragraph listing use cases (studying, working, sleeping, reading, focus, unwinding, background ambience). Then a one-line CTA: 'Subscribe for new lo-fi music every day.' Weave today's mood, style and instruments in naturally as searchable phrases. End with one blank line then a hashtag block of 8-10 relevant tags as #hashtags matching the style (e.g. #lofi #studymusic #jazz #relaxingmusic).",
  "tags": "Generate 12-15 UNIQUE tags TAILORED to today's mood and style (do NOT reuse a fixed list). Mix: (1) broad high-volume terms (lofi, study music, relaxing music), (2) style/instrument-specific to today's mood (e.g. 'slow jazz', 'saxophone music', 'jazz piano' for a jazz mood; 'rain sounds', 'ambient piano' for rainy), (3) use-case long-tail ('music to study to', 'music for sleep', 'background music for work', 'cafe music'), (4) mood/aesthetic ('late night', 'cozy', 'calm'). Return as a JSON array of strings."
}}"""
        }]
    )

    raw = resp.content[0].text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    data = json.loads(raw)

    data["mood_name"]    = mood_name
    data["music_prompt"] = music_prompt
    data["visual_kw"]    = visual_kw

    print(f"   Title  : {data['title']}")
    print(f"   Mood   : {mood_name}")
    return data
