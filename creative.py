"""Claude generates the song title, music prompt, visual concept, and YouTube metadata."""

import json
import re
from datetime import date
import anthropic
from config import ANTHROPIC_API_KEY, LOFI_MOODS, VISUAL_KEYWORDS

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)


def generate_creative_brief() -> dict:
    """Returns {title, mood_name, music_prompt, visual_keyword, description, tags}"""
    print("[1/5] Generating creative brief...")

    day         = date.today().timetuple().tm_yday
    mood_name, music_prompt = LOFI_MOODS[day % len(LOFI_MOODS)]
    visual_kw   = VISUAL_KEYWORDS[day % len(VISUAL_KEYWORDS)]

    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        messages=[{
            "role": "user",
            "content": f"""You are a creative director for "Daily Lofi Beats" — a YouTube lo-fi music channel.

Today's mood: {mood_name}
Music style: {music_prompt}
Visual theme: {visual_kw}

Generate metadata for today's lo-fi music video.

Return ONLY valid JSON:
{{
  "title": "Poetic lo-fi track title — 4-7 words, evocative, no generic words like 'beats' or 'vibes'. Examples: 'Rain on Glass at 2am', 'The Last Café Before Dawn', 'Fog Over the Old Library'",
  "youtube_title": "YouTube title optimised for search — include keywords like 'lofi', 'study music', 'chill'. Under 60 chars.",
  "description": "2 short paragraphs: first sets the mood/scene for the listener, second is practical (study, sleep, focus, relax). End with: Subscribe for daily lo-fi music.",
  "tags": ["lofi", "lofi hip hop", "study music", "chill music", "ambient music", "focus music", "relaxing music", "lofi beats", "background music", "sleep music", "lofi study", "chill beats", "aesthetic music", "calm music", "lofi vibes"]
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
