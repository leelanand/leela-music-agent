"""
Daily Lofi Beats Agent
Generates a 2.5-minute lo-fi ambient music video and uploads to YouTube daily.

Run: python agent.py
Test: python agent.py --no-upload
"""

import argparse
import json
from datetime import date
from pathlib import Path

from config import OUTPUT_DIR, CLIPS_DIR
from creative import generate_creative_brief
from music import generate_lofi_track
from visuals import fetch_background_video, create_overlay
from editor import assemble_video
from uploader import upload_video


def run(upload: bool = True):
    OUTPUT_DIR.mkdir(exist_ok=True)
    CLIPS_DIR.mkdir(exist_ok=True)

    today = date.today().strftime("%Y-%m-%d")

    print(f"\n{'='*50}")
    print(f"  DAILY LOFI BEATS — {today}")
    print(f"{'='*50}\n")

    # Step 1: Creative brief
    brief = generate_creative_brief()

    # Step 2: Generate music
    audio_path = OUTPUT_DIR / f"{today}_music"
    audio_path = generate_lofi_track(brief["music_prompt"], audio_path)

    # Step 3: Fetch background video
    print("[3/5] Fetching background video...")
    bg_path = fetch_background_video(brief["visual_kw"], CLIPS_DIR / f"{today}_bg.mp4")
    if not bg_path:
        print("   No video found — using colour background fallback.")
        bg_path = None

    # Step 3b: Create overlay
    overlay_path = OUTPUT_DIR / f"{today}_overlay.png"
    create_overlay(brief["title"], overlay_path)
    print(f"   Overlay created.")

    # Step 3c: If no background video, generate a still colour background
    if not bg_path:
        from PIL import Image
        import subprocess
        from config import FFMPEG, W, H, VIDEO_DURATION
        still = Image.new("RGB", (W, H), (12, 14, 22))
        still_path = CLIPS_DIR / f"{today}_bg_still.jpg"
        still.save(str(still_path))
        bg_path = CLIPS_DIR / f"{today}_bg_fallback.mp4"
        subprocess.run([
            str(FFMPEG), "-y", "-loop", "1", "-i", str(still_path),
            "-t", str(VIDEO_DURATION), "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(bg_path),
        ], check=True, capture_output=True)

    # Step 4: Assemble video
    video_path = OUTPUT_DIR / f"{today}_lofi.mp4"
    assemble_video(bg_path, overlay_path, audio_path, video_path)

    # Step 5: Save metadata
    meta = {
        "title":         brief["title"],
        "youtube_title": brief["youtube_title"],
        "description":   brief["description"],
        "tags":          brief["tags"],
        "mood":          brief["mood_name"],
        "video_file":    str(video_path),
    }
    meta_path = OUTPUT_DIR / f"{today}_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n{'='*50}")
    print("  DONE!")
    print(f"{'='*50}")
    print(f"  Video : {video_path}")
    safe_title = brief['youtube_title'].encode('ascii', 'replace').decode()
    print(f"  Title : {safe_title}")

    if not upload:
        print("\n  [--no-upload] Skipping YouTube upload.")
        print(f"{'='*50}\n")
        return

    url = upload_video(
        video_path=video_path,
        title=brief["youtube_title"],
        description=brief["description"],
        tags=brief["tags"],
    )

    meta["youtube_url"] = url
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Live: {url}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-upload", action="store_true", help="Build video but skip upload")
    args = parser.parse_args()
    run(upload=not args.no_upload)
