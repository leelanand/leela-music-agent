"""Assemble final lo-fi video: loop background + overlay + music."""

import subprocess
from pathlib import Path
from config import FFMPEG, FFPROBE, VIDEO_DURATION, W, H


def _duration(path: Path) -> float:
    r = subprocess.run(
        [str(FFPROBE), "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


def assemble_video(bg_video: Path, overlay_png: Path,
                   audio_path: Path, output_path: Path):
    """
    Loop bg_video to VIDEO_DURATION, composite overlay, mix in audio.
    Produces a clean lo-fi music video.
    """
    print("[4/5] Assembling video...")

    r = subprocess.run([
        str(FFMPEG), "-y",
        # Loop background video indefinitely
        "-stream_loop", "-1", "-i", str(bg_video),
        # Overlay PNG (static)
        "-i", str(overlay_png),
        # Music audio
        "-i", str(audio_path),
        # Duration
        "-t", str(VIDEO_DURATION),
        # Composite: scale bg to 1280x720, overlay the PNG on top
        "-filter_complex",
        f"[0:v]scale={W}:{H},setsar=1[bg];[bg][1:v]overlay=0:0[v]",
        "-map", "[v]",
        "-map", "2:a",
        "-c:v", "libx264", "-preset", "slow", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-shortest",
        str(output_path),
    ], capture_output=True, text=True)

    if r.returncode != 0:
        raise RuntimeError(f"Video assembly failed:\n{r.stderr[-2000:]}")

    mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   Video saved: {output_path.name} ({mb:.1f} MB)")
