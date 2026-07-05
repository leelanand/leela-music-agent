import os
import shutil
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PEXELS_API_KEY    = os.getenv("PEXELS_API_KEY")

OUTPUT_DIR = Path(__file__).parent / "output"
CLIPS_DIR  = Path(__file__).parent / "clips"

def _find_ffmpeg() -> Path:
    if shutil.which("ffmpeg"):
        return Path(shutil.which("ffmpeg")).parent
    winget_bin = Path(r"C:\Users\leela\AppData\Local\Microsoft\WinGet\Packages")
    for p in winget_bin.glob("Gyan.FFmpeg*/**/bin"):
        if (p / "ffmpeg.exe").exists():
            return p
    # Fallback: check fresh install at C:\ffmpeg\bin
    if Path(r"C:\ffmpeg\bin\ffmpeg.exe").exists():
        return Path(r"C:\ffmpeg\bin")
    raise RuntimeError("FFmpeg not found.")

FFMPEG_DIR = _find_ffmpeg()
FFMPEG     = FFMPEG_DIR / "ffmpeg.exe"
FFPROBE    = FFMPEG_DIR / "ffprobe.exe"

VIDEO_DURATION = 900    # final video length: 15 minutes — YouTube free account standard
LOOP_SECONDS   = 180    # length of the seamless audio segment we generate in RAM;
                        # FFmpeg loops it to fill VIDEO_DURATION (keeps memory low)
W, H = 1280, 720

# Rotating moods — one per day
LOFI_MOODS = [
    ("Peaceful Study",   "lofi hip hop, peaceful, piano, study music, jazz, vinyl crackle, soft drums, 70 bpm"),
    ("Rainy Day",        "ambient lofi, rain, melancholic, piano, atmospheric, slow tempo, cinematic, 65 bpm"),
    ("Late Night City",  "late night lofi, urban, saxophone, moody, chill, mellow, 75 bpm"),
    ("Cozy Morning",     "morning lofi, warm, acoustic guitar, soft, gentle, hopeful, bright, 72 bpm"),
    ("Deep Focus",       "dark lofi, deep focus, minimal, electronic, ambient, concentration, 68 bpm"),
    ("Nostalgic",        "nostalgic lofi, vintage, warm piano, cassette tape, emotional, dreamy, 70 bpm"),
    ("Cafe Vibes",       "cafe lofi, bossa nova, jazz, warm afternoon, relaxed, acoustic, 74 bpm"),
    ("Dreamy Night",     "dreamy lofi, ethereal, ambient, soft synth, floating, peaceful, stars, 66 bpm"),
    ("Autumn Walk",      "autumn lofi, melancholic, acoustic, leaves, warm, introspective, 69 bpm"),
    ("Midnight Rain",    "midnight lofi, rain, jazz piano, moody, cinematic, emotional, 67 bpm"),
    ("Slow Jazz",        "slow jazz, mellow saxophone, brushed drums, upright bass, warm jazz piano, smoky late night jazz club, relaxing music for quiet souls, peaceful, 60 bpm"),
]

VISUAL_KEYWORDS = [
    "cozy cottage window rain countryside warm candle",
    "misty forest ancient trees ethereal soft light",
    "train window countryside golden hour landscape",
    "canal town evening lanterns reflection water",
    "meadow wildflowers morning mist sunrise",
    "old library warm lamp wooden books reading",
    "harbor small boats morning calm misty",
    "japanese countryside rice fields mountains peaceful",
    "garden path green lush peaceful sunlight",
    "countryside aerial green hills village soft light",
    "dim jazz club candlelit table warm wooden bar night cozy",
]
