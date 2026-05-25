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
    raise RuntimeError("FFmpeg not found.")

FFMPEG_DIR = _find_ffmpeg()
FFMPEG     = FFMPEG_DIR / "ffmpeg.exe"
FFPROBE    = FFMPEG_DIR / "ffprobe.exe"

VIDEO_DURATION = 90   # 1.5 minutes
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
]

VISUAL_KEYWORDS = [
    "cozy room rainy window bokeh",
    "coffee shop window condensation evening",
    "night city street rain neon lights",
    "study desk lamp books night",
    "fireplace warm living room winter",
    "autumn leaves falling forest path",
    "cafe window table cup coffee",
    "bedroom fairy lights dark aesthetic",
    "rainy window drops bokeh blur",
    "misty forest morning light fog",
]
