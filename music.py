"""
Lo-fi music generator — boom-bap drums, jazz piano, Rhodes, ambient rain.
Spec: 70-85 BPM, dusty boom-bap, warm sub bass, vinyl crackle, Rhodes textures,
      city rain ambience, gradual layering, occasional tape-stop effects.
"""

import wave
import subprocess
import numpy as np
from scipy import signal
from pathlib import Path
from config import FFMPEG, VIDEO_DURATION

SAMPLE_RATE = 44100
BPM         = 75          # Boom-bap sweet spot — warm slow head-nod feel
TOTAL       = int(SAMPLE_RATE * VIDEO_DURATION)
BEAT_LEN    = int(SAMPLE_RATE * 60 / BPM)
BAR_LEN     = BEAT_LEN * 4

# Jazz chord progressions — jazzy 7th, 9th, minor extended voicings
PROGRESSIONS = {
    "nocturne": [   # Am9 - Fmaj13 - Cmaj9 - G13
        (55.00, [220.00, 261.63, 329.63, 415.30, 493.88]),
        (43.65, [174.61, 220.00, 261.63, 349.23, 440.00]),
        (65.41, [261.63, 329.63, 392.00, 493.88, 587.33]),
        (49.00, [196.00, 246.94, 311.13, 392.00, 493.88]),
    ],
    "rain": [       # Dm9 - G13 - Cmaj9 - Am11
        (36.71, [146.83, 185.00, 220.00, 277.18, 329.63]),
        (49.00, [196.00, 246.94, 293.66, 369.99, 440.00]),
        (65.41, [261.63, 329.63, 392.00, 493.88, 587.33]),
        (55.00, [220.00, 261.63, 329.63, 415.30, 523.25]),
    ],
    "dusk": [       # Fmaj9 - Em7 - Dm9 - Am7
        (43.65, [174.61, 220.00, 261.63, 329.63, 415.30]),
        (41.20, [164.81, 207.65, 246.94, 311.13, 392.00]),
        (36.71, [146.83, 185.00, 220.00, 277.18, 349.23]),
        (55.00, [220.00, 261.63, 329.63, 392.00, 493.88]),
    ],
    "midnight": [   # Cm9 - Abmaj7 - Ebmaj9 - Bb13
        (32.70, [130.81, 164.81, 196.00, 246.94, 311.13]),
        (51.91, [207.65, 261.63, 311.13, 392.00, 493.88]),
        (38.89, [155.56, 196.00, 233.08, 293.66, 369.99]),
        (58.27, [233.08, 293.66, 349.23, 440.00, 523.25]),
    ],
    "nostalgia": [  # Gmaj9 - Em9 - Cmaj7 - D13
        (49.00, [196.00, 246.94, 293.66, 369.99, 440.00]),
        (41.20, [164.81, 207.65, 246.94, 311.13, 392.00]),
        (65.41, [261.63, 329.63, 392.00, 493.88, 587.33]),
        (36.71, [146.83, 185.00, 220.00, 277.18, 349.23]),
    ],
}

MOOD_MAP = {
    "nocturne":  ["night", "city", "neon", "urban", "focus", "study", "late"],
    "rain":      ["rain", "midnight", "dark", "moody", "melancholic"],
    "dusk":      ["autumn", "leaves", "nostalgic", "vintage", "evening", "dusk"],
    "midnight":  ["dream", "ethereal", "ambient", "floating", "deep"],
    "nostalgia": ["morning", "cafe", "warm", "cozy", "peaceful", "bossa"],
}


# ── Jazz Piano ─────────────────────────────────────────────────────────────────

def _jazz_piano(freq: float, dur: float, vel: float = 0.65) -> np.ndarray:
    """Warm jazz piano — muted, soft attack, rich sustain."""
    n = int(SAMPLE_RATE * dur)
    t = np.linspace(0, dur, n, endpoint=False)

    partials = [(1.000, 1.00, 3.5, 0.5),
                (2.001, 0.42, 6.0, 0.9),
                (3.003, 0.20, 10.0, 1.4),
                (4.006, 0.10, 16.0, 2.2),
                (5.010, 0.06, 24.0, 3.5),
                (6.015, 0.03, 35.0, 5.5)]

    tone = np.zeros(n)
    for ratio, amp, fast_dc, slow_dc in partials:
        f_k = freq * ratio
        if f_k > 17000:
            continue
        env   = 0.55 * np.exp(-fast_dc * t) + 0.45 * np.exp(-slow_dc * t)
        tone += amp * np.sin(2 * np.pi * f_k * t) * env

    at = int(0.010 * SAMPLE_RATE)
    tone[:at] *= np.linspace(0, 1, at)

    # Warm low-pass — removes harshness, lo-fi piano character
    b, a = signal.butter(3, 4200 / (SAMPLE_RATE / 2), btype="low")
    tone = signal.lfilter(b, a, tone)
    return tone * vel * 0.50


# ── Rhodes Electric Piano ──────────────────────────────────────────────────────

def _rhodes(freq: float, dur: float, vel: float = 0.55) -> np.ndarray:
    """Rhodes electric piano — tine bell attack, warm sine sustain, tremolo."""
    n = int(SAMPLE_RATE * dur)
    t = np.linspace(0, dur, n, endpoint=False)

    # Fast-decaying 2nd harmonic (tine click) + long sustaining fundamental
    fund = np.sin(2 * np.pi * freq * t) * (0.25 + 0.75 * np.exp(-2.5 * t))
    tine = np.sin(2 * np.pi * freq * 2 * t) * np.exp(-7.0 * t) * 0.50
    h3   = np.sin(2 * np.pi * freq * 3 * t) * np.exp(-14.0 * t) * 0.15
    tone = fund + tine + h3

    # Tremolo (5-7 Hz wobble — classic Rhodes character)
    trem_rate  = np.random.uniform(5.0, 6.5)
    tremolo    = 1.0 - 0.22 * (0.5 + 0.5 * np.sin(2 * np.pi * trem_rate * t))
    tone      *= tremolo

    at = int(0.006 * SAMPLE_RATE)
    tone[:at] *= np.linspace(0, 1, at)
    return tone * vel * 0.42


# ── Sub Bass ──────────────────────────────────────────────────────────────────

def _sub_bass(freq: float, dur: float, vel: float = 0.80) -> np.ndarray:
    """Warm 808-style sub bass — pitch-bend attack, round deep low end."""
    t     = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    pitch = freq * (1 + 0.10 * np.exp(-18 * t))
    phase = np.cumsum(2 * np.pi * pitch / SAMPLE_RATE)
    sub   = (0.72 * np.sin(phase) +
             0.22 * np.sin(2 * phase) +
             0.06 * np.sin(3 * phase))

    at  = int(0.020 * SAMPLE_RATE)
    rel = int(0.12 * SAMPLE_RATE)
    env = np.ones(len(t))
    env[:at]  = np.linspace(0, 1, at)
    if rel < len(t):
        env[-rel:] = np.linspace(1, 0, rel)

    b, a = signal.butter(4, 190 / (SAMPLE_RATE / 2), btype="low")
    return signal.lfilter(b, a, sub * env) * vel


# ── Boom-Bap Drums ─────────────────────────────────────────────────────────────

def _boom_kick(vel: float = 0.85) -> np.ndarray:
    """Boom-bap kick — punchy 808 pitch-sweep, muted thud character."""
    dur   = 0.38
    t     = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    freq  = 175 * np.exp(-22 * t) + 50
    phase = np.cumsum(2 * np.pi * freq / SAMPLE_RATE)
    tone  = np.sin(phase)
    env   = np.exp(-7 * t)
    at    = int(0.003 * SAMPLE_RATE)
    env[:at] = np.linspace(0, 1, at)

    thud_n = int(0.018 * SAMPLE_RATE)
    thud   = np.random.randn(thud_n) * 0.12
    b, a   = signal.butter(2, 180 / (SAMPLE_RATE / 2), btype="low")
    thud   = signal.lfilter(b, a, thud) * np.linspace(1, 0, thud_n) ** 2

    kick = tone * env * vel
    kick[:thud_n] += thud
    return kick


def _muted_snare(vel: float = 0.65) -> np.ndarray:
    """Dusty boom-bap snare — muted, papery, no harsh crack."""
    dur  = 0.22
    t    = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    body = (np.sin(2 * np.pi * 215 * t) * np.exp(-28 * t) * 0.45 +
            np.sin(2 * np.pi * 305 * t) * np.exp(-38 * t) * 0.25)

    wires_n = np.random.randn(len(t))
    b, a    = signal.butter(2, [700 / (SAMPLE_RATE / 2), 5500 / (SAMPLE_RATE / 2)], btype="band")
    wires   = signal.lfilter(b, a, wires_n) * np.exp(-18 * t) * 0.40

    snare = body + wires
    b2, a2 = signal.butter(3, 7500 / (SAMPLE_RATE / 2), btype="low")
    snare  = signal.lfilter(b2, a2, snare)

    at = int(0.002 * SAMPLE_RATE)
    snare[:at] *= np.linspace(0, 1, at)
    return snare * vel


def _hihat_closed(vel: float = 0.40) -> np.ndarray:
    dur  = 0.07
    t    = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    n    = np.random.randn(len(t))
    b, a = signal.butter(3, [5500 / (SAMPLE_RATE / 2), 17000 / (SAMPLE_RATE / 2)], btype="band")
    hh   = signal.lfilter(b, a, n) * np.exp(-55 * t)
    b2, a2 = signal.butter(2, 11000 / (SAMPLE_RATE / 2), btype="low")
    return signal.lfilter(b2, a2, hh) * vel


def _hihat_open(vel: float = 0.32) -> np.ndarray:
    dur  = 0.28
    t    = np.linspace(0, dur, int(SAMPLE_RATE * dur), endpoint=False)
    n    = np.random.randn(len(t))
    b, a = signal.butter(3, [3500 / (SAMPLE_RATE / 2), 15000 / (SAMPLE_RATE / 2)], btype="band")
    hh   = signal.lfilter(b, a, n) * np.exp(-14 * t)
    b2, a2 = signal.butter(2, 9000 / (SAMPLE_RATE / 2), btype="low")
    return signal.lfilter(b2, a2, hh) * vel


def _build_boom_bap_drums() -> np.ndarray:
    """
    Classic boom-bap pattern:
    - Kick beats 1 and 3 (slight timing variation, 80% chance on 3)
    - Snare beats 2 and 4 (off-grid human feel)
    - Closed hi-hats every swung 8th note
    - Occasional open hi-hat on the 'and' of 2 or 4
    """
    track = np.zeros(TOTAL)
    pos   = 0
    bar   = 0
    SWING = 0.58   # subtle boom-bap swing (not full jazz triplet)

    while pos < TOTAL:
        for beat in range(4):
            bp = pos + beat * BEAT_LEN

            # Kick: beats 1 and 3
            if beat == 0:
                kick = _boom_kick(vel=np.random.uniform(0.80, 1.00))
                p = bp + np.random.randint(-250, 350)
                e = min(p + len(kick), TOTAL)
                if 0 <= p < TOTAL:
                    track[p:e] += kick[:e - p]
            elif beat == 2 and np.random.random() > 0.15:
                kick = _boom_kick(vel=np.random.uniform(0.65, 0.88))
                p = bp + np.random.randint(-300, 500)
                e = min(p + len(kick), TOTAL)
                if 0 <= p < TOTAL:
                    track[p:e] += kick[:e - p]

            # Snare: beats 2 and 4
            if beat in (1, 3):
                snare = _muted_snare(vel=np.random.uniform(0.58, 0.82))
                p = bp + np.random.randint(-350, 450)
                e = min(p + len(snare), TOTAL)
                if 0 <= p < TOTAL:
                    track[p:e] += snare[:e - p]

            # Hi-hat on beat
            hh = _hihat_closed(vel=np.random.uniform(0.28, 0.48))
            p  = bp + np.random.randint(-180, 180)
            e  = min(p + len(hh), TOTAL)
            if 0 <= p < TOTAL:
                track[p:e] += hh[:e - p]

            # Swung 8th hi-hat (slightly off-grid)
            hh2 = _hihat_closed(vel=np.random.uniform(0.18, 0.36))
            p   = bp + int(SWING * BEAT_LEN) + np.random.randint(-220, 220)
            e   = min(p + len(hh2), TOTAL)
            if 0 <= p < TOTAL:
                track[p:e] += hh2[:e - p]

        # Occasional open hi-hat every other bar
        if bar % 2 == 1 and np.random.random() > 0.50:
            ohh  = _hihat_open(vel=np.random.uniform(0.22, 0.38))
            bpck = np.random.choice([1, 3])
            p    = pos + bpck * BEAT_LEN + int(SWING * BEAT_LEN) + np.random.randint(-200, 200)
            e    = min(p + len(ohh), TOTAL)
            if 0 <= p < TOTAL:
                track[p:e] += ohh[:e - p]

        pos += BAR_LEN
        bar += 1

    return track


# ── Texture Layers ─────────────────────────────────────────────────────────────

def _rain_ambience(n: int) -> np.ndarray:
    """Gentle city rain on a window — filtered noise + distant low rumble."""
    noise  = np.random.randn(n)
    b, a   = signal.butter(4, [150 / (SAMPLE_RATE / 2), 2800 / (SAMPLE_RATE / 2)], btype="band")
    rain   = signal.lfilter(b, a, noise) * 0.55
    b2, a2 = signal.butter(3, 70 / (SAMPLE_RATE / 2), btype="low")
    rumble = signal.lfilter(b2, a2, np.random.randn(n)) * 0.12
    return (rain + rumble) * 0.016


def _tape_hiss(n: int) -> np.ndarray:
    """Subtle tape hiss — high-frequency background texture."""
    noise = np.random.randn(n)
    b, a  = signal.butter(2, [1800 / (SAMPLE_RATE / 2), 9000 / (SAMPLE_RATE / 2)], btype="band")
    return signal.lfilter(b, a, noise) * 0.0038


def _vinyl_crackle(n: int) -> np.ndarray:
    """Vinyl surface noise with random crackle pops."""
    noise = np.random.randn(n) * 0.0025
    for _ in range(n // 7000):
        p   = np.random.randint(0, n - 200)
        ln  = np.random.randint(12, 65)
        pop = np.random.randn(ln) * np.random.uniform(0.012, 0.038)
        pop *= np.linspace(1, 0, ln) ** 2
        noise[p:p + ln] += pop
    return noise


# ── Chord / Melody / Bass Builders ────────────────────────────────────────────

def _build_piano_chords(prog: list) -> np.ndarray:
    """Jazz piano comping — staggered voicings, warm tone."""
    track = np.zeros(TOTAL)
    pos = 0; ci = 0
    while pos < TOTAL:
        _, freqs = prog[ci % len(prog)]
        dur = BAR_LEN / SAMPLE_RATE
        for k, freq in enumerate(freqs):
            delay = int(k * 0.042 * SAMPLE_RATE)
            vel   = max(0.62 - k * 0.06, 0.32)
            note  = _jazz_piano(freq, dur * 0.88, vel=vel)
            s     = pos + delay
            e     = min(s + len(note), TOTAL)
            if s < TOTAL:
                track[s:e] += note[:e - s]
        pos += BAR_LEN; ci += 1
    return track


def _build_rhodes_layer(prog: list) -> np.ndarray:
    """Rhodes upper-register sparse comping — warmth and texture."""
    track = np.zeros(TOTAL)
    pos = 0; ci = 0
    while pos < TOTAL:
        _, freqs = prog[ci % len(prog)]
        if np.random.random() > 0.22:
            upper = freqs[-3:]
            dur   = BAR_LEN / SAMPLE_RATE * np.random.uniform(0.65, 1.0)
            off   = np.random.choice([0, BEAT_LEN, int(BEAT_LEN * 1.5)])
            for k, freq in enumerate(upper[:2]):
                delay = int(k * 0.020 * SAMPLE_RATE)
                note  = _rhodes(freq * 2, dur, vel=np.random.uniform(0.42, 0.58))
                s     = pos + off + delay
                e     = min(s + len(note), TOTAL)
                if 0 <= s < TOTAL:
                    track[s:e] += note[:e - s]
        pos += BAR_LEN; ci += 1
    return track


def _build_melody(prog: list) -> np.ndarray:
    """Sparse right-hand melody — emotional, unhurried."""
    track = np.zeros(TOTAL)
    pos = 0; ci = 0
    while pos < TOTAL:
        _, freqs = prog[ci % len(prog)]
        if np.random.random() > 0.48:
            top  = freqs[-2:]
            freq = np.random.choice(top) * 2
            dur  = np.random.uniform(0.50, 1.30)
            off  = np.random.choice([0, BEAT_LEN, BEAT_LEN * 2, int(BEAT_LEN * 2.5)])
            note = _jazz_piano(freq, dur, vel=np.random.uniform(0.52, 0.72))
            s    = pos + off
            e    = min(s + len(note), TOTAL)
            if 0 <= s < TOTAL:
                track[s:e] += note[:e - s]
        pos += BAR_LEN; ci += 1
    return track


def _build_bass_line(prog: list) -> np.ndarray:
    """Root on beat 1, chromatic walk on beat 3."""
    track = np.zeros(TOTAL)
    pos = 0; ci = 0
    while pos < TOTAL:
        root, _   = prog[ci % len(prog)]
        next_root = prog[(ci + 1) % len(prog)][0]

        note = _sub_bass(root, BEAT_LEN * 2 / SAMPLE_RATE, vel=0.85)
        e    = min(pos + len(note), TOTAL)
        track[pos:e] += note[:e - pos]

        walk = _sub_bass((root + next_root) / 2,
                          BEAT_LEN * 1.5 / SAMPLE_RATE, vel=0.68)
        p = pos + 2 * BEAT_LEN + np.random.randint(-100, 200)
        e = min(p + len(walk), TOTAL)
        if 0 <= p < TOTAL:
            track[p:e] += walk[:e - p]

        pos += BAR_LEN; ci += 1
    return track


# ── Effects ────────────────────────────────────────────────────────────────────

def _reverb(audio: np.ndarray, wet: float = 0.50) -> np.ndarray:
    """Wide atmospheric Schroeder reverb — comb + allpass."""
    combs     = [37.2, 44.5, 49.8, 54.6, 60.2, 66.7]
    allpasses  = [6.2, 2.4, 1.1]
    rev = np.zeros_like(audio)
    for delay in combs:
        d  = int(delay * SAMPLE_RATE / 1000)
        fb = 0.76 * (1 - 0.38 * delay / 68)
        b  = np.zeros(d + 1); b[0] = 1.0
        a  = np.zeros(d + 1); a[0] = 1.0; a[d] = -fb
        rev += signal.lfilter(b, a, audio)
    rev /= len(combs)
    for delay in allpasses:
        d  = int(delay * SAMPLE_RATE / 1000)
        b  = np.zeros(d + 1); b[0] = -0.7; b[d] = 1.0
        a  = np.zeros(d + 1); a[0] =  1.0; a[d] = -0.7
        rev = signal.lfilter(b, a, rev)
    return audio * (1 - wet) + rev * wet


def _stereo_widen(mono: np.ndarray) -> np.ndarray:
    d     = int(24 * SAMPLE_RATE / 1000)   # 24ms Haas delay
    left  = mono
    right = np.concatenate([np.zeros(d), mono[:-d]])
    t     = np.linspace(0, len(mono) / SAMPLE_RATE, len(mono))
    right = right * (1 + 0.0020 * np.sin(2 * np.pi * 0.28 * t))
    return np.stack([left, right], axis=1)


def _lopass(audio: np.ndarray, hz: float) -> np.ndarray:
    b, a = signal.butter(5, hz / (SAMPLE_RATE / 2), btype="low")
    return signal.filtfilt(b, a, audio)


def _saturate(audio: np.ndarray, drive: float = 1.6) -> np.ndarray:
    return np.tanh(audio * drive) / drive


def _compress(audio: np.ndarray, thresh: float = 0.45, ratio: float = 3.0) -> np.ndarray:
    out  = audio.copy()
    mask = np.abs(out) > thresh
    out[mask] = np.sign(out[mask]) * (thresh + (np.abs(out[mask]) - thresh) / ratio)
    return out


def _fade(audio: np.ndarray, secs: float = 5.0) -> np.ndarray:
    n = int(secs * SAMPLE_RATE)
    audio[:n]  *= np.linspace(0, 1, n)
    audio[-n:] *= np.linspace(1, 0, n)
    return audio


def _norm(audio: np.ndarray, target: float = 0.85) -> np.ndarray:
    peak = np.max(np.abs(audio))
    return audio * (target / peak) if peak > 0 else audio


def _apply_tape_stop(audio: np.ndarray, position: int) -> np.ndarray:
    """Tape-stop: progressive low-pass + volume fade, simulating tape slowing."""
    stop_len = int(0.55 * SAMPLE_RATE)
    end      = min(position + stop_len, len(audio))
    n        = end - position
    if n < 500:
        return audio

    n_chunks  = 22
    chunk_len = max(1, n // n_chunks)
    for i in range(n_chunks):
        cs       = position + i * chunk_len
        ce       = min(cs + chunk_len, end)
        progress = i / n_chunks
        cutoff_hz   = max(300, int(5500 * (1 - progress) ** 1.8))
        cutoff_norm = cutoff_hz / (SAMPLE_RATE / 2)
        if cutoff_norm < 0.95:
            b, a = signal.butter(2, cutoff_norm, btype="low")
            audio[cs:ce] = signal.lfilter(b, a, audio[cs:ce])
        audio[cs:ce] *= (1 - progress) ** 0.75

    silence_end = min(end + int(0.10 * SAMPLE_RATE), len(audio))
    audio[end:silence_end] = 0
    return audio


def _layer_fade_in(track: np.ndarray, start: int, fade_bars: int = 2) -> np.ndarray:
    """Silence track before 'start', then fade in over 'fade_bars'."""
    result = track.copy()
    result[:start] = 0
    fade_len = min(fade_bars * BAR_LEN, TOTAL - start)
    if fade_len > 0 and start < TOTAL:
        result[start:start + fade_len] *= np.linspace(0, 1, fade_len)
    return result


# ── Entry point ────────────────────────────────────────────────────────────────

def generate_lofi_track(prompt: str, output_path: Path) -> Path:
    print(f"[2/5] Generating lo-fi music ({VIDEO_DURATION}s)...")

    p = prompt.lower()
    mood_key = "nocturne"
    for key, keywords in MOOD_MAP.items():
        if any(kw in p for kw in keywords):
            mood_key = key
            break

    prog = PROGRESSIONS[mood_key]
    print(f"   Mood: {mood_key} | BPM: {BPM} | Boom-bap + jazz piano + Rhodes + rain")

    # Texture — present throughout (rain fades in with main fade)
    print("   Synthesizing: ambience + texture...")
    rain    = _rain_ambience(TOTAL)
    hiss    = _tape_hiss(TOTAL)
    crackle = _vinyl_crackle(TOTAL)

    # Musical layers
    print("   Synthesizing: bass + drums + piano...")
    bass   = _build_bass_line(prog)
    drums  = _build_boom_bap_drums()
    chords = _build_piano_chords(prog)
    print("   Synthesizing: Rhodes + melody...")
    rhodes = _build_rhodes_layer(prog)
    melody = _build_melody(prog)

    # Gradual arrangement — layers enter progressively
    # At 75 BPM: 1 bar = 3.2s  →  intro 4 bars = 12.8s
    bass   = _layer_fade_in(bass,   4  * BAR_LEN, fade_bars=3)
    drums  = _layer_fade_in(drums,  4  * BAR_LEN, fade_bars=4)
    chords = _layer_fade_in(chords, 7  * BAR_LEN, fade_bars=2)
    rhodes = _layer_fade_in(rhodes, 10 * BAR_LEN, fade_bars=2)
    melody = _layer_fade_in(melody, 11 * BAR_LEN, fade_bars=3)

    # Mix
    mix = (chords  * 0.62 +
           melody  * 0.50 +
           rhodes  * 0.46 +
           bass    * 0.78 +
           drums   * 0.60 +
           rain    +
           hiss    +
           crackle)

    # Tape-stop effect at ~62% through the track
    if VIDEO_DURATION >= 60:
        stop_pos = int(0.62 * TOTAL)
        stop_pos = (stop_pos // BAR_LEN) * BAR_LEN + int(0.25 * BAR_LEN)
        mix = _apply_tape_stop(mix, stop_pos)

    # Effects chain — warm analog processing
    print("   Effects: lo-fi filter, warmth, reverb, stereo...")
    mix = _lopass(mix, 6800)
    mix = _saturate(mix, drive=1.6)
    mix = _reverb(mix, wet=0.50)
    mix = _compress(mix, thresh=0.45)
    mix = _fade(mix, secs=5.0)
    mix = _norm(mix, target=0.85)

    stereo = _stereo_widen(mix)
    pcm    = (stereo * 32767).astype(np.int16)

    wav_path = Path(output_path).with_suffix(".wav")
    mp3_path = Path(output_path).with_suffix(".mp3")

    with wave.open(str(wav_path), "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())

    subprocess.run([
        str(FFMPEG), "-y", "-i", str(wav_path),
        "-codec:a", "libmp3lame", "-b:a", "320k",
        str(mp3_path),
    ], check=True, capture_output=True)

    wav_path.unlink(missing_ok=True)
    print(f"   Saved: {mp3_path.name} ({mp3_path.stat().st_size // 1024} KB)")
    return mp3_path
