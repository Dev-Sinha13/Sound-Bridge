from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3"}


@dataclass(frozen=True)
class Window:
    start: float
    end: float
    samples: np.ndarray


def read_audio(path: Path, sample_rate: int, start: float = 0, end: float | None = None) -> np.ndarray:
    with sf.SoundFile(path) as f:
        source_rate = f.samplerate
        first = max(0, round(start * source_rate))
        last = len(f) if end is None else min(len(f), round(end * source_rate))
        if first >= last:
            raise ValueError(f"Empty audio interval: {path.name}")
        f.seek(first)
        samples = f.read(last - first, dtype="float32", always_2d=True).mean(axis=1)
    if not np.isfinite(samples).all():
        raise ValueError(f"Non-finite audio: {path.name}")
    if source_rate != sample_rate:
        factor = gcd(source_rate, sample_rate)
        samples = resample_poly(samples, sample_rate // factor, source_rate // factor).astype(np.float32)
    return samples


def windows(path: Path, sample_rate: int, seconds: float = 10, hop: float = 5):
    if not (0 < hop <= seconds <= 60):
        raise ValueError("Require 0 < hop <= window <= 60 seconds")
    info = sf.info(path)
    duration = info.frames / info.samplerate
    if duration <= 0:
        raise ValueError(f"Empty audio: {path.name}")
    # Stop after the first window that reaches the end: no tiny redundant tail windows.
    start = 0.0
    while True:
        end = min(duration, start + seconds)
        yield Window(start, end, read_audio(path, sample_rate, start, end))
        if end >= duration - 1e-9:
            break
        start += hop


def discover(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise ValueError(f"Audio folder does not exist: {folder}")
    return sorted(p.resolve() for p in folder.rglob("*") if p.suffix.lower() in EXTENSIONS and p.is_file())
