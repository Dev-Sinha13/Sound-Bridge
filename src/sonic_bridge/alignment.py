from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .util import normalize, save_npz


@dataclass
class Alignment:
    method: str
    weight: np.ndarray
    audio_mean: np.ndarray
    text_mean: np.ndarray
    alpha: float = 0.0

    @classmethod
    def fit(cls, audio, text, method="ridge", alpha=1.0):
        x, y = normalize(audio).astype(np.float64), normalize(text).astype(np.float64)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
            raise ValueError("Need at least two paired embedding rows")
        mx, my = x.mean(axis=0), y.mean(axis=0)
        x, y = x - mx, y - my
        if method == "ridge":
            if alpha <= 0:
                raise ValueError("Ridge alpha must be positive")
            # Dual solve keeps memory bounded when calibration rows < feature dimensions.
            if len(x) < x.shape[1]:
                w = x.T @ np.linalg.solve(x @ x.T + alpha * np.eye(len(x)), y)
            else:
                w = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)
        elif method == "procrustes":
            if x.shape[1] != y.shape[1]:
                raise ValueError("Orthogonal Procrustes requires equal dimensions; use ridge for unequal spaces")
            u, _, vt = np.linalg.svd(x.T @ y, full_matrices=False)
            w = u @ vt
        else:
            raise ValueError(f"Unknown alignment method: {method}")
        return cls(method, w.astype(np.float32), mx.astype(np.float32), my.astype(np.float32), alpha)

    def transform(self, audio):
        # Both modalities are searched in the centered text space. Adding the text mean
        # only to the audio side would create an inconsistent cosine retrieval space.
        return normalize((normalize(audio) - self.audio_mean) @ self.weight)

    def text_transform(self, text):
        return normalize(normalize(text) - self.text_mean)

    def save(self, path: Path):
        save_npz(path, method=np.array(self.method), weight=self.weight,
                 audio_mean=self.audio_mean, text_mean=self.text_mean, alpha=np.array(self.alpha))

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as z:
            return cls(str(z["method"]), z["weight"], z["audio_mean"], z["text_mean"], float(z["alpha"]))
