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
    text_weight: np.ndarray | None = None
    feature_weight: np.ndarray | None = None
    feature_bias: np.ndarray | None = None
    activation: str = "identity"

    @classmethod
    def fit(cls, audio, text, method="ridge", alpha=1.0, components=64):
        x, y = normalize(audio).astype(np.float64), normalize(text).astype(np.float64)
        if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or len(x) < 2:
            raise ValueError("Need at least two paired embedding rows")
        mx, my = x.mean(axis=0), y.mean(axis=0)
        x, y = x - mx, y - my
        text_weight = feature_weight = feature_bias = None
        activation = "identity"
        if method == "cca":
            if alpha <= 0:
                raise ValueError("CCA regularization must be positive")
            n = len(x)

            def whitening(z):
                covariance = z.T @ z / n
                scale = max(float(np.trace(covariance) / len(covariance)), 1e-12)
                values, vectors = np.linalg.eigh(covariance)
                return (vectors * (1 / np.sqrt(np.maximum(values, 0) + alpha * scale))) @ vectors.T

            wx, wy = whitening(x), whitening(y)
            u, singular, vt = np.linalg.svd(wx @ (x.T @ y / n) @ wy, full_matrices=False)
            k = min(components, len(x) - 1, len(singular))
            if k < 1:
                raise ValueError("CCA requires at least one component")
            scale = np.sqrt(np.maximum(singular[:k], 0))
            w, text_weight = (wx @ u[:, :k]) * scale, (wy @ vt[:k].T) * scale
        elif method in {"ridge", "rff"}:
            if alpha <= 0:
                raise ValueError("Ridge alpha must be positive")
            if method == "rff":
                # Seeded random Fourier features approximate a nonlinear Gaussian kernel.
                rng = np.random.default_rng(2026)
                feature_weight = rng.normal(size=(x.shape[1], 256))
                feature_bias = rng.uniform(0, 2 * np.pi, 256)
                activation = "cos"
                x = np.sqrt(2 / 256) * np.cos(x @ feature_weight + feature_bias)
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
        f32 = lambda value: None if value is None else value.astype(np.float32)
        return cls(method, f32(w), f32(mx), f32(my), alpha, f32(text_weight),
                   f32(feature_weight), f32(feature_bias), activation)

    def transform(self, audio):
        # Both modalities are searched in the centered text space. Adding the text mean
        # only to the audio side would create an inconsistent cosine retrieval space.
        x = normalize(audio) - self.audio_mean
        if self.feature_weight is not None:
            x = x @ self.feature_weight + self.feature_bias
            x = np.sqrt(2 / x.shape[1]) * np.cos(x) if self.activation == "cos" else np.tanh(x)
        return normalize(x @ self.weight)

    def text_transform(self, text):
        text = normalize(text) - self.text_mean
        return normalize(text if self.text_weight is None else text @ self.text_weight)

    def save(self, path: Path):
        optional = {key: value for key in ["text_weight", "feature_weight", "feature_bias"]
                    if (value := getattr(self, key)) is not None}
        save_npz(path, method=np.array(self.method), weight=self.weight,
                 audio_mean=self.audio_mean, text_mean=self.text_mean, alpha=np.array(self.alpha),
                 activation=np.array(self.activation), **optional)

    @classmethod
    def load(cls, path: Path):
        with np.load(path, allow_pickle=False) as z:
            return cls(str(z["method"]), z["weight"], z["audio_mean"], z["text_mean"], float(z["alpha"]),
                       *(z.get(key) for key in ["text_weight", "feature_weight", "feature_bias"]),
                       str(z["activation"]) if "activation" in z else "identity")

    @classmethod
    def fit_contrastive(cls, audio, text, caption_indices, epochs=40, seed=2026, progress=lambda msg: None):
        """A small tanh audio projection, fitted with every caption as a positive.

        In-batch other recordings are negatives. No encoder parameters are updated.
        """
        import torch
        from torch.nn import functional as F

        torch.manual_seed(seed)
        x, y = normalize(audio), normalize(text)
        mx, my = x.mean(0), y.mean(0)
        x, y = torch.from_numpy(x - mx), torch.from_numpy(y - my)
        labels = torch.as_tensor(caption_indices, dtype=torch.long)
        if len(x) < 2 or len(y) != len(labels) or set(labels.tolist()) != set(range(len(x))):
            raise ValueError("Contrastive fitting needs captions for every training recording")
        model = torch.nn.Sequential(torch.nn.Linear(x.shape[1], 128), torch.nn.Tanh(),
                                    torch.nn.Linear(128, y.shape[1], bias=False))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        generator = torch.Generator().manual_seed(seed)
        for epoch in range(epochs):
            progress(f"Contrastive projection epoch {epoch + 1}/{epochs}")
            order = torch.randperm(len(x), generator=generator)
            for offset in range(0, len(x), 128):
                indices = order[offset:offset + 128]
                if len(indices) < 2:
                    continue
                mask = torch.isin(labels, indices)
                positive = indices[:, None] == labels[mask][None, :]
                logits = F.normalize(model(x[indices]), dim=1) @ F.normalize(y[mask], dim=1).T / .07
                forward = torch.logsumexp(logits, 1) - torch.logsumexp(logits.masked_fill(~positive, -torch.inf), 1)
                backward = F.cross_entropy(logits.T, positive.float().argmax(0))
                loss = forward.mean() / 2 + backward / 2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return cls("contrastive", model[2].weight.detach().numpy().T.copy(), mx, my, .01,
                   feature_weight=model[0].weight.detach().numpy().T.copy(),
                   feature_bias=model[0].bias.detach().numpy().copy(), activation="tanh")
