from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

from .alignment import Alignment
from .util import digest, file_digest, normalize, read_json

# Keep downloaded weights and custom code inside the project by default.
ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))


class DemoEncoder:
    """Deliberately tiny synthetic tone fixture, NOT a learned semantic audio model."""

    sample_rate = 16000
    def __init__(self):
        self.identity = {"backend": "demo", "version": 1}

    def encode_audio(self, waveforms):
        rows = []
        for samples in waveforms:
            power = np.abs(np.fft.rfft(samples)) ** 2
            freq = np.fft.rfftfreq(len(samples), 1 / self.sample_rate)
            rows.append([power[(freq >= lo) & (freq < hi)].sum()
                         for lo, hi in [(0, 450), (450, 1500), (1500, 8001)]])
        return normalize(np.asarray(rows))

    def encode_text(self, texts):
        rows = []
        for text in texts:
            words = set(re.findall(r"[a-z]+", text.lower()))
            row = [float(bool(words & group)) for group in
                   [{"low", "bass", "deep"}, {"mid", "middle", "medium"}, {"high", "treble", "shrill"}]]
            if not any(row):
                raise ValueError("Synthetic demo only supports low, mid, or high tone queries. Use CLAP for real sounds.")
            rows.append(row)
        return normalize(np.asarray(rows))


def _runtime():
    try:
        import torch
        import transformers
    except ImportError as e:
        raise RuntimeError('Install model dependencies: python -m pip install -e ".[models]"') from e
    return torch, transformers


class ClapEncoder:
    sample_rate = 48000

    def __init__(self, device="cpu", offline=False, model="laion/clap-htsat-unfused", revision="main"):
        torch, tr = _runtime()
        self.torch = torch
        self.device = device
        self.model = tr.ClapModel.from_pretrained(model, revision=revision, local_files_only=offline).to(device).eval()
        self.processor = tr.ClapProcessor.from_pretrained(model, revision=revision, local_files_only=offline)
        self.identity = {"backend": "clap", "model": model,
                         "revision": getattr(self.model.config, "_commit_hash", None) or revision}

    def encode_audio(self, waveforms):
        inputs = self.processor(audio=[np.asarray(w) for w in waveforms], sampling_rate=self.sample_rate,
                                return_tensors="pt", padding=True)
        with self.torch.inference_mode():
            result = self.model.get_audio_features(**{k: v.to(self.device) for k, v in inputs.items()})
        return normalize(result.cpu().numpy())

    def encode_text(self, texts):
        inputs = self.processor(text=list(texts), return_tensors="pt", padding=True, truncation=True)
        with self.torch.inference_mode():
            result = self.model.get_text_features(**{k: v.to(self.device) for k, v in inputs.items()})
        return normalize(result.cpu().numpy())


class JepaTextEncoder:
    """Frozen Audio-JEPA plus a frozen sentence encoder. Text is not a text JEPA."""

    def __init__(self, config, device="cpu", offline=False):
        if not config.get("trust_remote_code") or not re.fullmatch(r"[a-f0-9]{40}", config.get("audio_revision", "")):
            raise ValueError("Audio-JEPA requires trust_remote_code: true and a reviewed 40-character audio_revision")
        torch, tr = _runtime()
        self.torch, self.device = torch, device
        audio_id = config.get("audio_model", "ltuncay/Audio-JEPA-base")
        text_id = config.get("text_model", "sentence-transformers/all-mpnet-base-v2")
        common = {"revision": config["audio_revision"], "trust_remote_code": True, "local_files_only": offline}
        self.audio_processor = tr.AutoFeatureExtractor.from_pretrained(audio_id, **common)
        self.audio_model = tr.AutoModel.from_pretrained(audio_id, **common).to(device).eval()
        text_revision = config.get("text_revision", "main")
        self.tokenizer = tr.AutoTokenizer.from_pretrained(text_id, revision=text_revision, local_files_only=offline)
        self.text_model = tr.AutoModel.from_pretrained(text_id, revision=text_revision,
                                                      local_files_only=offline).to(device).eval()
        self.sample_rate = self.audio_processor.sampling_rate
        self.identity = {"backend": "jepa", "audio_model": audio_id, "audio_revision": config["audio_revision"],
                         "text_model": text_id,
                         "text_revision": getattr(self.text_model.config, "_commit_hash", None) or text_revision,
                         "text_pooling": "masked-mean-384", "sample_rate": self.sample_rate}

    def encode_audio(self, waveforms):
        inputs = self.audio_processor(list(waveforms), sampling_rate=self.sample_rate,
                                      return_tensors="pt", padding=True)
        with self.torch.inference_mode():
            result = self.audio_model(**{k: v.to(self.device) for k, v in inputs.items()}).pooler_output
        return normalize(result.cpu().numpy())

    def encode_text(self, texts):
        inputs = self.tokenizer(list(texts), padding=True, truncation=True, max_length=384, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.inference_mode():
            hidden = self.text_model(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            result = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        return normalize(result.cpu().numpy())


class AlignedEncoder:
    def __init__(self, base, alignment, artifact_hash, synthetic_calibration=False):
        self.base, self.alignment = base, alignment
        self.sample_rate = base.sample_rate
        self.identity = {"backend": "aligned", "base": base.identity, "artifact": artifact_hash}
        self.synthetic_calibration = synthetic_calibration

    def encode_audio(self, waveforms):
        return self.alignment.transform(self.base.encode_audio(waveforms))

    def encode_text(self, texts):
        return self.alignment.text_transform(self.base.encode_text(texts))


def make_encoder(backend, config_path=None, device="cpu", offline=False):
    config = read_json(Path(config_path)) if config_path else {}
    if backend == "demo":
        return DemoEncoder()
    if backend == "clap":
        return ClapEncoder(device, offline, config.get("model", "laion/clap-htsat-unfused"), config.get("revision", "main"))
    if backend in {"jepa", "aligned"}:
        if not config_path:
            raise ValueError("JEPA needs --config configs/jepa.json (see example)")
        base = JepaTextEncoder(config, device, offline)
        if backend == "jepa":
            return base
        artifact = Path(config["alignment"])
        if not artifact.is_absolute():
            artifact = Path(config_path).resolve().parent / artifact
        if config.get("alignment_sha256") and file_digest(artifact) != config["alignment_sha256"]:
            raise ValueError("Alignment artifact failed its SHA-256 integrity check")
        metadata = read_json(artifact.with_suffix(".json"))
        if metadata["encoder"] != base.identity:
            raise ValueError("Alignment was trained with different encoder revisions")
        provenance_path = artifact.parent / "provenance.json"
        synthetic = bool(config.get("synthetic_calibration") or
                         (read_json(provenance_path).get("synthetic") if provenance_path.exists() else False))
        return AlignedEncoder(base, Alignment.load(artifact), file_digest(artifact), synthetic)
    raise ValueError(f"Unknown backend: {backend}")


def fingerprint(encoder):
    return digest(encoder.identity)
