"""Download/load CLAP and check actual inference; not a semantic accuracy benchmark."""
import json
import time
from pathlib import Path

import numpy as np

from sonic_bridge.encoders import ClapEncoder
from sonic_bridge.util import write_json

started = time.perf_counter()
encoder = ClapEncoder()
t = np.arange(encoder.sample_rate * 2) / encoder.sample_rate
audio = encoder.encode_audio([(.2 * np.sin(2 * np.pi * f * t)).astype(np.float32) for f in [180, 2800]])
text = encoder.encode_text(["A low pitched steady tone", "A high pitched steady tone"])
assert audio.shape == text.shape == (2, 512)
assert np.isfinite(audio).all() and np.isfinite(text).all()
result = {"encoder": encoder.identity, "audio_shape": list(audio.shape), "text_shape": list(text.shape),
          "cosine_scores": (audio @ text.T).tolist(), "seconds": round(time.perf_counter() - started, 2),
          "scope": "Real pretrained CLAP forward passes on generated tones. Not a retrieval benchmark."}
write_json(Path("artifacts/validation/clap-smoke.json"), result)
print(json.dumps(result, indent=2))
