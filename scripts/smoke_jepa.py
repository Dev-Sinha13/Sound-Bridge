"""Check real Audio-JEPA and MPNet inference using the reviewed pinned config."""
import json
import time
from pathlib import Path

import numpy as np

from sonic_bridge.encoders import make_encoder
from sonic_bridge.util import write_json

started = time.perf_counter()
encoder = make_encoder("jepa", Path("configs/jepa.json"))
t = np.arange(encoder.sample_rate * 2) / encoder.sample_rate
audio = encoder.encode_audio([(.2 * np.sin(2 * np.pi * f * t)).astype(np.float32) for f in [180, 2800]])
text = encoder.encode_text(["A low pitched steady tone", "A high pitched steady tone"])
assert audio.shape == text.shape == (2, 768)
assert np.isfinite(audio).all() and np.isfinite(text).all()
result = {"encoder": encoder.identity, "audio_shape": list(audio.shape), "text_shape": list(text.shape),
          "seconds": round(time.perf_counter() - started, 2),
          "scope": "Real frozen Audio-JEPA and MPNet forward passes. Spaces are not yet aligned; no quality claim."}
write_json(Path("artifacts/validation/jepa-smoke.json"), result)
print(json.dumps(result, indent=2))
