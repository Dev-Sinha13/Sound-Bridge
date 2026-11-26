"""Measure CPU extraction time on a few real recordings without fitting or ranking models."""
import argparse
import json
import time
from pathlib import Path

import soundfile as sf

from sonic_bridge.audio import windows
from sonic_bridge.encoders import make_encoder
from sonic_bridge.util import write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("folder", type=Path)
parser.add_argument("--count", type=int, default=3)
args = parser.parse_args()
paths = sorted(args.folder.glob("*.wav"))[:args.count]
results = []
for backend in ["clap", "jepa"]:
    encoder = make_encoder(backend, Path("configs/jepa.json") if backend == "jepa" else None, offline=True)
    encoder.torch.set_num_threads(4)
    started = time.perf_counter()
    count = 0
    for path in paths:
        batch = []
        for chunk in windows(path, encoder.sample_rate):
            batch.append(chunk.samples)
            if len(batch) == 4:
                encoder.encode_audio(batch)
                count += len(batch)
                batch = []
        if batch:
            encoder.encode_audio(batch)
            count += len(batch)
    seconds = time.perf_counter() - started
    result = {"backend": backend, "threads": 4, "recordings": len(paths), "windows": count,
              "audio_seconds": sum(sf.info(p).duration for p in paths), "elapsed_seconds": round(seconds, 3)}
    print(json.dumps(result), flush=True)
    results.append(result)
    del encoder
write_json(Path("artifacts/validation/cpu-profile.json"), results)
