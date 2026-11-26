import json
from pathlib import Path

import numpy as np
import soundfile as sf


def create_demo(folder):
    folder = Path(folder).resolve()
    audio_dir = folder / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2026)
    rows = []
    for split, count in [("train", 18), ("val", 6), ("test", 6)]:
        for i in range(count):
            category = i % 3
            label = ["low", "mid", "high"][category]
            frequency = [180, 850, 2800][category] * rng.uniform(0.9, 1.1)
            time = np.arange(32000) / 16000
            wave = 0.35 * np.sin(2 * np.pi * frequency * time + rng.uniform(0, 6.28))
            wave += rng.normal(0, 0.002, len(time))
            name = f"{split}-{i:02d}-{label}.wav"
            sf.write(audio_dir / name, wave, 16000)
            rows.append({"id": name, "audio": f"audio/{name}", "group": name, "split": split, "synthetic": True,
                         "captions": [f"A {label} tone", f"A steady {label} pitched tone"]})
    manifest = folder / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return {"folder": str(audio_dir), "manifest": str(manifest), "synthetic": True,
            "note": "Signal-processing fixture with three tone classes; not evidence of semantic retrieval quality."}
