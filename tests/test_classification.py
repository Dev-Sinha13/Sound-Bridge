import csv
import json

import numpy as np
import soundfile as sf

from sonic_bridge.classification import evaluate_esc50
from sonic_bridge.encoders import DemoEncoder
from sonic_bridge.util import write_json


def test_fixed_prompt_classification_and_calibration_source_exclusion(tmp_path):
    (tmp_path / "audio").mkdir()
    (tmp_path / "meta").mkdir()
    time = np.arange(16000) / 16000
    rows = []
    for i, (frequency, category) in enumerate([(180, "low_tone"), (2800, "high_tone")]):
        name = f"sound-{i}.wav"
        sf.write(tmp_path / "audio" / name, .2 * np.sin(2 * np.pi * frequency * time), 16000)
        rows.append({"filename": name, "fold": "5", "target": str(i), "category": category,
                     "src_file": str(100 + i)})
    with (tmp_path / "meta" / "esc50.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = evaluate_esc50(tmp_path, DemoEncoder(), tmp_path / "report.json")
    assert report["accuracy"] == 1
    assert report["examples"] == 2
    assert report["accuracy_ci95"] == [1, 1]
    metadata = tmp_path / "alignment.json"
    write_json(metadata, {"training_ids": ["calibration"]})
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"id": "calibration", "metadata": {"sound_id": "100"}}))
    report = evaluate_esc50(tmp_path, DemoEncoder(), tmp_path / "filtered.json",
                            calibration_manifest=manifest, alignment_metadata=metadata)
    assert report["examples"] == 1
    assert report["excluded_calibration_source_matches"] == 1
    assert report["accuracy"] == 1
