"""Frozen text-prompt classification on ESC-50; no fitting on its test examples."""
from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

import numpy as np

from .audio import windows
from .experiments import bootstrap_recall
from .util import digest, file_digest, normalize, save_npz, write_json

PROMPTS = ["The sound of {label}.", "A recording of {label}.", "You can hear {label}."]


def cached_audio_feature(path, base, cache_dir):
    path, cache_dir = Path(path), Path(cache_dir)
    key = digest({"sha": file_digest(path), "encoder": base.identity, "policy": "mean-10s-hop5-v1"})
    cache = cache_dir / f"{key}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            return data["audio"]
    audio = normalize(np.mean([base.encode_audio([w.samples])[0]
                               for w in windows(path, base.sample_rate)], axis=0))
    save_npz(cache, audio=audio)
    return audio


def evaluate_esc50(folder, encoder, output, fold=5, cache_dir=None, calibration_manifest=None,
                   alignment_metadata=None, progress=lambda msg: None):
    if not 1 <= fold <= 5:
        raise ValueError("ESC-50 fold must be between 1 and 5")
    folder, output = Path(folder), Path(output)
    started = time.perf_counter()
    with (folder / "meta" / "esc50.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    categories = {int(row["target"]): row["category"].replace("_", " ") for row in rows}
    classes = sorted(categories)
    candidates = [row for row in rows if int(row["fold"]) == fold]
    excluded = set()
    if calibration_manifest and alignment_metadata:
        metadata = json.loads(Path(alignment_metadata).read_text(encoding="utf-8"))
        training = set(metadata["training_ids"])
        for line in Path(calibration_manifest).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["id"] in training:
                sound_id = row.get("metadata", {}).get("sound_id")
                if sound_id:
                    excluded.add(str(sound_id))
                match = re.search(r"/sounds/(\d+)", row.get("source", ""))
                if match:
                    excluded.add(match.group(1))
    before = len(candidates)
    candidates = [row for row in candidates if row["src_file"] not in excluded]
    if not candidates:
        raise ValueError("No evaluation rows remain")
    prompts = [template.format(label=categories[c]) for c in classes for template in PROMPTS]
    text = []
    for start in range(0, len(prompts), 32):
        text.extend(encoder.encode_text(prompts[start:start + 32]))
    prototypes = normalize(np.asarray(text).reshape(len(classes), len(PROMPTS), -1).mean(axis=1))
    base = getattr(encoder, "base", encoder)
    cache_dir = Path(cache_dir or output.parent / "esc50-cache")
    features, targets, groups = [], [], []
    for i, row in enumerate(candidates):
        if i % 20 == 0:
            progress(f"ESC-50 {i + 1}/{len(candidates)}")
        path = folder / "audio" / row["filename"]
        audio = cached_audio_feature(path, base, cache_dir)
        if hasattr(encoder, "alignment"):
            audio = encoder.alignment.transform(audio[None])[0]
        features.append(audio)
        targets.append(int(row["target"]))
        groups.append(row["src_file"])
    scores = normalize(np.stack(features)) @ prototypes.T
    predicted = np.asarray(classes)[np.argmax(scores, axis=1)]
    targets = np.asarray(targets)
    correct = predicted == targets
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    for target, prediction in zip(targets, predicted):
        confusion[classes.index(int(target)), classes.index(int(prediction))] += 1
    per_class = {categories[c]: float(np.mean(correct[targets == c])) for c in classes if np.any(targets == c)}
    report = {"dataset": "ESC-50", "fold": fold, "examples": len(candidates), "classes": categories,
              "encoder": encoder.identity, "accuracy": float(correct.mean()),
              "macro_accuracy": float(np.mean(list(per_class.values()))), "per_class_accuracy": per_class,
              "accuracy_ci95": bootstrap_recall(np.where(correct, 1, 2), groups)["recall@1"],
              "confusion_matrix": confusion.tolist(), "prompt_templates": PROMPTS,
              "excluded_calibration_source_matches": before - len(candidates),
              "seconds": round(time.perf_counter() - started, 3),
              "protocol": "One fixed fold, fixed prompt ensemble; no fitting or prompt selection on ESC-50. "
                          "Source-ID overlap with supplied calibration training manifest is excluded. "
                          "Upstream encoder pretraining overlap remains unknown; not a claim of novel-class generalization."}
    write_json(output, report)
    return report
