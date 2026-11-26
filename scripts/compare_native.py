"""Paired source-query bootstrap for two completed native runs with cached features."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sonic_bridge.experiments import extract, load_manifest
from sonic_bridge.util import file_digest, normalize, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("left_report", type=Path)
    parser.add_argument("right_report", type=Path)
    parser.add_argument("--left-cache", type=Path, required=True)
    parser.add_argument("--right-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [r for r in load_manifest(args.manifest) if r["split"] == "test"]
    ids = np.array([r["id"] for r in rows])
    hits = []

    def missing(*args):
        raise RuntimeError("Features missing: complete both native experiments first")

    for report_path, cache in [(args.left_report, args.left_cache), (args.right_report, args.right_cache)]:
        report = read_json(report_path)
        if report["manifest_sha"] != file_digest(args.manifest):
            raise ValueError("Both reports must use the provided manifest")
        reference = next(r for r in report["results"] if r["method"] == "native")
        encoder = SimpleNamespace(identity=report["encoder"], encode_audio=missing, encode_text=missing, sample_rate=16000)
        audio, _, text, text_ids = extract(rows, encoder, cache)
        success = []
        for offset in range(0, len(text), 128):
            scores = normalize(text[offset:offset + 128]) @ normalize(audio).T
            relevant = text_ids[offset:offset + 128, None] == ids[None, :]
            best = np.max(np.where(relevant, scores, -np.inf), axis=1)
            success.extend(np.sum((scores >= best[:, None]) & ~relevant, axis=1) == 0)
        success = np.asarray(success)
        expected = reference["test"]["text_to_audio"]["recall@1"]
        if not np.isclose(success.mean(), expected):
            raise ValueError("Cached feature scores disagree with the report")
        hits.append(np.array([success[text_ids == identifier].sum() for identifier in ids]))
    counts = np.array([(text_ids == identifier).sum() for identifier in ids])
    difference = hits[1] - hits[0]
    rng = np.random.default_rng(2026)
    draws = []
    for _ in range(100):
        indices = rng.integers(0, len(rows), size=(100, len(rows)))
        draws.append(difference[indices].sum(axis=1) / counts[indices].sum(axis=1))
    result = {"left": str(args.left_report), "right": str(args.right_report),
              "difference_right_minus_left": float(difference.sum() / counts.sum()),
              "ci95": np.quantile(np.concatenate(draws), [.025, .975]).tolist(),
              "draws": 10000, "seed": 2026, "recordings": len(rows),
              "scope": "Paired query-recording bootstrap with a fixed candidate gallery; exploratory, "
                       "not a correction for multiple comparisons or upstream pretraining overlap."}
    write_json(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
