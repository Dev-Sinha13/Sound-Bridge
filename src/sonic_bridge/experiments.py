from __future__ import annotations

import csv
import json
import platform
import time
from pathlib import Path

import numpy as np

from .alignment import Alignment
from .audio import windows
from .util import digest, file_digest, normalize, save_npz, write_json


def load_manifest(path):
    path = Path(path).resolve()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids, groups, contents, paths = set(), {}, {}, {}
    for row in rows:
        if not isinstance(row.get("id"), str) or not row["id"] or row["id"] in ids:
            raise ValueError("Every manifest row needs a unique string id")
        ids.add(row["id"])
        if row.get("split") not in {"train", "val", "test"}:
            raise ValueError("Every row needs split: train, val, or test")
        if not isinstance(row.get("captions"), list) or not row["captions"] or any(
                not isinstance(c, str) or not c.strip() for c in row["captions"]):
            raise ValueError("Every row needs a non-empty list of captions")
        audio = (path.parent / row["audio"]).resolve()
        if not audio.is_file():
            raise ValueError(f"Audio missing: {audio}")
        row["audio"] = str(audio)
        row["sha"] = file_digest(audio)
        row["group"] = row.get("group") or row["id"]
        for key, table in [(row["group"], groups), (row["sha"], contents), (str(audio), paths)]:
            if key in table and table[key] != row["split"]:
                raise ValueError("Data leakage: a source group, audio path, or identical file occurs across splits")
            table[key] = row["split"]
    if not rows or {r["split"] for r in rows} != {"train", "val", "test"}:
        raise ValueError("Manifest must contain non-empty train, val, and test splits")
    return rows


def extract(rows, encoder, cache_dir, progress=lambda msg: None, batch_size=4):
    cache_dir = Path(cache_dir)
    audio, targets, all_text, text_ids = [], [], [], []
    pending = []

    def fill_cache():
        if not pending:
            return
        # Text batching avoids running the sentence model once per captioned recording.
        captions = [caption for _, row in pending for caption in row["captions"]]
        text_features = []
        for offset in range(0, len(captions), 32):
            text_features.extend(encoder.encode_text(captions[offset:offset + 32]))
        text_offset = 0
        for cache, row in pending:
            clip_features, batch = [], []
            for window in windows(Path(row["audio"]), encoder.sample_rate):
                batch.append(window.samples)
                if len(batch) == batch_size:
                    clip_features.extend(encoder.encode_audio(batch))
                    batch = []
            if batch:
                clip_features.extend(encoder.encode_audio(batch))
            a = normalize(np.mean(clip_features, axis=0))
            count = len(row["captions"])
            t = normalize(np.asarray(text_features[text_offset:text_offset + count]))
            text_offset += count
            save_npz(cache, audio=a, text=t)
        pending.clear()

    # Keep the extraction policy stable: batching changes execution, not preprocessing.
    keys = []
    if not 1 <= batch_size <= 64:
        raise ValueError("Batch size must be between 1 and 64")
    for i, row in enumerate(rows):
        key = digest({"sha": row["sha"], "captions": row["captions"], "encoder": encoder.identity,
                      "policy": "mean-10s-hop5-v1"})
        cache = cache_dir / f"{key}.npz"
        keys.append(cache)
        if not cache.exists():
            pending.append((cache, row))
            if len(pending) == batch_size:
                progress(f"Embedding {i + 1}/{len(rows)}: {row['id']}")
                fill_cache()
    fill_cache()
    for row, cache in zip(rows, keys):
        with np.load(cache, allow_pickle=False) as z:
            a, t = z["audio"], z["text"]
        audio.append(a)
        targets.append(normalize(t.mean(axis=0)))
        all_text.extend(t)
        text_ids.extend([row["id"]] * len(t))
    return np.stack(audio), np.stack(targets), np.stack(all_text), np.array(text_ids)


def retrieval(audio, text, audio_ids, text_ids, bootstrap=0):
    """All captions are relevant to their clip. Ties receive the conservative rank."""
    audio, text = normalize(audio), normalize(text)
    audio_ids, text_ids = np.asarray(audio_ids), np.asarray(text_ids)
    metrics = {}
    # Block scores to avoid materializing a full N_clips x N_captions matrix.
    for name, queries, candidates, qids, cids in [
        ("text_to_audio", text, audio, text_ids, audio_ids),
        ("audio_to_text", audio, text, audio_ids, text_ids),
    ]:
        ranks = []
        for offset in range(0, len(queries), 128):
            scores = queries[offset:offset + 128] @ candidates.T
            for qid, score in zip(qids[offset:offset + 128], scores):
                relevant = cids == qid
                if not relevant.any():
                    raise ValueError("Query has no relevant candidate")
                best = score[relevant].max()
                ranks.append(1 + int(np.count_nonzero(score[~relevant] >= best)))
        ranks = np.asarray(ranks)
        metrics[name] = {**{f"recall@{k}": float(np.mean(ranks <= k)) for k in [1, 5, 10]},
                         "median_rank": float(np.median(ranks)), "queries": len(ranks),
                         "candidates": len(candidates)}
        if bootstrap:
            metrics[name]["recall_ci95"] = bootstrap_recall(ranks, qids, bootstrap)
    return metrics


def bootstrap_recall(ranks, ids, iterations=1000, seed=2026):
    """Resample recording IDs, retaining all captions together and candidates fixed."""
    _, inverse = np.unique(ids, return_inverse=True)
    counts = np.bincount(inverse)
    successes = np.stack([np.bincount(inverse, weights=np.asarray(ranks) <= k) for k in [1, 5, 10]])
    rng = np.random.default_rng(seed)
    samples = []
    for offset in range(0, iterations, 100):
        indices = rng.integers(0, len(counts), (min(100, iterations - offset), len(counts)))
        samples.append(successes[:, indices].sum(axis=2) / counts[indices].sum(axis=1))
    values = np.concatenate(samples, axis=1)
    return {f"recall@{k}": np.quantile(values[i], [0.025, 0.975]).tolist() for i, k in enumerate([1, 5, 10])}


def run_experiment(manifest, encoder, output, budgets=(100, 500, 1000, 3000), seeds=(7, 23, 42),
                   alphas=(0.01, 0.1, 1.0, 10.0), progress=lambda msg: None,
                   methods=None, cache_dir=None, batch_size=4):
    started = time.perf_counter()
    methods = list(methods) if methods is not None else ["native", "procrustes", "ridge"]
    if not methods or any(m not in {"native", "procrustes", "ridge", "cca", "rff", "contrastive"} for m in methods):
        raise ValueError("Methods must be native, procrustes, ridge, cca, rff, or contrastive")
    if encoder.identity["backend"] == "aligned":
        raise ValueError("Experiments require raw frozen encoders, not a previously fitted map")
    rows = load_manifest(manifest)
    manifest_rows = rows
    if methods == ["native"]:
        # A pretrained native reference has no calibration fit or validation selection.
        # Validate all splits for leakage, but only encode the held-out gallery and its queries.
        rows = [row for row in rows if row["split"] == "test"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    a, y, t, tids = extract(rows, encoder, cache_dir or output / "cache", progress, batch_size)
    extraction_seconds = time.perf_counter() - started
    ids = np.array([r["id"] for r in rows])
    masks = {split: np.array([r["split"] == split for r in rows]) for split in ["train", "val", "test"]}
    tmasks = {split: np.isin(tids, ids[mask]) for split, mask in masks.items()}

    def score(split, bridge=None, bootstrap=0):
        aa, tt = a[masks[split]], t[tmasks[split]]
        if bridge:
            aa, tt = bridge.transform(aa), bridge.text_transform(tt)
        return retrieval(aa, tt, ids[masks[split]], tids[tmasks[split]], bootstrap)

    records = []
    if "native" in methods and encoder.identity["backend"] in {"clap", "finelap", "demo"}:
        records.append({"method": "native", "budget": 0, "seed": None, "test": score("test", bootstrap=1000)})
    bridge_methods = [m for m in methods if m != "native"]
    if not records and not bridge_methods:
        raise ValueError("Native retrieval is only defined for an already aligned CLAP/demo encoder")
    train = np.flatnonzero(masks["train"])
    groups = sorted({rows[i]["group"] for i in train})
    requested = sorted({int(b) for b in budgets})
    if any(b < 2 for b in requested):
        raise ValueError("Pair budgets must be at least 2 source groups")
    usable = [b for b in requested if b <= len(groups)]
    if bridge_methods and not usable:
        raise ValueError(f"No budget fits {len(groups)} training groups; choose smaller --budgets")
    for seed in (seeds if bridge_methods else []):
        order = np.random.default_rng(seed).permutation(groups)
        for budget in usable:
            selected = set(order[:budget])
            ix = np.array([i for i in train if rows[i]["group"] in selected])
            for method in bridge_methods:
                if method == "procrustes" and a.shape[1] != y.shape[1]:
                    continue
                candidates = []
                for alpha in (alphas if method in {"ridge", "cca", "rff"} else [0.0]):
                    if method == "contrastive":
                        caption_mask = np.isin(tids, ids[ix])
                        indices = {identifier: j for j, identifier in enumerate(ids[ix])}
                        bridge = Alignment.fit_contrastive(a[ix], t[caption_mask],
                                                           [indices[i] for i in tids[caption_mask]],
                                                           seed=int(seed), progress=progress)
                    else:
                        bridge = Alignment.fit(a[ix], y[ix], method, alpha)
                    validation = score("val", bridge)
                    candidates.append((validation["text_to_audio"]["recall@1"], bridge, validation))
                _, bridge, validation = max(candidates, key=lambda c: c[0])
                name = f"{method}-n{budget}-seed{seed}"
                bridge.save(output / f"{name}.npz")
                write_json(output / f"{name}.json", {"encoder": encoder.identity, "method": method,
                           "groups": budget, "rows": len(ix), "seed": seed, "alpha": bridge.alpha,
                           "training_ids": ids[ix].tolist(), "validation": validation})
                record = {"method": method, "budget": budget, "rows": len(ix), "seed": seed,
                          "alpha": bridge.alpha, "artifact": f"{name}.npz", "validation": validation,
                          "test": score("test", bridge, bootstrap=1000)}
                records.append(record)
                progress(f"Finished {name}")
    selected = max((r for r in records if "validation" in r),
                   key=lambda r: r["validation"]["text_to_audio"]["recall@1"], default=None)
    report = {"schema": 2, "encoder": encoder.identity, "manifest_sha": file_digest(Path(manifest)),
              "synthetic": encoder.identity["backend"] == "demo" or any(r.get("synthetic", False) for r in manifest_rows),
              "requested_budgets": requested,
              "skipped_budgets": [b for b in requested if b not in usable],
              "splits": {s: sum(r["split"] == s for r in manifest_rows) for s in ["train", "val", "test"]},
              "encoded_recordings": len(rows), "results": records,
              "selected_artifact": selected["artifact"] if selected else None,
              "runtime": {"extraction_seconds": round(extraction_seconds, 3),
                          "total_seconds": round(time.perf_counter() - started, 3),
                          "python": platform.python_version(), "numpy": np.__version__, "batch_size": batch_size},
              "confidence_intervals": "95% percentile bootstrap over query recording IDs, 1000 draws; "
                                      "all captions stay together; candidate gallery fixed. Not seed uncertainty.",
              "protocol": "Source-disjoint splits; alpha chosen on validation T2A R@1; test only reported. "
                          "Budget counts source groups; captions averaged per clip for linear/RFF fitting; "
                          "contrastive fitting uses all captions as positives and other recordings as negatives. "
                          "No model selected using test scores. Pretraining overlap is not automatically known."}
    write_json(output / "report.json", report)
    if selected:
        import shutil
        artifact = output / selected["artifact"]
        shutil.copyfile(artifact, output / "selected.npz")
        shutil.copyfile(artifact.with_suffix(".json"), output / "selected.json")
    return report


def import_clotho(captions, audio_dir, split, destination):
    """Append an official Clotho caption CSV as explicit manifest rows."""
    if split not in {"train", "val", "test"}:
        raise ValueError("Invalid split")
    destination, audio_dir = Path(destination), Path(audio_dir).resolve()
    rows = []
    with Path(captions).open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            audio = (audio_dir / row["file_name"]).resolve()
            if not audio.is_relative_to(audio_dir) or not audio.is_file():
                raise ValueError(f"Missing or invalid Clotho audio path: {row['file_name']}")
            rows.append({"id": f"clotho:{audio.name}", "audio": str(audio), "group": audio.name,
                         "split": split, "captions": [row[f"caption_{i}"] for i in range(1, 6)]})
    existing = destination.read_text(encoding="utf-8") if destination.exists() else ""
    old_ids = {json.loads(line)["id"] for line in existing.splitlines() if line.strip()}
    if any(r["id"] in old_ids for r in rows):
        raise ValueError("Import contains IDs already in manifest")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(existing.rstrip() + ("\n" if existing.strip() else "") +
                           "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return {"imported": len(rows), "split": split}
