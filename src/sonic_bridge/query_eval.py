"""Explicit query judgments, validation-only cutoffs, and inspectable failure records."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .index import index_info, search
from .util import digest, file_digest, read_json, write_json


def load_queries(path):
    suite = read_json(Path(path))
    rows = suite["queries"]
    seen, groups, sources = set(), {}, {}
    if not rows:
        raise ValueError("Query suite is empty")
    for row in rows:
        key = row["query"].strip().casefold()
        if not key or key in seen or row.get("split") not in {"val", "test"}:
            raise ValueError("Queries must be unique, nonempty, and split into val/test")
        seen.add(key)
        if not row.get("group") or not isinstance(row.get("relevant"), list):
            raise ValueError("Each query requires a source group and explicit relevant list (empty for absent sounds)")
        for value, table in [(row["group"], groups), *[(r["file"], sources) for r in row["relevant"]]]:
            if value in table and table[value] != row["split"]:
                raise ValueError("Query source leakage across validation/test")
            table[value] = row["split"]
        for item in row["relevant"]:
            name = Path(item["file"])
            if name.is_absolute() or ".." in name.parts:
                raise ValueError("Relevant files must be relative to the collection root")
            if ("start" in item or "end" in item) and not 0 <= item.get("start", -1) < item.get("end", -1):
                raise ValueError("Timestamp judgments require 0 <= start < end")
    if {r["split"] for r in rows} != {"val", "test"}:
        raise ValueError("Both validation and test queries are required")
    return suite


def temporal_iou(hit, relevant):
    overlap = max(0, min(hit["end"], relevant["end"]) - max(hit["start"], relevant["start"]))
    return overlap / (max(hit["end"], relevant["end"]) - min(hit["start"], relevant["start"]))


def choose_cutoff(rows, max_false_accept=.1):
    if not 0 <= max_false_accept <= 1:
        raise ValueError("False acceptance target must be between zero and one")
    validation = [r for r in rows if r["split"] == "val"]
    positives = [r for r in validation if r["relevant"]]
    negatives = [r for r in validation if not r["relevant"]]
    if not positives or not negatives:
        raise ValueError("Validation needs both present and absent sound judgments to choose a cutoff")
    candidates = [-1.0, 1.000001, *[float(np.nextafter(r["top_score"], np.inf)) for r in validation]]
    valid = [t for t in candidates if np.mean([r["top_score"] >= t for r in negatives]) <= max_false_accept]
    # Maximize correctly accepted positives, then minimize false accepts, then use the lower threshold.
    return max(valid, key=lambda t: (sum(r["top_correct"] and r["top_score"] >= t for r in positives),
                                    -sum(r["top_score"] >= t for r in negatives), -t))


def evaluate_queries(suite_path, database, encoder, output, max_false_accept=.1, progress=lambda msg: None):
    suite = load_queries(suite_path)
    info = index_info(database)
    root = Path(info["settings"]["root"])
    records = []
    for number, row in enumerate(suite["queries"]):
        for relevant in row["relevant"]:
            if not (root / relevant["file"]).is_file():
                raise ValueError(f"Judged file is missing: {relevant['file']}")
        progress(f"Query {number + 1}/{len(suite['queries'])}: {row['query']}")
        started = time.perf_counter()
        matches = search(database, encoder, row["query"], limit=100)
        elapsed = time.perf_counter() - started
        for match in matches:
            match["file"] = Path(match["path"]).relative_to(root).as_posix()
            del match["path"]
        relevant_files = {r["file"] for r in row["relevant"]}
        unique = list(dict.fromkeys(m["file"] for m in matches))
        rank = next((i + 1 for i, file in enumerate(unique) if file in relevant_files), None)
        top = matches[0] if matches else None
        intervals = [r for r in row["relevant"] if "start" in r]
        iou = max((temporal_iou(top, r) for r in intervals if top and top["file"] == r["file"]), default=0)
        records.append({**row, "rank": rank, "top_score": top["score"] if top else -1.0,
                        "top_correct": bool(top and top["file"] in relevant_files),
                        "top_temporal_iou": iou if intervals else None, "seconds": elapsed,
                        "matches": matches[:10]})
    cutoff = choose_cutoff(records, max_false_accept)
    for row in records:
        accepted = bool(row["matches"]) and row["top_score"] >= cutoff
        row["accepted"] = accepted
        row["failure"] = ("false_accept" if accepted else None) if not row["relevant"] else (
            "false_reject" if not accepted else "wrong_recording" if not row["top_correct"] else
            "timestamp_mismatch" if row["top_temporal_iou"] is not None and row["top_temporal_iou"] < .5 else None)

    def summarize(rows):
        positive = [r for r in rows if r["relevant"]]
        negative = [r for r in rows if not r["relevant"]]
        mean = lambda values: float(np.mean(values)) if values else None
        return {"queries": len(rows), "present": len(positive), "absent": len(negative),
                **{f"recall@{k}": mean([r["rank"] is not None and r["rank"] <= k for r in positive])
                   for k in [1, 5, 10]},
                "mrr@100_windows": mean([1 / r["rank"] if r["rank"] else 0 for r in positive]),
                "correct_accept_rate": mean([r["accepted"] and r["top_correct"] for r in positive]),
                "false_accept_rate": mean([r["accepted"] for r in negative]),
                "temporal_recall@1_iou50": mean([r["top_temporal_iou"] >= .5 for r in positive
                                                if r["top_temporal_iou"] is not None]),
                "latency_p50_seconds": mean([float(np.median([r["seconds"] for r in rows]))]) if rows else None}
    report = {"suite_sha256": file_digest(Path(suite_path)), "suite_name": suite.get("name"),
              "judgment_status": suite.get("judgment_status", "unspecified"), "encoder": encoder.identity,
              "index": info, "index_settings_sha256": digest(info["settings"]),
              "cutoff": cutoff, "cutoff_selection": "validation only; per model, collection, and window policy",
              "max_validation_false_accept": max_false_accept,
              "splits": {s: summarize([r for r in records if r["split"] == s]) for s in ["val", "test"]},
              "by_category": {c: summarize([r for r in records if r.get("category", "other") == c])
                              for c in sorted({r.get("category", "other") for r in records})},
              "records": records,
              "limits": "Rank is among distinct files in the first 100 temporally suppressed windows. "
                        "Judgments must cover all relevant gallery files. Cutoffs do not transfer between indexes. "
                        "Small authored suites are development checks, not independent user studies."}
    write_json(Path(output), report)
    return report
