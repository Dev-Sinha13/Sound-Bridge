"""Compare native encoders and window policies on explicit, source-separated query judgments."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from sonic_bridge.encoders import make_encoder
from sonic_bridge.index import build_index
from sonic_bridge.query_eval import evaluate_queries
from sonic_bridge.util import peak_memory_bytes, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--backend", choices=["clap", "finelap", "aligned", "demo"], default="clap")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    encoder = make_encoder(args.backend, args.config, offline=args.offline)
    load_seconds = time.perf_counter() - started
    torch = getattr(encoder, "torch", getattr(getattr(encoder, "base", None), "torch", None))
    if torch:
        torch.set_num_threads(4)
    results = []
    for name, scales in [("coarse", []), ("multi", [2, 5])]:
        database = args.output / f"{name}.sqlite"
        started = time.perf_counter()
        info = build_index(args.folder, database, encoder, scales=scales, progress=lambda s: print(s, flush=True))
        indexing_seconds = time.perf_counter() - started
        report = evaluate_queries(args.suite, database, encoder, args.output / f"{name}.json",
                                  progress=lambda s: print(s, flush=True))
        results.append({"policy": name, "indexing_seconds": indexing_seconds, "index": info,
                        "cutoff": report["cutoff"], "splits": report["splits"],
                        "process_peak_memory_bytes": peak_memory_bytes()})
        write_json(args.output / "comparison.json", {"encoder": encoder.identity, "results": results,
                   "model_load_seconds": load_seconds,
                   "limits": "Exploratory comparison. Index time may include cached batches. "
                             "Unchanged-file indexing measures a refresh, not cold inference. "
                             "Peak memory is process-wide, not per operation. Concurrent jobs may affect timing. "
                             "Do not select models or window policies on the test split."})
    print(args.output / "comparison.json")


if __name__ == "__main__":
    main()
