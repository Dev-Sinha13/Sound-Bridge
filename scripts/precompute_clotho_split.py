"""Precompute one verified, extracted split while other archives are downloading."""
import argparse
import csv
import json
from pathlib import Path

from sonic_bridge.datasets import clotho_exclusions, select_clotho_rows, source_group
from sonic_bridge.encoders import make_encoder
from sonic_bridge.experiments import extract
from sonic_bridge.util import file_digest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", type=Path, default=Path("data/clotho"))
parser.add_argument("--split", choices=["development", "validation", "evaluation"], default="validation")
parser.add_argument("--count", type=int, default=200)
parser.add_argument("--seed", type=int, default=2026)
args = parser.parse_args()
if not (args.dataset / f".{args.split}.extracted").exists():
    raise ValueError("Wait for the verified split extraction to complete")
with (args.dataset / "downloads" / f"clotho_captions_{args.split}.csv").open(
        encoding="utf-8-sig", newline="") as f:
    captions = sorted(csv.DictReader(f), key=lambda row: row["file_name"])
with (args.dataset / "downloads" / f"clotho_metadata_{args.split}.csv").open(encoding="latin-1", newline="") as f:
    metadata = {row["file_name"]: row for row in csv.DictReader(f)}
excluded = clotho_exclusions(args.dataset)[args.split]
captions = select_clotho_rows([row for row in captions if source_group(metadata[row["file_name"]]) not in excluded],
                              metadata, args.count, args.seed)
rows = []
for caption in captions:
    audio = (args.dataset / args.split / caption["file_name"]).resolve()
    rows.append({"id": f"clotho:{args.split}:{audio.name}", "audio": str(audio), "sha": file_digest(audio),
                 "captions": [caption[f"caption_{i}"] for i in range(1, 6)]})
for backend in ["clap", "jepa"]:
    encoder = make_encoder(backend, Path("configs/jepa.json") if backend == "jepa" else None, offline=True)
    encoder.torch.set_num_threads(4)
    extract(rows, encoder, args.dataset / "embedding-cache" / backend, progress=lambda msg: print(msg, flush=True))
    print(json.dumps({"backend": backend, "cached": len(rows), "split": args.split}), flush=True)
    del encoder
