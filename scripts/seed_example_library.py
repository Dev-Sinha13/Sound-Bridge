"""Create a small attributed local demo library from the downloaded evaluation split."""
import argparse
import csv
import shutil
from pathlib import Path

from sonic_bridge.datasets import select_clotho_rows
from sonic_bridge.util import write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", type=Path, default=Path("data/clotho"))
parser.add_argument("--output", type=Path, default=Path("data/example_library"))
parser.add_argument("--count", type=int, default=16)
args = parser.parse_args()
if not (args.dataset / ".evaluation.extracted").exists():
    raise ValueError("The verified Clotho evaluation split must be extracted first")
with (args.dataset / "downloads" / "clotho_metadata_evaluation.csv").open(encoding="latin-1", newline="") as f:
    metadata = {row["file_name"]: row for row in csv.DictReader(f)}
with (args.dataset / "downloads" / "clotho_captions_evaluation.csv").open(encoding="utf-8-sig", newline="") as f:
    rows = select_clotho_rows(sorted(csv.DictReader(f), key=lambda r: r["file_name"]), metadata, args.count)
args.output.mkdir(parents=True, exist_ok=True)
credits = []
for row in rows:
    name = row["file_name"]
    shutil.copyfile(args.dataset / "evaluation" / name, args.output / name)
    credits.append({"file": name, "metadata": metadata[name], "captions": [row[f"caption_{i}"] for i in range(1, 6)]})
write_json(args.output / "credits.json", {"source": "https://zenodo.org/records/4783391", "recordings": credits})
print(args.output.resolve())
