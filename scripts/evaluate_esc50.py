import argparse
import json
from pathlib import Path

from sonic_bridge.classification import evaluate_esc50
from sonic_bridge.encoders import make_encoder
from sonic_bridge.util import read_json

parser = argparse.ArgumentParser(description="Fixed-prompt ESC-50 evaluation without test-domain fitting")
parser.add_argument("--dataset", type=Path, default=Path("data/esc50"))
parser.add_argument("--backend", choices=["clap", "aligned"], required=True)
parser.add_argument("--config", type=Path)
parser.add_argument("--calibration-manifest", type=Path)
parser.add_argument("--alignment-metadata", type=Path, help="Use the same calibration-source exclusions for both backends")
parser.add_argument("--fold", type=int, default=5)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
folder = Path(read_json(args.dataset / "provenance.json")["folder"])
encoder = make_encoder(args.backend, args.config, offline=True)
getattr(encoder, "base", encoder).torch.set_num_threads(4)
metadata = args.alignment_metadata
if args.config:
    config = read_json(args.config)
    metadata = (args.config.resolve().parent / config["alignment"]).with_suffix(".json")
report = evaluate_esc50(folder, encoder, args.output, args.fold, cache_dir=args.dataset / "embedding-cache",
                        calibration_manifest=args.calibration_manifest, alignment_metadata=metadata,
                        progress=lambda msg: print(msg, flush=True))
print(json.dumps({k: report[k] for k in ["accuracy", "accuracy_ci95", "examples", "seconds"]}))
