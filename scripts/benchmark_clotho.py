"""Compare CLAP and frozen Audio-JEPA bridges on official Clotho splits."""
import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

from sonic_bridge.bundles import export_alignment
from sonic_bridge.datasets import prepare_clotho
from sonic_bridge.encoders import make_encoder
from sonic_bridge.experiments import run_experiment
from sonic_bridge.util import write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", type=Path, default=Path("data/clotho"))
parser.add_argument("--output", type=Path, default=Path("artifacts/clotho"))
parser.add_argument("--config", type=Path, default=Path("configs/jepa.json"))
parser.add_argument("--budgets", type=int, nargs="+", default=[100, 500, 1000, 3000])
parser.add_argument("--seeds", type=int, nargs="+", default=[7, 23, 42])
parser.add_argument("--limits", type=int, nargs=3, metavar=("TRAIN", "VAL", "TEST"))
parser.add_argument("--backend", choices=["both", "clap", "jepa"], default="both")
parser.add_argument("--offline", action="store_true")
parser.add_argument("--parallel", action="store_true", help="Run the two independent backends concurrently")
parser.add_argument("--wait-for-extraction", action="store_true", help="Wait for an ongoing verified dataset download")
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
progress = lambda msg: print(msg, flush=True)
if args.limits and any(n < 1 for n in args.limits):
    raise ValueError("Subset counts must be positive")
if args.wait_for_extraction:
    while not all((args.dataset / f".{part}.extracted").exists()
                  for part in ["development", "validation", "evaluation"]):
        progress("Waiting for verified Clotho archive extraction")
        time.sleep(30)
provenance = prepare_clotho(args.dataset, output=args.output / "manifest.jsonl",
                            limits=dict(zip(["train", "val", "test"], args.limits)) if args.limits else None,
                            progress=progress)
reports = {}
preloaded = {}
backends = ["clap", "jepa"] if args.backend == "both" else [args.backend]
if args.parallel:
    # Transformer/PyTorch lazy imports and model construction share global initialization
    # state. Complete them serially, then parallelize independent forward passes.
    for backend in backends:
        print(f"Loading {backend} models serially", flush=True)
        preloaded[backend] = make_encoder(backend, args.config if backend == "jepa" else None, offline=args.offline)
        preloaded[backend].torch.set_num_threads(4)
    print("Both model backends are ready", flush=True)


def benchmark(backend):
    print(f"Loading {backend}", flush=True)
    encoder = preloaded.pop(backend) if args.parallel else make_encoder(
        backend, args.config if backend == "jepa" else None, offline=args.offline)
    encoder.torch.set_num_threads(4)
    report = run_experiment(provenance["manifest"], encoder, args.output / backend,
                            budgets=args.budgets, seeds=args.seeds,
                            methods=["native"] if backend == "clap" else ["procrustes", "ridge"],
                            cache_dir=args.dataset / "embedding-cache" / backend, progress=progress)
    del encoder
    if backend == "jepa":
        bundle = args.output / "selected-bundle"
        if not bundle.exists():
            export_alignment(args.output / "jepa", bundle)
    return backend, report


if args.parallel:
    with ThreadPoolExecutor(max_workers=len(backends)) as pool:
        reports.update(pool.map(benchmark, backends))
else:
    reports.update(benchmark(backend) for backend in backends)
if len(reports) == 2:
    comparison = {"dataset": provenance, "reports": reports,
                  "chance_text_to_audio_recall@1": 1 / provenance["splits"]["test"],
                  "limits": "Clotho retrieval only, not independent target-domain evaluation. CLAP has multimodal "
                            "pretraining; calibration counts do not measure total supervision. "
                            "Pretraining overlap has not been excluded. No test-driven model selection."}
    write_json(args.output / "comparison.json", comparison)
    print(json.dumps({"comparison": str(args.output / "comparison.json"), "splits": provenance["splits"]}))
