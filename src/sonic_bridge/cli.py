from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .demo import create_demo
from .encoders import make_encoder
from .experiments import import_clotho, run_experiment
from .index import build_index, search
from .server import serve


def model_options(parser):
    parser.add_argument("--backend", choices=["clap", "jepa", "aligned", "demo"], default="clap")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--offline", action="store_true", help="Only load cached model files")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Sonic Bridge: local sound search and alignment experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Generate synthetic tone fixtures (not semantic model evidence)")
    demo.add_argument("--output", type=Path, default=Path("data/demo"))
    index = commands.add_parser("index", help="Index an audio folder into overlapping timestamped windows")
    model_options(index)
    index.add_argument("folder", type=Path)
    index.add_argument("--database", type=Path, default=Path("data/indexes/library.sqlite"))
    index.add_argument("--window", type=float, default=10)
    index.add_argument("--hop", type=float, default=5)
    query = commands.add_parser("search", help="Search a previously built index")
    model_options(query)
    query.add_argument("query")
    query.add_argument("--database", type=Path, default=Path("data/indexes/library.sqlite"))
    query.add_argument("--limit", type=int, default=10)
    experiment = commands.add_parser("experiment", help="Fit and evaluate bridges on source-disjoint splits")
    model_options(experiment)
    experiment.add_argument("manifest", type=Path)
    experiment.add_argument("--output", type=Path, default=Path("artifacts/experiment"))
    experiment.add_argument("--budgets", type=int, nargs="+", default=[100, 500, 1000, 3000])
    experiment.add_argument("--seeds", type=int, nargs="+", default=[7, 23, 42])
    web = commands.add_parser("serve", help="Start the local browser app")
    model_options(web)
    web.add_argument("--workspace", type=Path, default=Path("data"))
    web.add_argument("--port", type=int, default=8765)
    clotho = commands.add_parser("import-clotho", help="Import one downloaded Clotho caption split")
    clotho.add_argument("captions", type=Path)
    clotho.add_argument("--audio-dir", type=Path, required=True)
    clotho.add_argument("--split", choices=["train", "val", "test"], required=True)
    clotho.add_argument("--output", type=Path, default=Path("data/clotho.jsonl"))
    args = parser.parse_args(argv)
    progress = lambda msg: print(msg, file=sys.stderr, flush=True)
    try:
        if args.command == "demo":
            result = create_demo(args.output)
        elif args.command == "import-clotho":
            result = import_clotho(args.captions, args.audio_dir, args.split, args.output)
        elif args.command == "serve":
            serve(args.workspace, args.backend, args.config, args.device, args.offline, args.port)
            return
        else:
            if args.command in {"index", "search"} and args.backend == "jepa":
                raise ValueError("Raw audio and text JEPA features are not aligned; use --backend aligned after fitting")
            if args.command == "experiment" and args.backend == "aligned":
                raise ValueError("Fit experiments with raw --backend jepa, not a previously aligned encoder")
            encoder = make_encoder(args.backend, args.config, args.device, args.offline)
            if args.command == "index":
                result = build_index(args.folder, args.database, encoder, args.window, args.hop, progress)
            elif args.command == "search":
                result = search(args.database, encoder, args.query, args.limit)
            else:
                result = run_experiment(args.manifest, encoder, args.output, args.budgets, args.seeds, progress=progress)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (ValueError, RuntimeError, OSError, KeyError) as e:
        parser.exit(1, f"Error: {e}\n")
