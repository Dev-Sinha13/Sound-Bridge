"""Download and verify the official 7 GB Clotho v2.1 release."""
import argparse
import json

from sonic_bridge.datasets import download_clotho

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", default="data/clotho")
args = parser.parse_args()
print(json.dumps(download_clotho(args.output, progress=lambda msg: print(msg, flush=True)), indent=2))
