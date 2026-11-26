"""Check a saved Audio-JEPA bridge through actual indexing and text search."""
import argparse
import json
from pathlib import Path

from sonic_bridge.encoders import make_encoder
from sonic_bridge.index import build_index, search
from sonic_bridge.util import write_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config", type=Path, required=True)
parser.add_argument("--folder", type=Path, default=Path("data/smoke_library"))
parser.add_argument("--database", type=Path, default=Path("data/indexes/synthetic-tones-jepa.sqlite"))
args = parser.parse_args()
encoder = make_encoder("aligned", args.config, offline=True)
encoder.base.torch.set_num_threads(4)
index = build_index(args.folder, args.database, encoder)
matches = search(args.database, encoder, "A high pitched steady tone", 2)
assert len(matches) == 2
result = {"encoder": encoder.identity, "index": index, "matches": matches,
          "scope": "Synthetic tone pipeline check through an exported real JEPA alignment. Not an accuracy benchmark."}
write_json(Path("artifacts/validation/aligned-smoke.json"), result)
print(json.dumps(result, indent=2))
