"""Fetch the official ESC-50 release at a fixed Git revision; do not redistribute audio."""
import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from sonic_bridge.util import file_digest, write_json

REVISION = "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("data/esc50"))
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
archive = args.output / "esc50.zip"
url = f"https://codeload.github.com/karolpiczak/ESC-50/zip/{REVISION}"
if not archive.exists():
    partial = archive.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as f:
        shutil.copyfileobj(response, f, length=4 * 1024 * 1024)
    partial.replace(archive)
with zipfile.ZipFile(archive) as zipped:
    bad = zipped.testzip()
    if bad:
        raise ValueError(f"Archive failed CRC: {bad}")
    for name in zipped.namelist():
        if not (args.output.resolve() / name).resolve().is_relative_to(args.output.resolve()):
            raise ValueError("Unsafe archive path")
    zipped.extractall(args.output)
folder = args.output / f"ESC-50-{REVISION}"
write_json(args.output / "provenance.json", {"repository": "https://github.com/karolpiczak/ESC-50",
           "revision": REVISION, "archive_sha256": file_digest(archive), "folder": str(folder.resolve()),
           "license": "See the downloaded repository LICENSE; recordings are not redistributed."})
print(folder.resolve())
