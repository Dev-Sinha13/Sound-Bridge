"""Reproducible download and preparation of the official Clotho v2.1 release."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .util import write_json

CLOTHO_RECORD = "https://zenodo.org/records/4783391"
# Checksums published in the immutable Zenodo v2.1 record.
FILES = {
    "clotho_audio_development.7z": "c8b05bc7acdb13895bb3c6a29608667e",
    "clotho_audio_validation.7z": "7dba730be08bada48bd15dc4e668df59",
    "clotho_audio_evaluation.7z": "4569624ccadf96223f19cb59fe4f849f",
    "clotho_captions_development.csv": "d4090b39ce9f2491908eebf4d5b09bae",
    "clotho_captions_validation.csv": "5879e023032b22a2c930aaa0528bead4",
    "clotho_captions_evaluation.csv": "1b16b9e57cf7bdb7f13a13802aeb57e2",
    "clotho_metadata_development.csv": "170d20935ecfdf161ce1bb154118cda5",
    "clotho_metadata_validation.csv": "2e010427c56b1ce6008b0f03f41048ce",
    "clotho_metadata_evaluation.csv": "13946f054d4e1bf48079813aac61bf77",
    "LICENSE": "38d422ac8a2c9c35c288232576e2e810",
}


def md5(path):
    checksum = hashlib.md5(usedforsecurity=False)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def download_verified(url, destination, checksum, progress=lambda msg: None):
    """Resume interrupted downloads, verify official bytes, and publish atomically."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if md5(destination) == checksum:
            return destination
        raise ValueError(f"Existing file fails checksum: {destination}. Move it aside before retrying.")
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "Sound-Bridge/0.1 (research dataset download)"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as response:
                resumed = response.status == 206
                if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise ValueError("Server returned an unexpected download range")
                received = offset if resumed else 0
                last_log = time.monotonic()
                with partial.open("ab" if resumed else "wb") as output:
                    for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                        output.write(block)
                        received += len(block)
                        if time.monotonic() - last_log >= 15:
                            progress(f"{destination.name}: {received / 1e6:.0f} MB downloaded")
                            last_log = time.monotonic()
            break
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == 4:
                raise
            progress(f"Retrying {destination.name} (attempt {attempt + 2})")
            time.sleep(min(2 ** attempt, 8))
    if md5(partial) != checksum:
        raise ValueError(f"Download checksum mismatch: {partial}")
    partial.replace(destination)
    progress(f"Verified {destination.name}")
    return destination


def download_clotho(output, progress=lambda msg: None):
    try:
        import py7zr  # noqa: F401 - check the optional dependency before starting a large download
    except ImportError as e:
        raise RuntimeError('Install dataset dependencies: pip install -e ".[research]"') from e
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 20 * 1024 ** 3:
        raise ValueError("Clotho download/extraction requires at least 20 GiB free")

    def download(item):
        name, checksum = item
        url = f"https://zenodo.org/api/records/4783391/files/{name}/content"
        return download_verified(url, output / "downloads" / name, checksum, progress)

    with ThreadPoolExecutor(max_workers=3) as pool:
        # Each result is consumed so a failed file never silently produces a partial manifest.
        list(pool.map(download, sorted(FILES.items(), key=lambda item: item[0].endswith(".7z"))))
    for part in ["development", "validation", "evaluation"]:
        extract_clotho_archive(output, part, progress)
    return prepare_clotho(output, progress=progress)


def extract_clotho_archive(output, part, progress=lambda msg: None):
    import py7zr
    if part not in {"development", "validation", "evaluation"}:
        raise ValueError("Invalid Clotho split")
    output = Path(output).resolve()
    marker = output / f".{part}.extracted"
    if marker.exists():
        return
    progress(f"Extracting {part}")
    archive = output / "downloads" / f"clotho_audio_{part}.7z"
    if md5(archive) != FILES[archive.name]:
        raise ValueError("Archive checksum failed before extraction")
    with py7zr.SevenZipFile(archive) as seven:
        for member in seven.list():
            target = (output / member.filename).resolve()
            if not target.is_relative_to(output) or getattr(member, "is_symlink", False):
                raise ValueError("Unsafe archive member")
        seven.extractall(path=output)
    marker.write_text(FILES[archive.name], encoding="utf-8")


def prepare_clotho(folder, output=None, limits=None, seed=2026, progress=lambda msg: None):
    """Build the canonical split manifest and retain per-file source/license metadata."""
    folder = Path(folder).resolve()
    output = Path(output) if output else folder / "manifest.jsonl"
    rows = []
    for source_split, split in [("development", "train"), ("validation", "val"), ("evaluation", "test")]:
        with (folder / "downloads" / f"clotho_metadata_{source_split}.csv").open(
                encoding="latin-1", newline="") as stream:
            metadata = {row["file_name"]: row for row in csv.DictReader(stream)}
        with (folder / "downloads" / f"clotho_captions_{source_split}.csv").open(
                encoding="utf-8-sig", newline="") as stream:
            captions = list(csv.DictReader(stream))
        captions.sort(key=lambda row: row["file_name"])
        if limits and split in limits and limits[split] < len(captions):
            order = np.random.default_rng(seed).permutation(len(captions))[:limits[split]]
            captions = [captions[i] for i in sorted(order)]
        for caption in captions:
            filename = caption["file_name"]
            audio = folder / source_split / filename
            if not audio.is_file():
                raise ValueError(f"Missing Clotho file: {audio}")
            info = metadata.get(filename)
            if info is None:
                raise ValueError(f"Missing provenance for {filename}")
            # A source recording can have several excerpts; use its Freesound URL as group.
            source = info.get("sound_link") or info.get("url") or filename
            rows.append({"id": f"clotho:{filename}", "audio": str(audio), "split": split,
                         "group": source, "captions": [caption[f"caption_{i}"] for i in range(1, 6)],
                         "synthetic": False, "dataset": "Clotho v2.1", "source": source,
                         "license": info.get("license", "See official metadata"), "metadata": info})
        progress(f"Prepared {len(captions)} {split} recordings")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    summary = {"dataset": "Clotho v2.1", "source": CLOTHO_RECORD,
               "manifest": str(output.resolve()), "seed": seed,
               "splits": {s: sum(r["split"] == s for r in rows) for s in ["train", "val", "test"]},
               "subset": bool(limits), "archive_checksums": FILES,
               "license": "Per-audio Freesound licenses; captions under Tampere University's license. "
                          "See downloads/LICENSE and official metadata; recordings are not redistributed."}
    write_json(output.with_suffix(".provenance.json"), summary)
    return summary
