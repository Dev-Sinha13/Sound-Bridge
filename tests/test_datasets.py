import csv
import hashlib
import json

import pytest

from sonic_bridge.datasets import download_verified, prepare_clotho, select_clotho_rows


def test_download_validates_local_bytes_without_network(tmp_path):
    target = tmp_path / "archive.bin"
    target.write_bytes(b"validated bytes")
    checksum = hashlib.md5(target.read_bytes(), usedforsecurity=False).hexdigest()
    assert download_verified("https://invalid.invalid", target, checksum) == target
    with pytest.raises(ValueError, match="fails checksum"):
        download_verified("https://invalid.invalid", target, "0" * 32)


def test_clotho_manifest_preserves_latin1_metadata_and_official_splits(tmp_path):
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    for part in ["development", "validation", "evaluation"]:
        folder = tmp_path / part
        folder.mkdir()
        (folder / "clip.wav").write_bytes(b"fixture")
        with (downloads / f"clotho_captions_{part}.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["file_name", *[f"caption_{i}" for i in range(1, 6)]])
            writer.writeheader()
            writer.writerow({"file_name": "clip.wav", **{f"caption_{i}": "A passing car" for i in range(1, 6)}})
        with (downloads / f"clotho_metadata_{part}.csv").open("w", newline="", encoding="latin-1") as stream:
            writer = csv.DictWriter(stream, fieldnames=["file_name", "sound_link", "manufacturer", "license"])
            writer.writeheader()
            writer.writerow({"file_name": "clip.wav", "sound_link": f"https://example.org/{part}",
                             "manufacturer": "André", "license": "CC-BY"})
    summary = prepare_clotho(tmp_path)
    assert summary["splits"] == {"train": 1, "val": 1, "test": 1}
    rows = [json.loads(line) for line in (tmp_path / "manifest.jsonl").read_text().splitlines()]
    assert rows[0]["group"] == "https://example.org/development"
    assert rows[0]["metadata"]["manufacturer"] == "André"
    assert rows[0]["license"] == "CC-BY"
    assert not rows[0]["synthetic"]


def test_source_group_sampling_is_stable_and_does_not_count_excerpts_as_pairs():
    captions = [{"file_name": name} for name in ["a.wav", "b.wav", "c.wav", "d.wav"]]
    metadata = {"a.wav": {"sound_id": "1"}, "b.wav": {"sound_id": "1"},
                "c.wav": {"sound_id": "2"}, "d.wav": {"sound_id": "3"}}
    selected = select_clotho_rows(captions, metadata, limit=3)
    assert len(selected) == 3
    assert len({metadata[row["file_name"]]["sound_id"] for row in selected}) == 3
    smaller = select_clotho_rows(captions, metadata, limit=2)
    assert {row["file_name"] for row in smaller} <= {row["file_name"] for row in selected}
