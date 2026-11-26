import json

import numpy as np
import pytest
import soundfile as sf

from sonic_bridge.audio import read_audio, windows
from sonic_bridge.demo import create_demo
from sonic_bridge.encoders import DemoEncoder
from sonic_bridge.experiments import bootstrap_recall, load_manifest, retrieval, run_experiment
from sonic_bridge.index import build_index, index_info, search


def test_window_tail_and_resampling(tmp_path):
    path = tmp_path / "stereo.wav"
    sf.write(path, np.ones((24000 * 12, 2), dtype=np.float32) * .1, 24000)
    chunks = list(windows(path, 16000, 10, 5))
    assert [(w.start, w.end) for w in chunks] == [(0, 10), (5, 12)]
    assert chunks[0].samples.shape == (160000,)
    assert len(read_audio(path, 16000, 11, 12)) == 16000
    with pytest.raises(ValueError):
        list(windows(path, 16000, 0, 0))


def test_index_search_cache_and_rollback(tmp_path):
    demo = create_demo(tmp_path / "demo")
    db = tmp_path / "index.sqlite"
    encoder = DemoEncoder()
    first = build_index(demo["folder"], db, encoder)
    assert first["files"] == first["segments"] == 30
    assert build_index(demo["folder"], db, encoder)["changed"] == 0
    results = search(db, encoder, "high tone", limit=3)
    assert len(results) == 3 and all("high" in r["name"] for r in results)
    assert all(r["end"] == 2 for r in results)
    with pytest.raises(ValueError, match="different settings"):
        build_index(demo["folder"], db, encoder, seconds=4, hop=2)
    broken = tmp_path / "demo" / "audio" / "broken.wav"
    broken.write_text("not audio")
    with pytest.raises(sf.LibsndfileError):
        build_index(demo["folder"], db, encoder)
    assert index_info(db)["files"] == 30
    assert search(db, encoder, "low tone")
    encoder.identity = {"backend": "wrong"}
    with pytest.raises(ValueError, match="does not match"):
        search(db, encoder, "low tone")


def test_multiple_captions_and_conservative_ties():
    audio = np.eye(2)
    text = np.array([[1, 0], [1, 0], [0, 1]])
    result = retrieval(audio, text, ["a", "b"], ["a", "a", "b"])
    assert result["text_to_audio"]["recall@1"] == 1
    assert result["audio_to_text"]["recall@1"] == 1
    result = retrieval(np.ones((2, 2)), np.ones((2, 2)), ["a", "b"], ["a", "b"])
    assert result["text_to_audio"]["recall@1"] == 0


def test_manifest_rejects_content_and_group_leakage(tmp_path):
    fixture = create_demo(tmp_path)
    manifest = tmp_path / "manifest.jsonl"
    rows = [json.loads(s) for s in manifest.read_text().splitlines()]
    rows[-1]["audio"] = rows[0]["audio"]
    manifest.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="leakage"):
        load_manifest(fixture["manifest"])
    create_demo(tmp_path)
    rows = [json.loads(s) for s in manifest.read_text().splitlines()]
    rows[-1]["group"] = rows[0]["group"]
    manifest.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="leakage"):
        load_manifest(manifest)


def test_experiment_end_to_end_and_train_only_metadata(tmp_path):
    fixture = create_demo(tmp_path / "fixture")
    output = tmp_path / "results"
    report = run_experiment(fixture["manifest"], DemoEncoder(), output, budgets=[6, 18, 100], seeds=[7])
    assert report["synthetic"]
    assert report["skipped_budgets"] == [100]
    assert len(report["results"]) == 5
    assert report["splits"] == {"train": 18, "val": 6, "test": 6}
    for path in output.glob("*-seed*.json"):
        metadata = json.loads(path.read_text())
        assert all(i.startswith("train-") for i in metadata["training_ids"])
        assert metadata["rows"] <= 18
    assert (output / "report.json").exists()
    selected = json.loads((output / "selected.json").read_text())
    best_validation = max(r["validation"]["text_to_audio"]["recall@1"]
                          for r in report["results"] if "validation" in r)
    assert selected["validation"]["text_to_audio"]["recall@1"] == best_validation


def test_synthetic_dataset_stays_labeled_with_a_research_backend(tmp_path):
    fixture = create_demo(tmp_path / "fixture")
    encoder = DemoEncoder()
    encoder.identity = {"backend": "jepa", "fixture": True}
    report = run_experiment(fixture["manifest"], encoder, tmp_path / "output", budgets=[6], seeds=[7])
    assert report["synthetic"]
    assert all(r["method"] != "native" for r in report["results"])


def test_native_only_benchmark_needs_no_calibration_budget(tmp_path):
    fixture = create_demo(tmp_path / "fixture")
    report = run_experiment(fixture["manifest"], DemoEncoder(), tmp_path / "output", methods=["native"])
    assert len(report["results"]) == 1
    assert report["selected_artifact"] is None
    assert report["encoded_recordings"] == 6
    assert report["splits"] == {"train": 18, "val": 6, "test": 6}
    assert report["results"][0]["test"]["text_to_audio"]["candidates"] == 6


def test_bootstrap_keeps_caption_groups_and_is_deterministic():
    ranks = np.array([1, 1, 7, 7])
    ids = np.array(["a", "a", "b", "b"])
    result = bootstrap_recall(ranks, ids, iterations=200)
    assert result == bootstrap_recall(ranks, ids, iterations=200)
    assert result["recall@1"] == [0.0, 1.0]
    assert result["recall@10"] == [1.0, 1.0]
