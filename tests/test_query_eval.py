import json

import numpy as np
import pytest
import soundfile as sf

from sonic_bridge.encoders import DemoEncoder
from sonic_bridge.index import build_index
from sonic_bridge.query_eval import choose_cutoff, evaluate_queries, load_queries, temporal_iou


def test_cutoff_uses_validation_only_and_rejects_negatives():
    rows = [
        {"split": "val", "relevant": ["a"], "top_score": .6, "top_correct": True},
        {"split": "val", "relevant": [], "top_score": .4, "top_correct": False},
        {"split": "test", "relevant": [], "top_score": .9, "top_correct": False},
    ]
    cutoff = choose_cutoff(rows, 0)
    assert .4 < cutoff <= .6
    rows[-1]["top_score"] = -.9
    assert choose_cutoff(rows, 0) == cutoff
    with pytest.raises(ValueError, match="both present and absent"):
        choose_cutoff(rows[1:])


def test_judgment_source_leakage_and_timestamps(tmp_path):
    path = tmp_path / "queries.json"
    rows = [{"query": "a sound", "split": "val", "group": "a", "relevant": [{"file": "a.wav"}]},
            {"query": "other sound", "split": "test", "group": "b", "relevant": [{"file": "a.wav"}]}]
    path.write_text(json.dumps({"queries": rows}))
    with pytest.raises(ValueError, match="leakage"):
        load_queries(path)
    rows[1]["relevant"] = [{"file": "b.wav", "start": 2, "end": 1}]
    path.write_text(json.dumps({"queries": rows}))
    with pytest.raises(ValueError, match="Timestamp"):
        load_queries(path)
    assert temporal_iou({"start": 2, "end": 4}, {"start": 1, "end": 5}) == .5


def test_query_evaluation_reports_absent_sounds_and_localization(tmp_path):
    folder = tmp_path / "audio"
    folder.mkdir()
    t = np.arange(32000) / 16000
    sf.write(folder / "low.wav", .1 * np.sin(2 * np.pi * 180 * t), 16000)
    sf.write(folder / "high.wav", .1 * np.sin(2 * np.pi * 2800 * t), 16000)
    rows = [
        {"split": "val", "group": "low", "query": "low", "relevant": [{"file": "low.wav"}]},
        {"split": "val", "group": "absent-val", "query": "mid", "relevant": []},
        {"split": "test", "group": "high", "query": "high", "relevant": [{"file": "high.wav", "start": 0, "end": 2}]},
        {"split": "test", "group": "absent-test", "query": "middle", "relevant": []},
    ]
    suite = tmp_path / "queries.json"
    suite.write_text(json.dumps({"queries": rows}))
    encoder = DemoEncoder()
    database = tmp_path / "index.sqlite"
    build_index(folder, database, encoder)
    result = evaluate_queries(suite, database, encoder, tmp_path / "report.json")
    assert result["splits"]["test"]["recall@1"] == 1
    assert result["splits"]["test"]["false_accept_rate"] == 0
    assert result["splits"]["test"]["temporal_recall@1_iou50"] == 1
    assert all(r["failure"] is None for r in result["records"])
