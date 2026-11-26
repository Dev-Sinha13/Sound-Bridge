import json

import pytest

from sonic_bridge.query_eval import choose_cutoff, load_queries, temporal_iou


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
