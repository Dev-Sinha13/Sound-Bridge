import numpy as np
import pytest
import soundfile as sf

from sonic_bridge.encoders import DemoEncoder
from sonic_bridge.index import build_index, index_info, search


def test_short_event_multiscale_filters_and_similarity(tmp_path):
    # A brief high tone is overwhelmed by a low tone in a whole-clip embedding.
    rate = 16000
    t = np.arange(rate * 10) / rate
    mixed = .3 * np.sin(2 * np.pi * 180 * t)
    mixed[rate * 4:rate * 6] = .3 * np.sin(2 * np.pi * 2800 * t[rate * 4:rate * 6])
    folder = tmp_path / "audio"
    folder.mkdir()
    sf.write(folder / "event.wav", mixed, rate)
    sf.write(folder / "low.wav", .3 * np.sin(2 * np.pi * 180 * t), rate)
    encoder = DemoEncoder()
    coarse, multi = tmp_path / "coarse.sqlite", tmp_path / "multi.sqlite"
    build_index(folder, coarse, encoder)
    build_index(folder, multi, encoder, scales=[2, 5])
    assert search(coarse, encoder, "high", min_score=.8) == []
    matches = search(multi, encoder, "high", min_score=.8)
    assert matches[0]["name"] == "event.wav"
    assert (matches[0]["start"], matches[0]["end"]) == (4, 6)
    assert search(multi, encoder, "high", name_filter="low", min_score=.8) == []
    assert len(search(multi, encoder, "low", unique_files=True)) == 2
    similar = search(multi, encoder, "unused", similar_to=matches[0]["id"])
    assert all(m["id"] != matches[0]["id"] for m in similar)
    with pytest.raises(ValueError, match="Minimum similarity"):
        search(multi, encoder, "high", min_score=2)


def test_cancelled_index_resumes_embeddings_and_removes_stale_files(tmp_path):
    folder = tmp_path / "audio"
    folder.mkdir()
    sf.write(folder / "long.wav", np.ones(16000 * 25) * .1, 16000)
    encoder = DemoEncoder()
    database = tmp_path / "index.sqlite"
    calls = []
    original = encoder.encode_audio

    def encode(samples):
        calls.append(len(samples))
        return original(samples)

    encoder.encode_audio = encode

    def interrupt(message):
        if "final" in message:
            raise RuntimeError("cancelled")

    with pytest.raises(RuntimeError, match="cancelled"):
        build_index(folder, database, encoder, seconds=5, hop=2.5, progress=interrupt)
    previous = len(calls)
    assert previous == 2 and index_info(database)["files"] == 0
    build_index(folder, database, encoder, seconds=5, hop=2.5)
    assert len(calls) == previous + 1  # The two committed cache batches are reused.
    (folder / "long.wav").unlink()
    assert build_index(folder, database, encoder, seconds=5, hop=2.5)["segments"] == 0
