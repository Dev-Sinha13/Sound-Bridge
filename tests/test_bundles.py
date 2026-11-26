import json

import numpy as np
import pytest

from sonic_bridge.alignment import Alignment
from sonic_bridge.bundles import export_alignment, publish_experiment
from sonic_bridge.util import file_digest, read_json, write_json


def test_exports_only_validation_selected_artifact_and_pins_integrity(tmp_path):
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    encoder = {"backend": "jepa", "audio_model": "audio-model", "audio_revision": "a" * 40,
               "text_model": "text-model", "text_revision": "b" * 40}
    bridge = Alignment.fit(np.eye(3), np.eye(3), "ridge")
    bridge.save(experiment / "chosen.npz")
    write_json(experiment / "chosen.json", {"encoder": encoder})
    write_json(experiment / "report.json", {"encoder": encoder, "selected_artifact": "chosen.npz", "synthetic": True})
    result = export_alignment(experiment, tmp_path / "bundle")
    config = read_json(tmp_path / "bundle" / "config.json")
    assert result["synthetic"]
    assert config["alignment"] == "alignment.npz"
    assert config["alignment_sha256"] == file_digest(experiment / "chosen.npz")
    assert config["audio_revision"] == "a" * 40
    published = publish_experiment(experiment, tmp_path / "workspace", "example")
    assert (tmp_path / "workspace" / "experiments" / "example" / "chosen.npz").exists()
    assert publish_experiment(experiment, tmp_path / "workspace", "example") == published
    with pytest.raises(ValueError, match="Experiment name"):
        publish_experiment(experiment, tmp_path / "workspace", "../escape")
    assert export_alignment(experiment, tmp_path / "bundle") == result
    (tmp_path / "bundle" / "alignment.npz").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="new or empty"):
        export_alignment(experiment, tmp_path / "bundle")
    report = json.loads((experiment / "report.json").read_text())
    report["selected_artifact"] = "../../escape.npz"
    write_json(experiment / "report.json", report)
    with pytest.raises(ValueError, match="Invalid selected"):
        export_alignment(experiment, tmp_path / "second")
