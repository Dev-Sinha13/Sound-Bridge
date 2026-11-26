"""Export a validation-selected bridge as a portable, integrity-checked config bundle."""
from __future__ import annotations

import shutil
from pathlib import Path

from .util import file_digest, read_json, write_json


def export_alignment(experiment, destination):
    experiment, destination = Path(experiment).resolve(), Path(destination).resolve()
    report = read_json(experiment / "report.json")
    selected = report.get("selected_artifact")
    if not selected:
        raise ValueError("This experiment has no validation-selected alignment")
    source = (experiment / selected).resolve()
    if not source.is_relative_to(experiment) or not source.is_file():
        raise ValueError("Invalid selected alignment path")
    metadata = read_json(source.with_suffix(".json"))
    encoder = metadata["encoder"]
    if encoder != report["encoder"] or encoder.get("backend") != "jepa":
        raise ValueError("Export requires an Audio-JEPA/text alignment with matching provenance")
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Bundle destination must be new or empty")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination / "alignment.npz")
    shutil.copyfile(source.with_suffix(".json"), destination / "alignment.json")
    config = {k: encoder[k] for k in ["audio_model", "audio_revision", "text_model", "text_revision"]}
    config.update(trust_remote_code=True, alignment="alignment.npz", alignment_sha256=file_digest(source))
    write_json(destination / "config.json", config)
    write_json(destination / "provenance.json", {"source_report_sha256": file_digest(experiment / "report.json"),
               "selection": "Highest validation text-to-audio Recall@1; no test selection",
               "selected_artifact": selected, "synthetic": report.get("synthetic", False),
               "encoder": encoder})
    return {"config": str(destination / "config.json"), "synthetic": report.get("synthetic", False)}
