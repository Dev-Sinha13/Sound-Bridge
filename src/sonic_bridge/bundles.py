"""Export a validation-selected bridge as a portable, integrity-checked config bundle."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from .util import file_digest, read_json, write_json


def publish_experiment(experiment, workspace, name):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name):
        raise ValueError("Experiment name must use letters, numbers, hyphens or underscores")
    experiment = Path(experiment).resolve()
    report = read_json(experiment / "report.json")
    destination = Path(workspace).resolve() / "experiments" / name
    if destination.exists() and (destination / "report.json").exists():
        if read_json(destination / "report.json") == report:
            return {"experiment": name, "folder": str(destination)}
        raise ValueError("A different experiment is already published under this name")
    destination.mkdir(parents=True, exist_ok=True)
    selected = report.get("selected_artifact")
    if selected:
        source = (experiment / selected).resolve()
        if not source.is_relative_to(experiment) or source.suffix != ".npz":
            raise ValueError("Invalid selected alignment path")
        shutil.copyfile(source, destination / source.name)
        shutil.copyfile(source.with_suffix(".json"), (destination / source.name).with_suffix(".json"))
    write_json(destination / "report.json", report)
    return {"experiment": name, "folder": str(destination)}


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
        config_path = destination / "config.json"
        provenance_path = destination / "provenance.json"
        if config_path.is_file() and provenance_path.is_file() and (destination / "alignment.npz").is_file():
            existing = read_json(config_path)
            provenance = read_json(provenance_path)
            expected = {k: encoder[k] for k in ["audio_model", "audio_revision", "text_model", "text_revision"]}
            if (provenance.get("source_report_sha256") == file_digest(experiment / "report.json")
                    and existing.get("alignment") == "alignment.npz"
                    and existing.get("trust_remote_code") is True
                    and all(existing.get(k) == v for k, v in expected.items())
                    and existing.get("alignment_sha256") == file_digest(source)
                    and file_digest(destination / "alignment.npz") == file_digest(source)
                    and read_json(destination / "alignment.json") == metadata):
                return {"config": str(config_path), "synthetic": report.get("synthetic", False)}
        raise ValueError("Bundle destination must be new or empty")
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination / "alignment.npz")
    shutil.copyfile(source.with_suffix(".json"), destination / "alignment.json")
    config = {k: encoder[k] for k in ["audio_model", "audio_revision", "text_model", "text_revision"]}
    config.update(trust_remote_code=True, alignment="alignment.npz", alignment_sha256=file_digest(source),
                  synthetic_calibration=bool(report.get("synthetic", False)))
    write_json(destination / "config.json", config)
    write_json(destination / "provenance.json", {"source_report_sha256": file_digest(experiment / "report.json"),
               "selection": "Highest validation text-to-audio Recall@1; no test selection",
               "selected_artifact": selected, "synthetic": report.get("synthetic", False),
               "encoder": encoder})
    return {"config": str(destination / "config.json"), "synthetic": report.get("synthetic", False)}
