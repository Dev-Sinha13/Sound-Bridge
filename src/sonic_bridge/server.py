from __future__ import annotations

import io
import json
import re
import secrets
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import soundfile as sf

from .audio import read_audio
from .bundles import export_alignment
from .demo import create_demo
from .encoders import make_encoder
from .experiments import run_experiment
from .index import build_index, index_info, search, segment
from .util import read_json, write_json


class JobCancelled(Exception):
    pass


class Application:
    def __init__(self, workspace, backend="clap", config=None, device="cpu", offline=False):
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.backend, self.config, self.device, self.offline = backend, config, device, offline
        self.encoder = None
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.lock = threading.Lock()
        self.jobs = {}
        self.cancel_signals = {}
        self.token = secrets.token_urlsafe(32)
        self.watches = {}

    def watch(self, name, enabled):
        info = index_info(self.database(name))
        if enabled:
            self.watches[name] = {"settings": info["settings"], "checked": 0, "signature": None, "error": None}
        else:
            self.watches.pop(name, None)
        return {"name": name, "watching": bool(enabled)}

    def poll_watches(self):
        # Called by the browser's state polling: watching stops when the app is closed.
        from .audio import discover
        for name, watch in list(self.watches.items()):
            if time.monotonic() - watch["checked"] < 10:
                continue
            watch["checked"] = time.monotonic()
            settings = watch["settings"]
            if self.encoder is None or settings["encoder"] != self.encoder.identity:
                continue
            try:
                last = self.jobs.get(watch.get("job"), {})
                if last.get("status") in {"failed", "cancelled"}:
                    watch["signature"] = None
                signature = [(str(p), p.stat().st_size, p.stat().st_mtime_ns)
                             for p in discover(Path(settings["root"]))]
                if signature != watch["signature"]:
                    job = self.submit("index", lambda p, s=settings, n=name: build_index(
                        s["root"], self.database(n), self.get_encoder(), s["window"], s["hop"], p,
                        scales=s.get("scales")))
                    watch["job"] = job["job"]
                    watch["signature"] = signature
                    watch["error"] = None
            except (OSError, ValueError) as error:
                watch["error"] = str(error)

    def get_encoder(self):
        if self.encoder is None:
            self.encoder = make_encoder(self.backend, self.config, self.device, self.offline)
            torch = getattr(self.encoder, "torch", getattr(getattr(self.encoder, "base", None), "torch", None))
            if torch:
                torch.set_num_threads(4)
        return self.encoder

    def switch_backend(self, backend, config, progress):
        if backend not in {"clap", "finelap", "aligned", "demo"}:
            raise ValueError("Search supports CLAP, a fitted alignment, or the synthetic demo")
        progress(f"Loading {backend} models")
        encoder = make_encoder(backend, config, self.device, self.offline)
        torch = getattr(encoder, "torch", getattr(getattr(encoder, "base", None), "torch", None))
        if torch:
            torch.set_num_threads(4)
        progress("Models ready")
        # Preserve the current working backend if the requested model fails to load.
        self.encoder, self.backend, self.config = encoder, backend, config
        return {"backend": backend, "encoder": encoder.identity}

    def database(self, name):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", name or ""):
            raise ValueError("Index name must use letters, numbers, hyphens or underscores")
        return self.workspace / "indexes" / f"{name}.sqlite"

    def submit(self, kind, callback):
        with self.lock:
            if any(j["status"] in {"queued", "running"} for j in self.jobs.values()):
                raise ValueError("Another job is running. Wait for it to finish.")
            key = secrets.token_hex(8)
            self.jobs[key] = {"id": key, "kind": kind, "status": "queued", "message": "Queued"}
            self.cancel_signals[key] = threading.Event()

        def progress(message):
            if self.cancel_signals[key].is_set():
                raise JobCancelled()
            with self.lock:
                self.jobs[key]["message"] = message

        def run():
            started = time.perf_counter()
            with self.lock:
                self.jobs[key]["status"] = "running"
            try:
                progress("Starting")
                result = callback(progress)
                if self.cancel_signals[key].is_set():
                    raise JobCancelled()
                with self.lock:
                    self.jobs[key].update(status="complete", result=result, message="Complete")
            except JobCancelled:
                with self.lock:
                    self.jobs[key].update(status="cancelled", message="Cancelled; completed embedding caches are retained")
            except Exception as e:  # noqa: BLE001 - report arbitrary model/runtime failures to the job UI
                traceback.print_exc()
                with self.lock:
                    self.jobs[key].update(status="failed", message=str(e))
            finally:
                with self.lock:
                    self.jobs[key]["seconds"] = round(time.perf_counter() - started, 2)

        self.pool.submit(run)
        return {"job": key}

    def cancel(self, key):
        with self.lock:
            if key not in self.jobs:
                raise ValueError("Unknown job")
            if self.jobs[key]["status"] not in {"queued", "running"}:
                raise ValueError("Job has already finished")
            self.cancel_signals[key].set()
            self.jobs[key]["message"] = "Cancellation requested; finishing the current model operation"
        return {"job": key, "cancellation_requested": True}

    def state(self):
        self.poll_watches()
        with self.lock:
            jobs = [dict(j) for j in self.jobs.values()][-20:]
        indexes = [index_info(p) for p in sorted((self.workspace / "indexes").glob("*.sqlite"))]
        for index in indexes:
            identity = index["settings"]["encoder"] if index["settings"] else None
            index["compatible"] = identity == self.encoder.identity if self.encoder else bool(
                identity and identity.get("backend") == self.backend)
        reports = []
        for path in sorted((self.workspace / "experiments").glob("*/report.json")):
            report = read_json(path)
            reports.append({"name": path.parent.name, **report})
        return {"backend": self.backend, "offline": self.offline, "workspace": str(self.workspace),
                "synthetic": self.backend == "demo", "indexes": indexes, "jobs": jobs,
                "reports": reports, "token": self.token, "can_search": self.backend != "jepa",
                "synthetic_alignment": bool(getattr(self.encoder, "synthetic_calibration", False)),
                "watches": {name: {"error": w["error"]} for name, w in self.watches.items()},
                "default_config": str(self.config or "configs/jepa.json")}


def make_handler(app):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def valid_host(self):
            return self.headers.get("Host") in {f"127.0.0.1:{self.server.server_port}",
                                                f"localhost:{self.server.server_port}"}

        def respond(self, body, code=200, content_type="application/json"):
            if content_type == "application/json":
                body = json.dumps(body, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; "
                             "media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if not self.valid_host():
                return self.respond({"error": "Invalid host"}, 403)
            parsed = urlparse(self.path)
            try:
                if parsed.path == "/api/state":
                    return self.respond(app.state())
                if parsed.path == "/api/audio":
                    args = parse_qs(parsed.query)
                    row = segment(app.database(args["index"][0]), int(args["id"][0]))
                    samples = read_audio(Path(row["path"]), 16000, row["start"], row["end"])
                    buf = io.BytesIO()
                    sf.write(buf, samples, 16000, format="WAV", subtype="PCM_16")
                    return self.respond(buf.getvalue(), content_type="audio/wav")
                if parsed.path == "/api/waveform":
                    args = parse_qs(parsed.query)
                    row = segment(app.database(args["index"][0]), int(args["id"][0]))
                    samples = read_audio(Path(row["path"]), 8000, row["start"], row["end"])
                    peaks = [float(np.max(np.abs(chunk))) for chunk in np.array_split(samples, min(160, len(samples)))]
                    return self.respond({"peaks": peaks, "start": row["start"], "end": row["end"]})
                static = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}
                if parsed.path in static:
                    name = static[parsed.path]
                    mime = {"html": "text/html; charset=utf-8", "js": "text/javascript", "css": "text/css"}
                    return self.respond((Path(__file__).parent / "static" / name).read_bytes(),
                                        content_type=mime[name.rsplit(".", 1)[1]])
                return self.respond({"error": "Not found"}, 404)
            except (ValueError, KeyError, OSError) as e:
                return self.respond({"error": str(e)}, 400)

        def do_POST(self):
            if not self.valid_host() or self.headers.get("X-Sonic-Token") != app.token:
                return self.respond({"error": "Refresh this local page before submitting"}, 403)
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://127.0.0.1:{self.server.server_port}",
                                          f"http://localhost:{self.server.server_port}"}:
                return self.respond({"error": "Cross-origin requests are disabled"}, 403)
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 65536 or self.headers.get_content_type() != "application/json":
                    raise ValueError("Expected a JSON request under 64 KB")
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise TypeError("Request body must be a JSON object")
                if self.path in {"/api/index", "/api/search"} and app.backend == "jepa":
                    raise ValueError("Raw JEPA features are not aligned. Run an experiment and export a bridge first.")
                if self.path == "/api/cancel":
                    return self.respond(app.cancel(body["job"]))
                if self.path == "/api/watch":
                    return self.respond(app.watch(body["index"], body.get("enabled", True)))
                if self.path == "/api/feedback":
                    query = str(body["query"]).strip()
                    label = body["label"]
                    if not query or len(query) > 2000 or label not in {"correct", "wrong_sound", "wrong_timestamp", "absent"}:
                        raise ValueError("Choose a valid judgment and a query under 2000 characters")
                    database = app.database(body["index"])
                    info = index_info(database)
                    row = segment(database, int(body["segment"])) if label != "absent" else None
                    key = secrets.token_hex(12)
                    write_json(app.workspace / "judgments" / f"{key}.json",
                               {"query": query, "label": label, "segment": row, "index": info,
                                "recorded_at_unix": time.time(), "source": "user_judgment"})
                    return self.respond({"saved": key})
                if self.path == "/api/backend":
                    return self.respond(app.submit("models", lambda progress: app.switch_backend(
                        body["backend"], body.get("config"), progress)), 202)
                if self.path == "/api/index":
                    folder, database = Path(body["folder"]).resolve(), app.database(body["name"])
                    seconds, hop = float(body.get("window", 10)), float(body.get("hop", 5))
                    return self.respond(app.submit("index", lambda p: build_index(
                        folder, database, app.get_encoder(), seconds, hop, p, scales=body.get("scales"))), 202)
                if self.path == "/api/search":
                    database = app.database(body["index"])
                    def do_search(progress):
                        progress("Encoding query and searching")
                        return {"index": body["index"], "query": body["query"],
                                "matches": search(database, app.get_encoder(), body["query"], 10,
                                                  min_score=body.get("min_score"),
                                                  name_filter=body.get("name_filter", ""),
                                                  unique_files=body.get("unique_files", False),
                                                  similar_to=body.get("similar_to"))}
                    return self.respond(app.submit("search", do_search), 202)
                if self.path == "/api/demo":
                    if app.backend != "demo":
                        raise ValueError("Restart with --backend demo to use the synthetic fixture")
                    return self.respond(create_demo(app.workspace / "demo"))
                if self.path == "/api/experiment":
                    name = body["name"]
                    app.database(name)  # Apply the same safe-name policy to report directories.
                    destination = app.workspace / "experiments" / name
                    if destination.exists():
                        raise ValueError("Experiment name already exists; choose a new name")
                    budgets = tuple(int(v) for v in body.get("budgets", [100, 500, 1000]))
                    backend = body.get("backend", "jepa" if app.backend == "aligned" else app.backend)
                    if backend not in {"demo", "clap", "finelap", "jepa"}:
                        raise ValueError("Choose CLAP or raw Audio-JEPA for an experiment")
                    config = body.get("config") or app.config

                    def experiment(progress):
                        if backend == app.backend and config == app.config:
                            encoder = app.get_encoder()
                        else:
                            encoder = make_encoder(backend, config, app.device, app.offline)
                            if hasattr(encoder, "torch"):
                                encoder.torch.set_num_threads(4)
                        methods = ["native"] if backend in {"clap", "finelap"} else body.get("methods", ["procrustes", "ridge"])
                        return run_experiment(body["manifest"], encoder, destination, budgets=budgets,
                                              methods=methods, progress=progress,
                                              cache_dir=app.workspace / "embedding-cache" / backend)
                    return self.respond(app.submit("experiment", experiment), 202)
                if self.path == "/api/export-alignment":
                    name = body["experiment"]
                    app.database(name)
                    return self.respond(export_alignment(app.workspace / "experiments" / name,
                                                         app.workspace / "bundles" / name))
                return self.respond({"error": "Not found"}, 404)
            except (ValueError, KeyError, OSError, TypeError) as e:
                return self.respond({"error": str(e)}, 400)

    return Handler


def serve(workspace, backend="clap", config=None, device="cpu", offline=False, port=8765):
    app = Application(workspace, backend, config, device, offline)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app))
    print(f"Sonic Bridge: http://127.0.0.1:{server.server_port} | backend={backend}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        app.pool.shutdown(wait=True)
