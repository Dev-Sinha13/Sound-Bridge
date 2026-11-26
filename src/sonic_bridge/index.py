from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from .audio import discover, windows
from .util import digest, file_digest, normalize, save_npz


def connect(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.executescript("""
        CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, sha TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS segments (
            id INTEGER PRIMARY KEY, path TEXT NOT NULL, start REAL, end REAL, vector BLOB NOT NULL);
    """)
    return db


def build_index(folder, database, encoder, seconds=10, hop=5, progress=lambda msg: None,
                scales=None, cache_dir=None):
    if not 0 < hop <= seconds <= 60:
        raise ValueError("Require 0 < hop <= window <= 60 seconds")
    files = discover(Path(folder))
    if not files and not Path(database).exists():
        raise ValueError("No supported audio files found (WAV, FLAC, OGG, MP3)")
    settings = {"encoder": encoder.identity, "window": seconds, "hop": hop,
                "root": str(Path(folder).resolve()), "schema": 1}
    scales = sorted({float(s) for s in (scales or [])})
    if any(not 0 < s <= 60 for s in scales):
        raise ValueError("Extra window scales must be between 0 and 60 seconds")
    if scales:
        settings["scales"] = scales
    cache_dir = Path(cache_dir) if cache_dir else Path(database).parent / "embedding-cache"
    db = connect(Path(database))
    try:
        row = db.execute("SELECT value FROM meta WHERE key='settings'").fetchone()
        if row and json.loads(row[0]) != settings:
            raise ValueError("This index uses different settings. Choose a new index name.")
        existing = dict(db.execute("SELECT path, sha FROM files"))
        changed = 0
        with db:
            for number, path in enumerate(files):
                progress(f"Checking {number + 1}/{len(files)}: {path.name}")
                sha = file_digest(path)
                if existing.get(str(path)) == sha:
                    continue
                changed += 1
                db.execute("DELETE FROM segments WHERE path=?", (str(path),))
                seen = set()
                for width, stride in [(seconds, hop), *[(s, s / 2) for s in scales if s != seconds]]:
                    batch = []
                    for window in windows(path, encoder.sample_rate, width, stride):
                        interval = (window.start, window.end)
                        if interval in seen:
                            continue
                        seen.add(interval)
                        batch.append(window)
                        if len(batch) == 4:
                            progress(f"Embedding {path.name}: {window.end:.1f}s / {width:g}s windows")
                            _insert_batch(db, path, batch, encoder, cache_dir, sha)
                            batch = []
                    if batch:
                        progress(f"Embedding {path.name}: final {width:g}s windows")
                        _insert_batch(db, path, batch, encoder, cache_dir, sha)
                db.execute("INSERT OR REPLACE INTO files VALUES (?,?)", (str(path), sha))
            live = {str(p) for p in files}
            for stale in set(existing) - live:
                db.execute("DELETE FROM segments WHERE path=?", (stale,))
                db.execute("DELETE FROM files WHERE path=?", (stale,))
            progress("Saving index")
            db.execute("INSERT OR REPLACE INTO meta VALUES ('settings',?)", (json.dumps(settings),))
        return {"files": len(files), "changed": changed,
                "segments": db.execute("SELECT COUNT(*) FROM segments").fetchone()[0]}
    finally:
        db.close()


def _insert_batch(db, path, batch, encoder, cache_dir, sha):
    key = digest({"sha": sha, "encoder": encoder.identity,
                  "intervals": [(w.start, w.end) for w in batch], "policy": "index-v1"})
    cache = cache_dir / f"{key}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            vectors = saved["vectors"]
    else:
        vectors = normalize(encoder.encode_audio([w.samples for w in batch]))
    if vectors.ndim != 2 or len(vectors) != len(batch) or not np.isfinite(vectors).all():
        raise ValueError("Encoder returned an invalid embedding batch")
    if not cache.exists():
        save_npz(cache, vectors=vectors)
    for window, vector in zip(batch, vectors):
        db.execute("INSERT INTO segments(path,start,end,vector) VALUES (?,?,?,?)",
                   (str(path), window.start, window.end, vector.astype(np.float32).tobytes()))


def index_info(database):
    path = Path(database)
    if not path.exists():
        raise ValueError("Index does not exist")
    db = connect(path)
    try:
        row = db.execute("SELECT value FROM meta WHERE key='settings'").fetchone()
        return {"name": path.stem, "settings": json.loads(row[0]) if row else None,
                "files": db.execute("SELECT COUNT(*) FROM files").fetchone()[0],
                "segments": db.execute("SELECT COUNT(*) FROM segments").fetchone()[0]}
    finally:
        db.close()


def search(database, encoder, query, limit=10, *, min_score=None, name_filter="", unique_files=False,
           similar_to=None):
    if not str(query).strip() or not 1 <= limit <= 100:
        raise ValueError("Enter a query and a result limit between 1 and 100")
    if min_score is not None and not -1 <= float(min_score) <= 1:
        raise ValueError("Minimum similarity must be between -1 and 1")
    if not Path(database).exists():
        raise ValueError("Index does not exist")
    db = connect(Path(database))
    try:
        row = db.execute("SELECT value FROM meta WHERE key='settings'").fetchone()
        if not row or json.loads(row[0])["encoder"] != encoder.identity:
            raise ValueError("Query encoder does not match index; use the same backend, config, and model revision")
        rows = db.execute("SELECT * FROM segments ORDER BY id").fetchall()
    finally:
        db.close()
    if not rows:
        return []
    matrix = np.stack([np.frombuffer(row["vector"], dtype=np.float32) for row in rows])
    if similar_to is None:
        q = normalize(encoder.encode_text([query]))[0]
    else:
        selected = next((i for i, row in enumerate(rows) if row["id"] == int(similar_to)), None)
        if selected is None:
            raise ValueError("Unknown source segment")
        q = normalize(matrix[selected])
    scores = matrix @ q
    results = []
    # Greedy temporal suppression prevents overlapping windows from crowding the list.
    for i in np.argsort(-scores, kind="stable"):
        row = rows[int(i)]
        if row["id"] == similar_to or name_filter.casefold() not in row["path"].casefold():
            continue
        if min_score is not None and scores[i] < min_score:
            continue
        if unique_files and any(r["path"] == row["path"] for r in results):
            continue
        if any(r["path"] == row["path"] and
               min(r["end"], row["end"]) - max(r["start"], row["start"]) > 0 for r in results):
            continue
        results.append({"id": row["id"], "path": row["path"], "name": Path(row["path"]).name,
                        "start": row["start"], "end": row["end"], "score": float(scores[i])})
        if len(results) == limit:
            break
    return results


def segment(database, segment_id):
    db = connect(Path(database))
    try:
        row = db.execute("SELECT path,start,end FROM segments WHERE id=?", (segment_id,)).fetchone()
        if row is None:
            raise ValueError("Unknown audio segment")
        return dict(row)
    finally:
        db.close()
