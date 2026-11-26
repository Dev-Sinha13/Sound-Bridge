# Sonic Bridge

Local natural-language audio search and reproducible experiments connecting frozen audio and text encoders.

## What is built

- CLAP baseline with real local inference, overlapping windows, timestamped playback, incremental content-hash indexing, and temporal duplicate suppression.
- Frozen Audio-JEPA + MPNet research backend, with ridge regression and equal-dimension orthogonal Procrustes. MPNet is a sentence embedding model, **not a text JEPA**. This first implementation tests practical frozen-encoder alignment, not the full LeCun theorem.
- Calibration curves across source-group budgets and random seeds; all captions remain with their recording; ridge regularization selected on validation data; bidirectional Recall@1/5/10 and median rank on held-out test data.
- Local browser interface: collections, search, audio playback, background jobs, and experiment tables.
- Explicit synthetic tone fixtures for exercising the complete pipeline without model weights. These are not evidence of semantic retrieval capability.

## Quick start (Windows PowerShell)

```powershell
cd C:\Users\devsi\personal_projects\sonic_bridge
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,models]"
.\.venv\Scripts\python.exe -m sonic_bridge serve
```

Open **http://127.0.0.1:8765**. Enter an audio folder, name a collection, index it, then search. Initial model downloads need internet; subsequent runs can use `serve --offline`. Inference is local. CPU is the default; add `--device cuda` with a suitable PyTorch/CUDA installation. Model and dataset licenses apply separately.

After installation, `./start.ps1` launches CLAP; `./start.ps1 -Backend demo` launches the fixture mode. `requirements-lock.txt` captures the development environment's exact package versions (Windows, Python 3.13); use the ranges in `pyproject.toml` for other platforms, and verify matching PyTorch audio/vision compatibility when changing versions.

The app listens only on loopback. It does not upload audio, create transcripts, or call an inference API. Hugging Face model downloads occur on first use; model cache defaults to `.cache/huggingface` inside this project. Set `HF_HOME` to choose another cache. Indexes contain local paths and embeddings and should be treated as sensitive.

### Lightweight synthetic demo

Only the base dependencies are needed (`pip install -e ".[dev]"`).

```powershell
.\.venv\Scripts\python.exe -m sonic_bridge serve --backend demo
```

Click **Generate synthetic tone fixtures**, then **Index collection**. Search for `low tone`, `mid tone`, or `high tone`. In Alignment lab the fixture's manifest and budgets are filled automatically. Demo search deliberately uses spectral bands and a tiny text vocabulary, not learned semantic embeddings.

## CLI

```powershell
# Real CLAP baseline
.\.venv\Scripts\python.exe -m sonic_bridge index C:\Recordings --database data/indexes/library.sqlite
.\.venv\Scripts\python.exe -m sonic_bridge search "metal scraping" --database data/indexes/library.sqlite

# Offline pipeline smoke test
.\.venv\Scripts\python.exe -m sonic_bridge demo
.\.venv\Scripts\python.exe -m sonic_bridge index data/demo/audio --backend demo --database data/indexes/tones.sqlite
.\.venv\Scripts\python.exe -m sonic_bridge search "high tone" --backend demo --database data/indexes/tones.sqlite
.\.venv\Scripts\python.exe -m sonic_bridge experiment data/demo/manifest.jsonl --backend demo --budgets 6 12 18 --output artifacts/demo
```

Keep each database tied to one encoder, model revision, root folder, and window policy. Changed settings need a new database. Changed audio is re-embedded; unchanged files reuse indexed vectors. A failed indexing job rolls back its changes. Search scores are cosine similarities, not probabilities. Short overlapping hits are suppressed; this first version does not reason about event order or separate simultaneous sources.

## Real-data calibration and evaluation

Create a JSONL manifest, one recording per row. Relative audio paths resolve against the manifest directory.

```json
{"id":"rec001","audio":"audio/rec001.wav","group":"original-session-01","split":"train","captions":["A dog barks beside passing traffic.","Traffic hums behind a barking dog."]}
```

Use non-empty `train`, `val`, and `test` splits. Put recordings from one original source/session in the same `group` and split. Exact file duplicates, reused paths, and groups across splits are rejected. Re-encoded duplicates require manual grouping: file hashes cannot detect them. All captions stay with their recording. Budgets count **source groups**, and the report also records the actual number of clip rows. Targets are averaged across a recording's captions before fitting. Evaluation treats all its captions as relevant; ties receive conservative ranks.

Download [Clotho v2.1](https://zenodo.org/records/4783391) and unpack each audio split yourself, then import its CSVs:

```powershell
.\.venv\Scripts\python.exe -m sonic_bridge import-clotho C:\Clotho\clotho_captions_development.csv --audio-dir C:\Clotho\development --split train
.\.venv\Scripts\python.exe -m sonic_bridge import-clotho C:\Clotho\clotho_captions_validation.csv --audio-dir C:\Clotho\validation --split val
.\.venv\Scripts\python.exe -m sonic_bridge import-clotho C:\Clotho\clotho_captions_evaluation.csv --audio-dir C:\Clotho\evaluation --split test
.\.venv\Scripts\python.exe -m sonic_bridge experiment data/clotho.jsonl --backend clap --output artifacts/clap
```

Download/extraction is intentionally explicit; the tool does not silently fetch a large dataset. Check upstream model pretraining overlap before treating benchmark retrieval as generalization. Add a separate collection of target-user recordings as an external test set. Reports do not claim statistical significance or independent replication of published work.

### Audio-JEPA experiment

1. Use the supplied `configs/jepa.json`, which pins Audio-JEPA and MPNet revisions. The Audio-JEPA loading/configuration/frontend source was inspected before enabling its custom-code loader; this is not a comprehensive security audit.
2. For different checkpoints, start from `configs/jepa.example.json`, review the repository's custom Python code, pin `audio_revision` to its full 40-character commit, and set `trust_remote_code` to `true`. Pin the text revision as well for reproducibility.
3. Run:

```powershell
.\.venv\Scripts\python.exe -m sonic_bridge experiment data/clotho.jsonl --backend jepa --config configs/jepa.json --budgets 100 500 1000 3000 --output artifacts/jepa
```

Each run saves `.npz` weights, JSON provenance, training IDs, validation results, and test metrics. Embedding caches are keyed by audio content, captions, extraction policy, and resolved model identities. Ridge can bridge unequal dimensions; this version skips Procrustes when dimensions differ rather than introducing an implicit projection. Both retrieval sides use the same centered text space after mapping.

Select a run **using validation scores**, update `alignment` in the config to its `.npz` path (relative to the config), then:

```powershell
.\.venv\Scripts\python.exe -m sonic_bridge serve --backend aligned --config configs/jepa.json
```

The app verifies that the map's encoder identities match the active models. Raw `jepa` is intentionally disallowed for search because its unimodal features have no calibrated common space.

## Validation

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests
```

Tests cover known-rotation recovery, ridge solutions in both sample regimes, model artifact roundtrips, audio resampling/window boundaries, incremental indexing and rollback, encoder mismatch rejection, split leakage, multiple-caption relevance, tied ranks, the complete synthetic experiment, and local HTTP access controls. Real pretrained inference and dataset results require separate validation; synthetic tests do not establish model quality.

`scripts/smoke_clap.py` and `scripts/smoke_jepa.py` exercise actual pretrained forward passes and save inference checks under `artifacts/validation`. They may download model weights. These checks verify runtime integration, not retrieval accuracy on environmental audio. The bundled tone fixtures contain repeated semantic categories, so low clip-level Recall@1 is expected even when the tone category is correctly retrieved; do not use these scores to choose a research architecture.

## Research scope and next steps

The first release implements retrieval and two linear bridges. Event-preserving JEPA pretraining, CCA, nonlinear adapters, active labeling, distillation, temporal-order reranking, and speaker-adversarial training are future experiments, not implemented features. For a privacy evaluation, train independent speaker probes on consented audit data and report the utility/leakage tradeoff; local embeddings are not anonymous by construction.

Research references: [conditional JEPA alignment](https://arxiv.org/abs/2610.39302), [LightAlign and shared-subspace failures](https://arxiv.org/abs/2611.20000), [Audio-JEPA](https://huggingface.co/ltuncay/Audio-JEPA-base), [CLAP](https://github.com/LAION-AI/CLAP), [MPNet embedding model](https://huggingface.co/sentence-transformers/all-mpnet-base-v2), [tinyCLAP](https://github.com/fpaissan/tinyCLAP), [speaker leakage audit](https://arxiv.org/abs/2611.31043).
