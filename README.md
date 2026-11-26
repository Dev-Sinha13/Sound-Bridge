# Sonic Bridge

Source repository: [Dev-Sinha13/Sound-Bridge](https://github.com/Dev-Sinha13/Sound-Bridge). Development changes are pushed as tested milestones with conventional commit messages; see [AGENTS.md](AGENTS.md).

Local natural-language audio search and reproducible experiments connecting frozen audio and text encoders.

Completed real-audio measurements and their evaluation limits are recorded in [the results report](docs/results.md).

Research-backed retrieval and bridge additions are explained in [improvements.md](docs/improvements.md), including evaluation limits and primary sources.

## What is built

- CLAP baseline with real local inference, overlapping windows, timestamped playback, incremental content-hash indexing, and temporal duplicate suppression.
- Frozen Audio-JEPA + MPNet research backend, with ridge regression and equal-dimension orthogonal Procrustes. MPNet is a sentence embedding model, **not a text JEPA**. This first implementation tests practical frozen-encoder alignment, not the full LeCun theorem.
- Optional pinned FineLAP native retrieval; CCA, nonlinear random-feature, and all-caption contrastive bridge comparisons. These candidates are evaluated before changing defaults.
- Calibration curves across source-group budgets and random seeds; all captions remain with their recording; ridge regularization selected on validation data; bidirectional Recall@1/5/10 and median rank on held-out test data.
- Local browser interface: collections, search, audio playback, background jobs, and experiment tables.
- Multi-scale windows, resumable indexing, folder watching, waveform previews, filters, similarity search, and local relevance judgments. Explicit query evaluation measures wrong recordings, timestamp errors, and absent-sound false accepts.
- Explicit synthetic tone fixtures for exercising the complete pipeline without model weights. These are not evidence of semantic retrieval capability.

## Quick start (Windows PowerShell)

```powershell
git clone https://github.com/Dev-Sinha13/Sound-Bridge.git
cd Sound-Bridge
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,models]"
.\.venv\Scripts\python.exe -m sonic_bridge serve
```

Open **http://127.0.0.1:8765**. Enter an audio folder, name a collection, index it, then search. Initial model downloads need internet; subsequent runs can use `serve --offline`. Inference is local. CPU is the default; add `--device cuda` with a suitable PyTorch/CUDA installation. Model and dataset licenses apply separately.

After installation, `./start.ps1` launches CLAP; `./start.ps1 -Backend demo` launches the fixture mode. `requirements-lock.txt` captures the development environment's exact package versions (Windows, Python 3.13); use the ranges in `pyproject.toml` for other platforms, and verify matching PyTorch audio/vision compatibility when changing versions.

The app listens only on loopback. It does not upload audio, create transcripts, or call an inference API. Hugging Face model downloads occur on first use; model cache defaults to `.cache/huggingface` inside this project. Set `HF_HOME` to choose another cache. Indexes contain local paths and embeddings and should be treated as sensitive.

After downloading Clotho, create a small local example collection with `python scripts/seed_example_library.py`. Its `data/example_library/credits.json` retains the original recording attribution and licenses. Index that folder as `clotho-examples` and try `paper crumpling`. The example audio and captions stay local and are excluded from Git.

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

### Finer search and a growing library

```powershell
# Preserve shorter events; a different window policy requires a new database.
python -m sonic_bridge index C:\Recordings --database data/indexes/multi.sqlite --scales 2 5
python -m sonic_bridge search "paper crumpling" --database data/indexes/multi.sqlite --unique-files --name-filter kitchen
# Recheck continuously; stop with Ctrl+C. Completed embedding batches survive retries.
python -m sonic_bridge index C:\Recordings --database data/indexes/multi.sqlite --scales 2 5 --watch 10
# Compare the optional pinned FineLAP model (downloads weights on first use).
python -m sonic_bridge serve --backend finelap --config configs/finelap.json
```

The browser offers model selection, extra window lengths, optional similarity cutoff, file/path filtering, one result per recording, waveform previews, and **Find similar sounds**. **Watch selected folder while open** refreshes changes every ten seconds while the page is open. It pauses for incompatible models; watching configuration is not saved across restarts. Multi-scale indexing costs more inference and storage, so the original ten-second policy stays the default. Embedding caches are local sensitive data and can grow with library changes.

Use the result judgment buttons to save correct sounds, wrong sounds or timestamps, and sounds absent from the entire collection. Judgments and collection/model provenance stay in `data/judgments/`; they do not automatically train a model or change its rankings. Reliable absent-sound rejection still needs a larger, listener-verified validation set.

### Query judgments and failure analysis

```powershell
python scripts/benchmark_search.py data/example_library docs/query-development.json --backend clap --offline --output artifacts/search-v2/clap
python -m sonic_bridge evaluate-queries docs/query-development.json --database artifacts/search-v2/clap/multi.sqlite --output artifacts/query-report.json --offline
```

For a real assessment, create a listener-verified suite using the JSON schema in `docs/query-development.json`. Each row needs a query, `val`/`test` split, source group, category and relevant relative filenames; `relevant: []` means absent. Optional `start`/`end` values label an event interval in seconds. Keep related source recordings and paraphrases within one split. Validation must include both present and absent sounds. The evaluator selects its rejection cutoff using validation only, then reports held-out false accepts, correct accepts, retrieval ranks, timestamp IoU and inspectable failures. A fitted cutoff is specific to that model, gallery and window policy; it is not a universal confidence score. The included authored suite is exploratory and not independently verified.

Additional bridge candidates can reuse existing embeddings:

```powershell
python -m sonic_bridge experiment artifacts/clotho-pilot/manifest.jsonl --backend jepa --config configs/jepa.json --offline --methods ridge procrustes cca rff contrastive --budgets 100 500 1000 --seeds 7 --cache-dir data/clotho/embedding-cache/jepa --output artifacts/bridge-v2
```

CCA projects both modalities into a shared space; RFF tests a small nonlinear map; the contrastive head trains against each caption individually. All upstream encoders remain frozen. Hyperparameters and exported maps are selected using validation data. The lab's comma-separated methods field exposes the same candidates.

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

### Verified official dataset download

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[research]"
.\.venv\Scripts\python.exe scripts/download_clotho.py --output data/clotho
```

This explicitly downloads about 7 GB, checks the Zenodo v2.1 MD5 values, extracts the archives, and prepares `data/clotho/manifest.jsonl` with original Freesound source groups and per-file licenses. Interrupted downloads are resumed when the server supports HTTP ranges. Captions and audio retain their upstream licenses and are excluded from Git. The source-recording URL, rather than the excerpt filename, is used to detect source overlap across splits.

The preparation now canonicalizes groups by Freesound sound ID and excludes source matches from earlier splits: development excludes validation/evaluation sources, and validation excludes evaluation sources. The provenance JSON records excluded counts. Pilot subsets select one excerpt per source using a deterministic SHA-256 priority keyed by seed and source ID. The official evaluation split is preserved before any explicit pilot subsetting. This produces a source-clean protocol, not an exact reproduction of the unmodified official benchmark.

For the native CLAP reference without fitting any adapters:

```powershell
.\.venv\Scripts\python.exe -m sonic_bridge experiment data/clotho/manifest.jsonl --backend clap --methods native --offline --output artifacts/clotho-clap --cache-dir data/embedding-cache/clap --threads 4
```

`--batch-size` controls audio-window batches and queued text extraction. Embedding caches can be shared across runs and subsets. Reports include the candidate gallery size and 95% bootstrap intervals that resample query recordings together with all their captions; the candidate gallery stays fixed. These intervals do not measure uncertainty across random training seeds.

### Audio-JEPA experiment

1. Use the supplied `configs/jepa.json`, which pins Audio-JEPA and MPNet revisions. The Audio-JEPA loading/configuration/frontend source was inspected before enabling its custom-code loader; this is not a comprehensive security audit.
2. For different checkpoints, start from `configs/jepa.example.json`, review the repository's custom Python code, pin `audio_revision` to its full 40-character commit, and set `trust_remote_code` to `true`. Pin the text revision as well for reproducibility.
3. Run:

```powershell
.\.venv\Scripts\python.exe -m sonic_bridge experiment data/clotho.jsonl --backend jepa --config configs/jepa.json --budgets 100 500 1000 3000 --output artifacts/jepa
```

Each run saves `.npz` weights, JSON provenance, training IDs, validation results, and test metrics. Embedding caches are keyed by audio content, captions, extraction policy, and resolved model identities. Ridge can bridge unequal dimensions; this version skips Procrustes when dimensions differ rather than introducing an implicit projection. Both retrieval sides use the same centered text space after mapping.

Select a run **using validation scores**, update `alignment` in the config to its `.npz` path (relative to the config), then:

Each experiment also copies the run with highest validation text-to-audio Recall@1 to `selected.npz`/`selected.json`. Ties use the first run in the declared method/budget/seed order. Test scores do not participate in this selection.

Export a self-contained alignment bundle (map, provenance, pinned encoder configuration; encoder weights are cached separately):

```powershell
.\.venv\Scripts\python.exe -m sonic_bridge export-alignment artifacts/jepa --output data/bundles/jepa
.\.venv\Scripts\python.exe -m sonic_bridge serve --backend aligned --config data/bundles/jepa/config.json --offline
```

The bundle records the map's SHA-256, which the loader verifies. The Alignment lab lets you choose CLAP for a native baseline or Audio-JEPA + MPNet for fitting, independently of the backend used for collection search. Completed JEPA experiments have an export button. Raw `--backend jepa` can open the research interface, but indexing/search is disabled until an alignment is fitted.

After exporting in the browser, **Use this alignment for search** loads it without restarting the app. Create a new collection for that embedding space; existing CLAP indexes remain separate. **Use CLAP baseline** switches back. Failed model loads preserve the previous working backend. Re-exporting an unchanged bundle is safe; a different or modified bundle is never overwritten.

Publish a completed CLI experiment into the local browser's Alignment lab:

```powershell
.\.venv\Scripts\python.exe -m sonic_bridge publish-experiment artifacts/clotho-pilot/jepa --name clotho-jepa
.\.venv\Scripts\python.exe -m sonic_bridge publish-experiment artifacts/clotho-pilot/clap --name clotho-clap
```

This copies the report and its validation-selected map into the local workspace, excluding recordings, embedding caches, and unused fitted maps.

### Reproduce the comparative benchmark

```powershell
# Official full split evaluation; CPU execution may take substantial time.
.\.venv\Scripts\python.exe scripts/benchmark_clotho.py --offline

# Explicitly labeled subset run for a shorter experiment.
.\.venv\Scripts\python.exe scripts/benchmark_clotho.py --offline --limits 1000 200 200 --budgets 100 500 1000 --output artifacts/clotho-subset
```

Both methods use the same manifest and gallery. `comparison.json` retains provenance, all seed/budget results, confidence intervals and runtime. A validation-selected JEPA bundle is exported automatically. Calibration sample counts exclude the encoders' pretraining data and therefore cannot be interpreted as total training-data savings over CLAP.

Add `--parallel` to run the independent CLAP and JEPA extraction jobs concurrently. The release evaluation uses a clearly labeled 1,000/200/200 train/validation/test pilot with calibration budgets 100/500/1,000 and three seeds; it is not the full Clotho benchmark. `--wait-for-extraction` can wait for an ongoing verified download before starting. `scripts/precompute_clotho_split.py` can cache a completed split while the other archives download, using the same deterministic subset selection as the benchmark.

### Separate fixed-prompt sound classification

```powershell
.\.venv\Scripts\python.exe scripts/download_esc50.py
.\.venv\Scripts\python.exe scripts/evaluate_esc50.py --backend clap --output artifacts/esc50-clap.json --calibration-manifest artifacts/clotho-pilot/manifest.jsonl --alignment-metadata artifacts/clotho-pilot/selected-bundle/alignment.json
.\.venv\Scripts\python.exe scripts/evaluate_esc50.py --backend aligned --config artifacts/clotho-pilot/selected-bundle/config.json --output artifacts/esc50-jepa.json --calibration-manifest artifacts/clotho-pilot/manifest.jsonl
```

This evaluates ESC-50 fold 5 with three fixed prompt templates and no fitting on ESC-50. Matching Freesound source IDs from the chosen bridge's calibration set are excluded from **both** comparisons. The default fold and prompts must be set before examining results. The audio is downloaded from a pinned official Git revision, CRC-checked, and tracked by archive SHA-256. Dataset and encoder pretraining overlap remains unknown, so this is a separate-dataset transfer check rather than proof of unseen-source or unseen-class generalization.

Long browser jobs can be cancelled at the next processing checkpoint. Completed embedding caches are retained; a cancelled index build rolls back when cancellation is observed before its commit. A running model operation must finish before cancellation is processed.

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

See [the validation record](docs/validation.md) for checked model revisions and scope. GitHub Actions runs core tests on Windows/Linux and Python 3.11/3.13 without downloading pretrained models.

## Research scope and next steps

The first release implements retrieval and two linear bridges. Event-preserving JEPA pretraining, CCA, nonlinear adapters, active labeling, distillation, temporal-order reranking, and speaker-adversarial training are future experiments, not implemented features. For a privacy evaluation, train independent speaker probes on consented audit data and report the utility/leakage tradeoff; local embeddings are not anonymous by construction.

Research references: [conditional JEPA alignment](https://arxiv.org/abs/2610.39302), [LightAlign and shared-subspace failures](https://arxiv.org/abs/2611.20000), [Audio-JEPA](https://huggingface.co/ltuncay/Audio-JEPA-base), [CLAP](https://github.com/LAION-AI/CLAP), [MPNet embedding model](https://huggingface.co/sentence-transformers/all-mpnet-base-v2), [tinyCLAP](https://github.com/fpaissan/tinyCLAP), [speaker leakage audit](https://arxiv.org/abs/2611.31043).
