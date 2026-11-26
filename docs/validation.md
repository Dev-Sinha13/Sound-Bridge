# Validation record

Validated locally on Windows with Python 3.13 on November 26, 2026. The complete local regression suite passes **23 tests**; Ruff, JavaScript syntax checking, and dependency consistency checks pass.

- Regression tests: known orthogonal transformation recovery, ridge primal/dual agreement, audio preprocessing, index caching and rollback, multi-caption retrieval and conservative ties, source/content leakage rejection, synthetic experiment execution, dataset-level synthetic labeling, and the HTTP API.
- Ruff: `python -m ruff check src tests scripts`.
- Dependency consistency: `python -m pip check` passed.
- Browser: synthetic high-tone search returned timestamped high-tone clips; the alignment lab displayed the completed three-budget, three-seed experiment.
- Real CLAP: `laion/clap-htsat-unfused` at `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a` produced finite `(2, 512)` audio and text embeddings on two generated tones. The local index/search pipeline ranked the generated high tone above the low tone for a high-tone query.
- Real Audio-JEPA: `ltuncay/Audio-JEPA-base` at `e5d4bcf5a3ce2ea48fb425869b5a1b68bf1a7017` and MPNet at `e8c3b32edf5434bc2275fc9bab85f82640a19130` produced finite `(2, 768)` audio and text embeddings.
- Saved JEPA alignment: an exported, integrity-checked bridge loaded successfully, indexed the generated low/high tones, and ranked the high tone first for a high-tone text query. This remains a synthetic integration check.
- Real selected bridge: the Clotho validation-selected ridge bundle passed its artifact hash check, indexed 16 real recordings into 61 timestamped windows, and returned the paper-crumpling recording first for `paper crumpling`. This small-library integration check is separate from the held-out benchmark scores.
- GitHub Actions: the initial Windows/Linux and Python 3.11/3.13 matrix completed successfully in [run 45768174453](https://github.com/Dev-Sinha13/Sound-Bridge/actions/runs/45768174453).
- The expanded Windows/Linux test matrix also passed at commit `77e0761` in [run 45871420340](https://github.com/Dev-Sinha13/Sound-Bridge/actions/runs/45871420340).
- Browser release workflow: real paper-crumpling retrieval, a 10-second audio segment loaded and played to completion without a media error, exported bridge selection, indexing and searching in the new space, and restoration of the CLAP baseline. The repeated warm search completed in 0.27 seconds on this machine; this is one functional check, not a latency benchmark.
- Packaging: a `sonic_bridge-0.1.0-py3-none-any.whl` builds successfully and contains all three browser assets. Model weights and datasets are separate downloads.

Machine-generated inference records are stored locally in `artifacts/validation/`. Generated weights, audio, caches and datasets are deliberately excluded from Git.

These checks establish runtime integration and numerical behavior. Completed real-data measurements are documented in [results.md](results.md). They do not establish total training-data efficiency relative to CLAP, anonymity, edge-device performance, or the theorem's assumptions. See the README for dataset manifests and reproducible experiment commands.
