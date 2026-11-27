# Validation record

Validated locally on Windows with Python 3.13 on November 26, 2026. The expanded local regression suite passes **33 tests** including the optional Torch projection/pooling checks; Ruff, JavaScript syntax checking, and dependency consistency checks pass. Hosted core-only jobs skip the two tests requiring Torch.

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

The retrieval improvement milestone passed the four-job hosted matrix in [run 45880332109](https://github.com/Dev-Sinha13/Sound-Bridge/actions/runs/45880332109), and judgment/provenance additions passed in [run 45881437154](https://github.com/Dev-Sinha13/Sound-Bridge/actions/runs/45881437154). New regression coverage includes a short event overwhelmed in a long window, multi-scale retrieval, batch-cache reuse after cancellation, empty-folder refresh, absent-query cutoff selection without test leakage, source-separated query judgments, timestamp IoU, CCA/RFF artifact roundtrips, all-caption contrastive fitting, masked frame pooling, and local judgment saving.

Real HTTP integration (`artifacts/validation/http-v2.json`) passed paper-crumpling search at a 2-second interval, audio/WAV decoding, 160-bin waveform delivery, segment similarity, file filtering, judgment persistence, and watched refresh. FineLAP completed real inference, the query suite and the matched 200-recording Clotho retrieval evaluation, then loaded successfully in offline mode with both pinned snapshots. Browser visual verification of the new controls could not be completed: Computer Use stopped because it could not establish the Windows browser URL reliably. The earlier browser release check above applies to the original interface.

The running app's offline model-switch API also loaded FineLAP, returned paper first for `paper crumpling` against its multi-scale example index, and restored CLAP successfully (`artifacts/validation/model-switch-v2.json`).

## Workspace redesign

The subsequent workspace redesign was verified in the live in-app browser at 1440×1000 and 390×844. Search returned the correct paper recording first; all ten returned audio players loaded finite durations without media errors. File filters reached the new empty state, and clearing them restored results. Collections, compact experiment summaries, the model dialog, Escape dismissal and arrow-key tab navigation were checked. The phone navigation and page had no horizontal overflow. Browser console warnings/errors were empty during this check. Temporary viewport overrides were reset afterward.

The seven HTTP regression tests pass, along with JavaScript syntax checking, Ruff and `git diff --check`. Preview screenshots remain local in `artifacts/validation/ui-redesign-desktop.jpg` and `ui-redesign-mobile.jpg`. This browser verification supersedes the earlier visual-verification limitation for the updated controls.

These checks establish runtime integration and numerical behavior. Completed real-data measurements are documented in [results.md](results.md). They do not establish total training-data efficiency relative to CLAP, anonymity, edge-device performance, or the theorem's assumptions. See the README for dataset manifests and reproducible experiment commands.
