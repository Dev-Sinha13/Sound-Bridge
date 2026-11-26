# Validation record

Validated locally on Windows with Python 3.13 on November 26, 2026.

- Regression tests: known orthogonal transformation recovery, ridge primal/dual agreement, audio preprocessing, index caching and rollback, multi-caption retrieval and conservative ties, source/content leakage rejection, synthetic experiment execution, dataset-level synthetic labeling, and the HTTP API.
- Ruff: `python -m ruff check src tests scripts`.
- Dependency consistency: `python -m pip check` passed.
- Browser: synthetic high-tone search returned timestamped high-tone clips; the alignment lab displayed the completed three-budget, three-seed experiment.
- Real CLAP: `laion/clap-htsat-unfused` at `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a` produced finite `(2, 512)` audio and text embeddings on two generated tones. The local index/search pipeline ranked the generated high tone above the low tone for a high-tone query.
- Real Audio-JEPA: `ltuncay/Audio-JEPA-base` at `e5d4bcf5a3ce2ea48fb425869b5a1b68bf1a7017` and MPNet at `e8c3b32edf5434bc2275fc9bab85f82640a19130` produced finite `(2, 768)` audio and text embeddings.

Machine-generated inference records are stored locally in `artifacts/validation/`. Generated weights, audio, caches and datasets are deliberately excluded from Git.

These checks establish runtime integration and numerical behavior. They do not establish environmental-sound accuracy, data efficiency relative to CLAP, anonymity, edge-device performance, or the theorem's assumptions. A real-data Clotho experiment and an independent target-domain evaluation remain necessary before making those claims. See the README for the dataset manifest and experiment commands.
