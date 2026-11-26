# Real-audio evaluation

These are completed local evaluations on November 26, 2026, using public environmental recordings and actual pretrained models. Synthetic tone fixtures are excluded from this table.

## CLAP retrieval reference

Checkpoint: `laion/clap-htsat-unfused`, revision `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a`.

The fixed Clotho v2.1 pilot selects 1,000 training, 200 validation, and 200 evaluation recordings, one per original source. Sources shared across official splits are excluded from earlier splits; evaluation is preserved before deterministic source-priority subsetting with seed 2026. Native CLAP requires no calibration and encodes only the evaluation gallery and its captions.

| Direction | Recall@1 | Recall@5 | Recall@10 | Median rank |
|---|---:|---:|---:|---:|
| Text → audio | 35.0% | 67.8% | 80.6% | 3 |
| Audio → text | 45.5% | 74.5% | 87.0% | 2 |

Text-to-audio has 1,000 caption queries against 200 recordings. Audio-to-text has 200 recording queries against 1,000 captions, with all five captions of a recording relevant. The 95% bootstrap interval for text-to-audio Recall@1 is 30.2–39.6%; for audio-to-text it is 38.5–52.5%. Bootstrap resampling keeps a recording's captions together and holds the candidate gallery fixed. Chance text-to-audio Recall@1 is 0.5%.

Audio extraction averages normalized embeddings of 10-second windows with a 5-second hop, then normalizes the result. Text features are normalized. This is a source-clean pilot using that explicit windowing policy, not the official full-gallery benchmark or a reproduction of a published CLAP score.

## ESC-50 reference

The same CLAP checkpoint scored **89.75% accuracy (359/400)** on ESC-50 fold 5 using three fixed class-prompt templates and no fitting on ESC-50. The source-group bootstrap interval is **85.93–92.95%**. This initial reference uses the complete fold. A matched comparison with a calibrated bridge additionally excludes any calibration source matches from both methods.

The dataset was downloaded from the official ESC-50 repository at `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`. This is classification among 50 class descriptions, a different task from exact-recording caption retrieval.

## Interpretation and reproducibility

The baseline supports useful local semantic search. These results do not establish superiority or sample-efficiency gains for a JEPA bridge. The frozen Audio-JEPA/MPNet calibration comparison is recorded separately when its runs complete; test scores never select its bridge.

Upstream encoder pretraining overlap is not excluded. Calibration counts exclude pretraining data. Neither dataset establishes device efficiency, clinical suitability, anonymity, or robustness on a user's own collection.

Reproduce using `scripts/benchmark_clotho.py --offline --parallel --limits 1000 200 200 --budgets 100 500 1000 --output artifacts/clotho-pilot`. See the README for verified dataset preparation and the separate ESC-50 commands. Machine-readable reports, per-seed maps, provenance and confidence intervals remain local under `artifacts/`; recordings and embeddings are not committed.
