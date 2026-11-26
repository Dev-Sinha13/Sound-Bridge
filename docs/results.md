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

After selecting the JEPA bridge on Clotho validation data, the matched transfer check excluded three ESC-50 clips sharing Freesound source IDs with its calibration recordings. Both methods were evaluated on the same remaining 397 clips, using the same fixed three-prompt ensemble:

| Method | Correct / evaluated | Accuracy | 95% source-group bootstrap interval |
|---|---:|---:|---:|
| CLAP | 357 / 397 | 89.92% | 86.00–93.27% |
| Validation-selected JEPA + ridge | 54 / 397 | 13.60% | 10.05–17.38% |

Random selection among 50 classes has 2% accuracy. The bridge carries transferable sound-category information but remains much weaker than CLAP in this experiment. ESC-50 labels and scores did not fit the bridge, choose its hyperparameters, or change its prompts.

## Frozen Audio-JEPA → MPNet comparison

Audio-JEPA revision: `e5d4bcf5a3ce2ea48fb425869b5a1b68bf1a7017`. MPNet revision: `e8c3b32edf5434bc2275fc9bab85f82640a19130`. Both encoders stay frozen. Each recording's five normalized caption embeddings are averaged and normalized to form its calibration target.

All 18 method/budget/seed runs completed. The highest validation text-to-audio Recall@1 selected **ridge regression, 1,000 source recordings, seed 7, alpha 1.0**. Validation Recall@1 was 11.8%. The exported bundle is `artifacts/clotho-pilot/selected-bundle/config.json` locally.

| Method at the largest calibration budget | Additional calibration recordings | Text → audio R@1 | R@5 | R@10 | Audio → text R@1 |
|---|---:|---:|---:|---:|---:|
| Native CLAP reference | 0 | 35.0% | 67.8% | 80.6% | 45.5% |
| JEPA + Procrustes | 1,000 | 7.4% | 25.3% | 39.2% | 5.0% |
| JEPA + ridge, validation-selected | 1,000 | 7.9% | 30.4% | 43.4% | 6.0% |

The selected ridge map's text-to-audio Recall@1 interval is **5.60–10.70%**, with median rank 14. Audio-to-text Recall@1 has an interval of **3.0–9.5%**, with median rank 22. The gallery and queries are the same as for CLAP; chance text-to-audio Recall@1 is 0.5%.

Mean test text-to-audio Recall@1 across the three calibration-subset seeds:

| Calibration source recordings | Procrustes | Ridge |
|---|---:|---:|
| 100 | 4.80% | 4.47% |
| 500 | 6.20% | 7.17% |
| 1,000 | 7.40% | 7.90% |

Seeds change the selected calibration subsets. At 1,000, every run uses the complete pilot training set, so the deterministic fitted maps and scores agree across seeds. These means are descriptive; query-bootstrap intervals and subset-seed variability are different uncertainties.

## Interpretation and reproducibility

CLAP remains the stronger default for this application. A small linear bridge recovers useful cross-modal signal from these frozen unimodal encoders, but its measured retrieval quality is substantially lower than CLAP's in this pilot. No superiority or total-training-data efficiency gain is established. Test scores never selected the bridge.

This experiment uses an existing masked Audio-JEPA and an MPNet sentence encoder. Testing the paper's stronger good-view JEPA training recipe requires different encoder training; these results characterize this implemented frozen-encoder combination.

Upstream encoder pretraining overlap is not excluded. Calibration counts exclude pretraining data. Neither dataset establishes device efficiency, clinical suitability, anonymity, or robustness on a user's own collection. Reported runtime includes resumed embedding caches and should not be compared as a full training or inference cost measurement.

Reproduce using `scripts/benchmark_clotho.py --offline --parallel --limits 1000 200 200 --budgets 100 500 1000 --output artifacts/clotho-pilot`. See the README for verified dataset preparation and the separate ESC-50 commands. Machine-readable reports, per-seed maps, provenance and confidence intervals remain local under `artifacts/`; recordings and embeddings are not committed.
