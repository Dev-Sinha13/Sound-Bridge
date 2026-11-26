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

CLAP remains substantially stronger than the frozen-encoder bridge and remains the application's default. A small linear bridge recovers useful cross-modal signal from these frozen unimodal encoders, but its measured retrieval quality is substantially lower than CLAP's in this pilot. No superiority or total-training-data efficiency gain is established. Test scores never selected the bridge.

This experiment uses an existing masked Audio-JEPA and an MPNet sentence encoder. Testing the paper's stronger good-view JEPA training recipe requires different encoder training; these results characterize this implemented frozen-encoder combination.

Upstream encoder pretraining overlap is not excluded. Calibration counts exclude pretraining data. Neither dataset establishes device efficiency, clinical suitability, anonymity, or robustness on a user's own collection. Reported runtime includes resumed embedding caches and should not be compared as a full training or inference cost measurement.

Reproduce using `scripts/benchmark_clotho.py --offline --parallel --limits 1000 200 200 --budgets 100 500 1000 --output artifacts/clotho-pilot`. See the README for verified dataset preparation and the separate ESC-50 commands. Machine-readable reports, per-seed maps, provenance and confidence intervals remain local under `artifacts/`; recordings and embeddings are not committed.

## Additional bridge methods

The same cached Audio-JEPA/MPNet features, Clotho pilot, budgets and source-group seed 7 were used for three added candidates. Regularization uses the original validation grid, and contrastive fitting uses 40 fixed epochs with each caption individually positive. No new independent test set was collected. These are exploratory comparisons on the previously examined test set.

| Method, 1,000 calibration recordings | Validation T→A R@1 | Test T→A R@1 |
|---|---:|---:|
| Original ridge | **11.8%** | 7.9% |
| Regularized CCA, 64 components | 10.6% | 8.7% |
| Nonlinear random features + ridge | 10.0% | 6.0% |
| All-caption contrastive projection | 9.8% | 8.0% |

CCA's test R@1 interval is 6.30–11.50%, overlapping the original ridge interval. CCA has the highest validation score among the three added methods, but **ridge remains the selection across original and added candidates** because its validation score is higher. The higher CCA test point estimate does not justify replacing ridge or claiming a significant improvement. The nonlinear and all-caption heads did not produce a clear gain in this run.

| Calibration recordings | CCA test R@1 | Nonlinear test R@1 | Contrastive test R@1 |
|---|---:|---:|---:|
| 100 | 4.3% | 4.3% | 3.8% |
| 500 | 7.7% | 6.2% | 6.7% |
| 1,000 | 8.7% | 6.0% | 8.0% |

Full reports and portable maps are local under `artifacts/bridge-v2/`. Non-default Audio-JEPA frame mean/max pooling is available for separate ablations but was not evaluated for retrieval quality here.

## Query wording and multi-scale search

The 18-query authored development suite has six present and three absent queries in each split, with related sources kept within a split. Judgments come from metadata/captions, not independent listening; the gallery was previously inspected. There are no real-event interval judgments. Percentages from this small suite should not be generalized.

| Model / windows | Held-out present queries ranked first | Held-out absent queries incorrectly accepted |
|---|---:|---:|
| CLAP / 10 seconds | 4 / 6 | 2 / 3 |
| CLAP / 2, 5, 10 seconds | 5 / 6 | 3 / 3 |
| FineLAP / 10 seconds | 4 / 6 | 1 / 3 |
| FineLAP / 2, 5, 10 seconds | 4 / 6 | 1 / 3 |

Each model/policy fitted its own similarity cutoff using the validation split only, where all absent queries were rejected. The held-out false accepts show that three validation negatives are insufficient for dependable rejection. Multi-scale CLAP fixed the snow-versus-gravel exclusion query in this suite but still missed the first church-bell query and accepted more absent sounds. It remains optional, and **reliable no-match detection is not solved**. Local judgment capture now supports collecting better data.

The example index grows from 61 to 541 windows for the same 16 recordings. A warm refresh skips unchanged files and reuses the completed index; those timings are not cold indexing throughput. On this Windows CPU, warm held-out query p50/p95 times were approximately 60/73 ms for coarse CLAP, 65/81 ms for multi-scale CLAP, 54/67 ms for coarse FineLAP and 66/136 ms for multi-scale FineLAP. OS process peak working sets were approximately 731–736 MiB for CLAP and 780–783 MiB for FineLAP. These are small-library functional measurements with other jobs running, not isolated hardware benchmarks or large-library scalability claims.

See [improvements.md](improvements.md) for research sources, implementation decisions and commands. Query reports and individual failures are local under `artifacts/search-v2/`.

## FineLAP on the matched Clotho gallery

FineLAP at `b419aa22947d29907a5567f21b81bf3b39a40449`, with nested RoBERTa at `e2da8e2f811d1448a5b465c236feacd80ffbac7b`, completed the same 200-recording/1,000-caption Clotho evaluation. The manifest hash matches the original CLAP report. Both use normalized, averaged 10-second windows at a 5-second hop; FineLAP's native frontend uses 16 kHz audio. Neither model is calibrated on this pilot.

| Native model | T→A R@1 | R@5 | R@10 | A→T R@1 | R@5 | R@10 |
|---|---:|---:|---:|---:|---:|---:|
| CLAP | 35.0% | 67.8% | 80.6% | 45.5% | 74.5% | 87.0% |
| FineLAP | **40.0%** | **73.0%** | **85.1%** | **46.5%** | **81.5%** | **91.0%** |

FineLAP's text-to-audio R@1 recording-bootstrap interval is 35.50–45.40%, and its audio-to-text interval is 40.00–53.51%. The paired text-to-audio difference is **+5.0 percentage points**, with a **+0.8 to +9.4 point** query-recording bootstrap interval (10,000 draws, seed 2026, fixed gallery). The paired calculation retains all captions together and compares both models on the same resampled recordings; it does not adjust for multiple comparisons.

Its higher retrieval point estimates make it a useful optional backend. This is an exploratory comparison on a previously examined test gallery; checkpoint-specific upstream training overlap has not been audited. It is not evidence of superiority on arbitrary user collections, and does not reproduce either paper's full benchmark. CLAP stays the default; deployment selection should use a separate representative validation collection.

Reproduce with `python -m sonic_bridge experiment artifacts/clotho-pilot/manifest.jsonl --backend finelap --config configs/finelap.json --offline --methods native --cache-dir data/clotho/embedding-cache/finelap --output artifacts/clotho-pilot/finelap --threads 4`. The report is local under `artifacts/clotho-pilot/finelap/` and can be published into the app's lab using `publish-experiment`.

Recompute the paired interval with `python scripts/compare_native.py artifacts/clotho-pilot/manifest.jsonl artifacts/clotho-pilot/clap/report.json artifacts/clotho-pilot/finelap/report.json --left-cache data/clotho/embedding-cache/clap --right-cache data/clotho/embedding-cache/finelap --output artifacts/clotho-pilot/native-comparison.json` after completing both native runs.
