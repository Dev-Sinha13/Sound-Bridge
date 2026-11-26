# Retrieval improvements and research decisions

Research checked November 26, 2026. These are implemented experiments, not assumed accuracy gains.

## Retrieval and evaluation

[FineLAP](https://arxiv.org/abs/2604.01155) jointly addresses clip- and frame-level audio/text alignment. Its [official implementation](https://github.com/xiquan-li/FineLAP) and [checkpoint](https://huggingface.co/AndreasXi/FineLAP) support global retrieval and dense features. This project adds an optional global-embedding adapter; it does not reproduce FineLAP training or claim to implement its dense localization head. The adapter follows the published 16 kHz Kaldi filterbank frontend. Outer model and nested RoBERTa dependencies are pinned. Source reviewed: model construction, feature preprocessing, import dependencies and file/network operations at checkpoint `b419aa22947d29907a5567f21b81bf3b39a40449`. This is a scoped source review, not a security audit.

[Omni-Embed-Audio](https://arxiv.org/abs/2604.18360) evaluates questions, commands, tags, paraphrases and exclusion queries, motivating evaluation beyond caption wording. The included `query-development.json` is an author-created, caption/metadata-derived 18-query development suite over the 16-recording example gallery. It is not independently listener-verified and is not a fresh external test set. Its labels may be incomplete, especially for background sounds. Six absent-sound judgments exercise rejection. Related sound sources and paraphrases stay in one split. Do not interpret these small-suite percentages as production accuracy.

`evaluate-queries` records per-query matches and failure categories: wrong recording, rejected present sound, accepted absent sound, and timestamp mismatch when interval judgments exist. It reports recall, reciprocal rank, false accepts, accepted correct matches and query time. Similarity cutoffs are fitted only on validation judgments, separately for each model/index; a cosine score is not a probability. Candidate coverage is limited to the first 100 suppressed windows. The validation false-accept target is an empirical constraint, not a statistical guarantee.

Multi-scale indexing combines the original 10-second/5-second-hop windows with optional 2- and 5-second windows at 50% overlap. It preserves brief events at higher storage/inference cost. A localized synthetic regression test demonstrates why this can help, but synthetic tone localization does not establish natural-event accuracy. Ten-second-only indexing remains the default pending stronger evidence. Temporal evaluation requires listener-labeled start/end judgments; the metadata-derived suite has none.

## Frozen-encoder bridge experiments

The original ridge and Procrustes baselines remain available. Added candidates:

- Regularized CCA: whiten the two centered training covariances with shrinkage, take their cross-covariance SVD, and project both modalities into up to 64 shared components, weighted by the square root of canonical correlation. Covariance shrinkage is selected on validation, including when sample count is below embedding dimensionality. See the [CCA documentation](https://scikit-learn.org/stable/modules/cross_decomposition.html#canonical-correlation-analysis).
- Nonlinear random Fourier features followed by ridge: a fixed 256-feature Gaussian-kernel approximation, with seeded parameters and validation-selected ridge strength. This provides a small nonlinear baseline without adding a training framework. See [Random Features for Large-Scale Kernel Machines](https://people.eecs.berkeley.edu/~brecht/papers/07.rah.rec.nips.pdf).
- Contrastive projection: a 128-unit tanh audio network trained for 40 epochs, with all captions individually positive and other recordings in the batch as negatives. Symmetric audio/text loss, temperature 0.07, AdamW, fixed training seed, frozen upstream encoders. This tests information lost by caption averaging without changing text/audio pretraining. The [CLAP paper](https://arxiv.org/abs/2211.06687) motivates contrastive audio/language alignment; this small calibration head is not CLAP pretraining.

All candidates use the existing source/hash-disjoint manifest, fixed budgets and validation selection. Test scores are descriptive. The previously examined Clotho test set is reused for comparability, so these experiments are exploratory extensions, not a new confirmatory study. Existing embedding caches are reused; extraction times cannot be compared as cold runs. Artifact serialization supports both projections and nonlinear parameters while retaining compatibility with existing ridge bundles.

[M2D-CLAP](https://github.com/nttcslab/m2d) is another promising candidate. Its loader/checkpoint stack needs a separate integration and matched evaluation; no M2D performance claim is made here. A much larger multimodal LLM such as OEA would also need explicit local resource measurements before adoption. FineLAP is the first additional implemented native model.

The pinned Audio-JEPA interface exposes scene and final timestamp embeddings, not arbitrary intermediate layers. Config `audio_pooling` now supports `scene` (unchanged default), `frame_mean`, and `frame_max`, with padding masked out. Non-default pooling changes model identity, invalidates old embedding caches/indexes, and survives bundle export. The alternatives have runtime/masking checks but no retrieval-quality evaluation yet; they are not recommended defaults.

## Library behavior

Each indexing batch is saved atomically to a content/model/interval-keyed embedding cache. Cancellation rolls back the searchable database, while a retry reuses completed batches. Unchanged files remain skipped; deleted files are removed on refresh, including an emptied existing collection. Caches contain sensitive derived data and have no automatic retention limit.

Local judgment buttons save correct sound, wrong sound, wrong timestamp, and absent-sound labels under `data/judgments/`, together with the collection content hash and model/window provenance. Labels do not automatically retrain models or change rankings. Users must judge the collection as a whole before marking a sound absent.

The UI includes file/path filtering, an optional similarity cutoff, one hit per recording, lazy waveform previews, and segment-to-segment similarity search. Folder watching checks size/mtime every ten seconds while the browser page polls. It pauses when a different encoder is selected and stops when the server exits. CLI `index --watch 10` continuously rechecks file hashes without a browser. Watching and search share the existing serialized job queue. A file must have a stable write before successful embedding; failed jobs retain the previous index and can be retried. Watching configuration is not persisted across server restarts.

## Next evidence needed

Collect listener-verified queries and event intervals, disjoint source groups, and genuinely absent sounds in a previously unused collection. Freeze model/window/cutoff choices on development data, then evaluate once on that collection. Profile cold and warm indexing, peak memory and p50/p95 search latency on target hardware before choosing a larger vector index, quantization, or a different default model. Training good-view audio/text JEPAs remains a separate research direction, not something these post-hoc projections establish.
