# Model Reproducibility — SN126 GBM-35clean miner

This document describes exactly how the served model was produced so that the
pipeline can be audited and reproduced from public data. It intentionally does
not include private research experiments; everything the served model needs is
in this repository.

## Served artifact

| Property | Value |
|---|---|
| model_version | `gbm-35clean-v1` |
| feature_version | `sn126-35clean-v1` |
| artifact | `poker44/miner_model/model.joblib` |
| artifact SHA256 | `b8b7fc78586568e4480ed83a880eb2639001605215a1a6631ed6206c89a9194d` |
| model class | `sklearn.ensemble.HistGradientBoostingClassifier` |
| parameters | `max_depth=3, learning_rate=0.08, max_iter=300, l2_regularization=1.0, random_state=0` |
| scikit-learn | 1.7.2 |
| training samples | 1222 groups over release dates 2026-07-17 … 2026-07-24 |

Full metadata (including the frozen 35-name feature order and calibration
constants) is in `poker44/miner_model/model_meta.json`, attested at miner
startup via the model manifest (`neurons/miner.py`).

## Training data (public benchmark only)

Training uses ONLY the public Poker44 training benchmark:

    https://api.poker44.net/api/v1/benchmark        (no auth)
    /releases                                       release history
    /chunks?sourceDate=YYYY-MM-DD                   labeled chunk-records, paginated

One benchmark *group* (one player's hands) is one labeled sample
(`groundTruth`: bot=1 / human=0). **No validator-only hidden labels, no live
synapse payloads, and no private endpoints are used in training** — the model
trains exclusively on the labels the benchmark publishes to everyone.

## Split methodology (release-date based)

Bot families rotate per release, so random row splits leak future families into
training. All model selection therefore used *release-date* splits: train on
dates `< D`, evaluate on date `D` (expanding walk-forward). The shipped model
trains on the 8 release dates listed above.

## Preprocessing (sanitizer-equivalent)

Each benchmark hand is projected through the **official validator sanitizer**
`poker44/validator/payload_view.py:prepare_hand_for_miner` — the exact
transformation applied to live miner traffic (seat aliasing, amount bucketing,
action windowing) — followed by `dedupe_padding`
(`poker44/miner_model/feature_core/sanitizer_core.py`), which collapses the
sanitizer's single-action → 12-identical-copies padding artifact. At inference
the validator has already applied `prepare_hand_for_miner`, so the miner
applies only `dedupe_padding` — training and serving see the same view.

## The 35 features

Implemented in `poker44/miner_model/feature_core/` (production-owned, vendored
verbatim from the validated research code) and assembled in the frozen order by
`poker44/miner_model/features.py` (`PROD_FEATURE_ORDER`, asserted against
`model_meta.json` at model load):

- **base 30** (`feature_core/base30.py`): pooled action rates (fold/check/call/
  bet/raise), aggression ratio+frequency, street rates and depths, reach-street
  rates, VPIP, PFR, action entropy, per-hand entropy/dispersion, bucket-level
  bet sizing (mean/std/bands/pot-ratio/overbet), hero presence rate;
- **+2 interaction** (`feature_core/plus.py`): `pot_escalation_mean`,
  `potodds_call_fold_gap`;
- **+3 determinism** (`feature_core/plus2.py`): `betsize_street_std_mean`,
  `action_bigram_entropy`, `mean_aggressors_per_hand`.

All features aggregate over ALL seats of a chunk, use amounts at bucket/ratio
resolution only, and never use hand ids, absolute seats, outcomes/cards/timing,
or the hand count — the signals survive the live sanitizer by construction.

## Calibration

`poker44/miner_model/calibration.py` — batch percentile anchor: within each
request the 0.5 decision gate is placed at the p0-th percentile (p0 = 0.85) of
that request's raw scores; requests smaller than `min_calib_batch=8` use the
absolute anchor `global_anchor_raw` learned at train time (the p85 raw-score
quantile of the training set). Rank-invariant: AP and recall@FPR are unchanged;
only the 0.5 gate moves.

## Class-label mapping

One benchmark *group* (one player's hands) is one sample; the label is
`groundTruth[i]` for group `i`: **1 = bot, 0 = human**. Rows are assembled in a
pinned order (sorted release dates → per-date manifest record order → group
order within each record), declared in `config/model/gbm35clean_v1.json`.

## Reproducing the model

The exact recipe is pinned in [`config/model/gbm35clean_v1.json`](../config/model/gbm35clean_v1.json)
(release dates, parameters, calibration, row ordering, frozen artifact hash).
With this repository installed (`pip install -e .`, scikit-learn 1.7.2):

```bash
python scripts/model/reproduce_gbm35.py \
    --output /tmp/poker44-reproduction \
    --cache  /tmp/poker44-benchmark-cache

python scripts/model/verify_reproduction.py \
    --reproduction /tmp/poker44-reproduction
```

`reproduce_gbm35.py` downloads (or reuses from `--cache`) the declared public
releases, checksums every record, extracts the frozen 35 features through the
production code, trains with the exact parameters, and emits the training
config, release manifest, feature order, artifact + SHA256, golden float-hex
prediction fixtures and a reproducibility report — all into `--output`, never
touching the served `model.joblib`. `verify_reproduction.py` then checks,
strictest first: exact binary hash match, else strict trained-state
equivalence (identical parameters, feature order, release manifest, sample
count, bit-for-bit `global_anchor_raw`, and bit-for-bit identical predictions
from both models on real benchmark chunks and a synthetic grid).

**Verified result on the pinned data snapshot: `STRICT_EQUIVALENCE` (12/12
checks)** — the reproduced model's predictions are bit-for-bit identical to
production. The *binary* artifact hash differs for two measured, documented
reasons: (1) `HistGradientBoostingClassifier` pickles `_bin_mapper.n_threads`,
a fit-time environment recording (e.g. 12 vs 1) with no effect on predictions;
(2) joblib serialization in this environment is not round-trip byte-stable even
for an identical object. Byte-identity of `model.joblib` is therefore not a
meaningful reproduction target; the served artifact is pinned by its SHA256
and the trained *state* is what reproduction verifies.

`python -m poker44.miner_model.smoke_test` additionally proves the served
pipeline (feature contract, calibration paths, fallback safety) end to end.

## Implementation attestation

The manifest published at miner startup attests every served runtime file with
`implementation_sha256` under the versioned scheme
**`repo-relative-path-and-content-v1`** (`implementation_sha256_scheme` field):
files are hashed as sorted repo-relative POSIX paths + content lengths + exact
bytes, so the digest is identical for identical source trees under any
checkout directory or OS path style. `model.joblib` is attested separately via
`artifact_sha256`.

## Runtime environment

Expected dependencies: `bittensor`, `numpy`, `scikit-learn==1.7.2`, `joblib`
(see `requirements.txt`). The production launch script pins
`OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1`,
`NUMEXPR_NUM_THREADS=1` before Python starts (operator-overridable): on a
12-core host, default OpenMP threading made the per-request GBM predict ~875×
slower (≈2900 ms vs 3.3 ms per 100-chunk batch) with bit-for-bit identical
scores. The miner logs the effective limits at startup.

## Limitations

- The public benchmark is a rolling window: past release dates may cycle out of
  the API. `reproduce_gbm35.py` reports `BLOCKED_MISSING_RELEASES` when a
  declared date is neither cached nor served, rather than silently training on
  different data.
- Binary artifact reproduction is not byte-deterministic (reasons documented
  above); strict trained-state equivalence is the verified guarantee.
- Distribution shift: bot families rotate per release, so offline metrics on
  past releases bound, but do not guarantee, live performance.
- Live validator evaluation uses labels that are never published; no claim of
  full live reproducibility is made. Offline evaluation on public benchmark
  releases (walk-forward by date) is the honest proxy used for selection, and
  no validator-only or private data enters training.
