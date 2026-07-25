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

## Reproducing the model

With this repository installed (`pip install -e .`, scikit-learn 1.7.2):

```python
import json, joblib, numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from poker44.miner_model.build_feature_profile import _iter_records, _recent_source_dates, to_live_group
from poker44.miner_model.features import PROD_FEATURE_ORDER, chunk_feature_vector

X, y = [], []
for date in sorted(_recent_source_dates(8)):
    for rec in _iter_records(date, max_records=200):
        groups, truth = rec.get("chunks") or [], rec.get("groundTruth") or []
        for gi, group in enumerate(groups):
            live = to_live_group(group)
            if gi < len(truth) and live:
                X.append(chunk_feature_vector(live)); y.append(int(truth[gi]))
X, y = np.asarray(X, float), np.asarray(y, int)
model = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=300,
                                       l2_regularization=1.0, random_state=0).fit(X, y)
anchor = float(np.quantile(model.predict_proba(X)[:, 1], 0.85))
joblib.dump(model, "model_reproduced.joblib")
```

`python -m poker44.miner_model.smoke_test` then proves the served pipeline
(feature contract, calibration paths, fallback safety) end to end.

## Limitations

- The public benchmark is a rolling window: past release dates may cycle out of
  the API, so byte-identical artifact reproduction requires the same data
  snapshot (the served artifact is pinned by its SHA256 instead).
- Bit-identical GBM training additionally requires the same scikit-learn
  (1.7.2), NumPy and platform.
- Live validator evaluation uses labels that are never published; no claim of
  full live reproducibility is made. Offline evaluation on public benchmark
  releases (walk-forward by date) is the honest proxy used for selection.
