"""Batch percentile calibration for SN126 risk scores.

This is the exact calibration used by the predictor, extracted into its own
module to match the documented architecture. LOGIC IS UNCHANGED.

Rationale (see the research findings): the validator's reward gate is a per-request
operating-point constraint, not a probability-accuracy problem, and the GBM's raw
score scale drifts across dates. So we anchor 0.5 at the p0-th percentile of THIS
request's scores (batch-relative). This is rank-invariant — it preserves AP and
recall@FPR and only moves the 0.5 gate. Small requests (< min_calib_batch) fall
back to an absolute anchor learned at train time.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-9


def batch_percentile_calibrate(
    raw: np.ndarray,
    *,
    p0: float,
    min_calib_batch: int,
    global_anchor_raw: float,
) -> np.ndarray:
    """Map raw GBM probabilities to calibrated risk scores in [0, 1].

    raw               : 1-D array of raw predict_proba scores (one per chunk)
    p0                : percentile anchor for the 0.5 threshold (e.g. 0.85 = top 15%)
    min_calib_batch   : below this batch size, use the absolute fallback anchor
    global_anchor_raw : raw-score value that maps to 0.5 in the small-batch fallback
    """
    raw = np.asarray(raw, dtype=float)
    n = raw.size
    if n == 0:
        return raw
    if n >= min_calib_batch:
        srt = np.sort(raw)
        r = np.searchsorted(srt, raw, side="right") / n        # within-request percentile
        below = 0.5 * r / max(p0, _EPS)
        above = 0.5 + 0.5 * (r - p0) / max(1.0 - p0, _EPS)
        out = np.where(r <= p0, below, above)
    else:
        c = global_anchor_raw
        below = 0.5 * (raw - 0.0) / max(c - 0.0, _EPS)
        above = 0.5 + 0.5 * (raw - c) / max(1.0 - c, _EPS)
        out = np.where(raw <= c, below, above)
    return np.clip(out, 0.0, 1.0)
