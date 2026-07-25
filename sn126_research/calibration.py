"""Score transforms for maximizing the Poker44 ``reward()``.

Key fact (see poker44/score/scoring.py):

    reward = 0.35*AP + 0.30*recall@5%FPR + 0.30*TSQ + 0.05      (TSQ<=0 => reward 0)

AP and recall@5%FPR depend ONLY on the ORDER of scores, so any strictly
monotonic transform preserves them exactly. The only lever a transform has is
TSQ (the threshold-sanity gate at 0.5), which is 1.0 iff:
    * at least one bot scores >= 0.5, AND
    * <= 10% of humans score >= 0.5 (human FPR@0.5 <= 0.10).

So "calibration" here is NOT probability calibration -- it is anchoring the value
0.5 at the raw-score quantile that yields a low human FPR. Standard Platt/isotonic
calibrate to P(bot); on class-balanced data that puts 0.5 near the median (~50%
FPR) and FAILS the gate. The winning transform is a monotonic map anchored so the
~95th human-score percentile -> 0.5.

Every calibrator here is fit on TRAIN scores/labels only and applied to new scores.
Outputs are clipped to [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover
    IsotonicRegression = None
    LogisticRegression = None

_EPS = 1e-9


class Calibrator(Protocol):
    name: str
    def fit(self, scores: np.ndarray, labels: np.ndarray) -> "Calibrator": ...
    def transform(self, scores: np.ndarray) -> np.ndarray: ...


# --------------------------------------------------------------------------- #
# Probability calibrators (monotonic, but calibrate to P(bot) -> gate-blind)
# --------------------------------------------------------------------------- #
@dataclass
class PlattCalibrator:
    name: str = "platt"
    _lr: object = None

    def fit(self, scores, labels):
        self._lr = LogisticRegression(C=1e6, solver="lbfgs")
        self._lr.fit(np.asarray(scores).reshape(-1, 1), np.asarray(labels).astype(int))
        return self

    def transform(self, scores):
        p = self._lr.predict_proba(np.asarray(scores).reshape(-1, 1))[:, 1]
        return np.clip(p, 0.0, 1.0)


@dataclass
class IsotonicCalibrator:
    name: str = "isotonic"
    _iso: object = None

    def fit(self, scores, labels):
        self._iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        self._iso.fit(np.asarray(scores, float), np.asarray(labels, float))
        return self

    def transform(self, scores):
        return np.clip(self._iso.predict(np.asarray(scores, float)), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Operating-point transforms (anchor 0.5 at a chosen human FPR)  <-- the fix
# --------------------------------------------------------------------------- #
@dataclass
class AffineAnchor:
    """T(s) = clip(0.5 + 0.5*(s - c)/scale, 0, 1), c = human quantile(1-target_fpr)."""

    target_fpr: float = 0.05
    name: str = "affine_anchor"
    _c: float = 0.5
    _scale: float = 1.0

    def fit(self, scores, labels):
        scores = np.asarray(scores, float)
        labels = np.asarray(labels, int)
        human = scores[labels == 0]
        self._c = float(np.quantile(human, 1.0 - self.target_fpr)) if human.size else 0.5
        lo, hi = float(scores.min()), float(scores.max())
        self._scale = max(self._c - lo, hi - self._c, _EPS)
        return self

    def transform(self, scores):
        s = np.asarray(scores, float)
        return np.clip(0.5 + 0.5 * (s - self._c) / self._scale, 0.0, 1.0)


@dataclass
class QuantileAnchor:
    """Piecewise-linear monotonic map anchored so c -> 0.5, spreading each side.

    Below c: linear into [0, 0.5]; above c: linear into [0.5, 1]. c = human
    quantile(1 - target_fpr). Preserves order; robust and simple.
    """

    target_fpr: float = 0.05
    name: str = "quantile_anchor"
    _c: float = 0.5
    _lo: float = 0.0
    _hi: float = 1.0

    def fit(self, scores, labels):
        scores = np.asarray(scores, float)
        labels = np.asarray(labels, int)
        human = scores[labels == 0]
        self._c = float(np.quantile(human, 1.0 - self.target_fpr)) if human.size else 0.5
        self._lo, self._hi = float(scores.min()), float(scores.max())
        return self

    def transform(self, scores):
        s = np.asarray(scores, float)
        below = 0.5 * (s - self._lo) / max(self._c - self._lo, _EPS)
        above = 0.5 + 0.5 * (s - self._c) / max(self._hi - self._c, _EPS)
        out = np.where(s <= self._c, below, above)
        return np.clip(out, 0.0, 1.0)


@dataclass
class NaiveRankTransform:
    """Percentile of each score within the TRAIN score distribution (median->0.5).

    Included to demonstrate why a naive rank/percentile map fails the gate on
    balanced data: 0.5 lands at the median -> ~50% human FPR.
    """

    name: str = "rank_naive"
    _sorted: np.ndarray = None

    def fit(self, scores, labels):
        self._sorted = np.sort(np.asarray(scores, float))
        return self

    def transform(self, scores):
        s = np.asarray(scores, float)
        idx = np.searchsorted(self._sorted, s, side="right")
        return np.clip(idx / max(self._sorted.size, 1), 0.0, 1.0)


@dataclass
class BatchPercentileAnchor:
    """Shift-ROBUST anchor: operate on within-request RANK, not absolute score.

    The validator computes TSQ per request, and the miner sees the whole request
    (many chunks). Absolute GBM scores drift across dates/windows, but the RANK
    structure is what AP/recall already reward and is far more stable. We learn a
    single batch-percentile threshold p0 (the human rank-quantile at 1-target_fpr)
    on train, then at inference rank each request internally and anchor p0 -> 0.5.
    """

    target_fpr: float = 0.05
    name: str = "batch_pct_anchor"
    _p0: float = 0.9

    @staticmethod
    def _pct(scores: np.ndarray) -> np.ndarray:
        s = np.asarray(scores, float)
        srt = np.sort(s)
        # fraction of batch <= each score (within-request percentile)
        return np.searchsorted(srt, s, side="right") / max(s.size, 1)

    def fit(self, scores, labels):
        scores = np.asarray(scores, float)
        labels = np.asarray(labels, int)
        r = self._pct(scores)
        human_r = r[labels == 0]
        self._p0 = float(np.quantile(human_r, 1.0 - self.target_fpr)) if human_r.size else 0.9
        return self

    def transform(self, scores):
        r = self._pct(scores)  # ranks within THIS request
        below = 0.5 * r / max(self._p0, _EPS)
        above = 0.5 + 0.5 * (r - self._p0) / max(1.0 - self._p0, _EPS)
        return np.clip(np.where(r <= self._p0, below, above), 0.0, 1.0)


def default_suite(target_fpr: float = 0.05):
    """The methods requested for comparison."""
    suite = [
        PlattCalibrator(),
        IsotonicCalibrator(),
        AffineAnchor(target_fpr=target_fpr),
        QuantileAnchor(target_fpr=target_fpr),
        NaiveRankTransform(),
        BatchPercentileAnchor(target_fpr=target_fpr),
    ]
    return [c for c in suite if not (c.name in {"platt", "isotonic"} and LogisticRegression is None)]
