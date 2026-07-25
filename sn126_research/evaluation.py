"""Evaluation against the Poker44 reward objective.

The authoritative metric is the validator's own ``reward()`` (imported from
``poker44.score.scoring``). We additionally expose its rank-based components so
experiments can reason about them directly:

  * average precision              (AP_WEIGHT = 0.35)   -- rank-invariant
  * recall at <=5% human FPR       (BOT_RECALL_WEIGHT = 0.30) -- rank-invariant
  * threshold-sanity gate at 0.5   (drives 0.30 of the reward + the zero-cliff)

Reminder for model selection: AP and recall@5%FPR depend ONLY on the ordering of
scores, so optimize ranking. The 0.5 gate is pass/fail — clear it, don't tune it.
ROC-AUC is reported for diagnostics only; it is NOT the objective.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from poker44.score.scoring import reward as official_reward  # noqa: E402

try:
    from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
except Exception:  # pragma: no cover
    average_precision_score = None
    roc_auc_score = None


def recall_at_fpr(y_score: np.ndarray, y_true: np.ndarray, *, max_fpr: float = 0.05):
    """Best bot recall while keeping human false-positive-rate <= max_fpr."""
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    pos = int(np.sum(labels == 1))
    neg = int(np.sum(labels == 0))
    if pos == 0 or neg == 0 or scores.size == 0:
        return 0.0, 0.0
    order = np.argsort(-scores, kind="mergesort")
    sl = labels[order]
    tp = np.cumsum(sl == 1)
    fp = np.cumsum(sl == 0)
    recall = tp / max(pos, 1)
    fpr = fp / max(neg, 1)
    allowed = fpr <= float(max_fpr)
    if not np.any(allowed):
        return 0.0, 0.0
    idx = np.flatnonzero(allowed)
    best = int(idx[np.argmax(recall[allowed])])
    return float(recall[best]), float(fpr[best])


def threshold_gate(y_score: np.ndarray, y_true: np.ndarray, *, threshold: float = 0.5) -> Dict[str, float]:
    """Report the 0.5 threshold-sanity gate (the zero-cliff condition)."""
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    pos = int(np.sum(labels == 1))
    neg = int(np.sum(labels == 0))
    pred = scores >= threshold
    tp = int(np.sum(pred & (labels == 1)))
    fp = int(np.sum(pred & (labels == 0)))
    hard_fpr = fp / max(neg, 1) if neg else 0.0
    if pos == 0 or neg == 0:
        quality = 1.0
    elif tp <= 0:
        quality = 0.0
    elif hard_fpr <= 0.10:
        quality = 1.0
    else:
        quality = max(0.0, 1.0 - (hard_fpr - 0.10) / 0.90)
    return {
        "gate_quality": float(quality),
        "hard_fpr_at_0.5": float(hard_fpr),
        "positive_rate_at_0.5": float(np.mean(pred)) if pred.size else 0.0,
        "passes_gate": float(quality > 0.0),
    }


def evaluate(y_true, y_score) -> Dict[str, float]:
    """Full evaluation dict. ``reward`` is the authoritative headline number."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    rew, res = official_reward(y_score, y_true)
    recall, fpr = recall_at_fpr(y_score, y_true, max_fpr=0.05)
    gate = threshold_gate(y_score, y_true)

    ap = (
        float(average_precision_score(y_true, y_score))
        if average_precision_score is not None and np.any(y_true == 1)
        else float(res.get("ap_score", 0.0))
    )
    auc = (
        float(roc_auc_score(y_true, y_score))
        if roc_auc_score is not None and 0 < int(np.sum(y_true == 1)) < len(y_true)
        else float("nan")
    )

    out = {
        "reward": float(rew),            # <-- authoritative objective
        "ap": ap,
        "recall_at_5pct_fpr": float(recall),
        "fpr_at_operating_point": float(fpr),
        "roc_auc_diagnostic": auc,       # diagnostic only, NOT the objective
        "n": int(y_true.size),
        "n_bots": int(np.sum(y_true == 1)),
        "n_humans": int(np.sum(y_true == 0)),
    }
    out.update(gate)
    return out


def format_report(name: str, m: Dict[str, float]) -> str:
    return (
        f"[{name}] reward={m['reward']:.4f} | AP={m['ap']:.4f} | "
        f"recall@5%FPR={m['recall_at_5pct_fpr']:.4f} | gate={m['gate_quality']:.2f} "
        f"(fpr@0.5={m['hard_fpr_at_0.5']:.3f}) | AUC(diag)={m['roc_auc_diagnostic']:.4f} | "
        f"n={m['n']} bots={m['n_bots']} humans={m['n_humans']}"
    )
