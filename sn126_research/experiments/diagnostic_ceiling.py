"""Diagnostic: is the near-chance separability caused by the SANITIZER, or are the
benchmark bots genuinely human-like at this feature level?

We compare four feature views on the SAME groups, split by release date:
  1. sanitized + hero-only     (what the miner actually gets = the "safe" set)
  2. raw       + hero-only     (full actions/amounts, no windowing/bucketing)
  3. raw       + all-seats     (whole-table behavior, unfiltered)
  4. raw       + all-seats + exact amounts + action counts   (upper-bound ceiling)

If (4) >> (1), the sanitizer is the bottleneck (low ceiling). If (4) ~ (1) ~ chance,
the bots are behaviorally stealthy and aggregate features won't separate them.

Run: python -m sn126_research.experiments.diagnostic_ceiling
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sn126_research.benchmark_client import BenchmarkClient  # noqa: E402
from sn126_research.evaluation import evaluate, format_report  # noqa: E402
from sn126_research.sanitizer import to_live_hand  # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

_TYPES = ("fold", "check", "call", "bet", "raise")
_AGGR = {"bet", "raise"}
_ORD = {"preflop": 0, "flop": 1, "turn": 2, "river": 3, "showdown": 3}


def _std(xs):
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / n)


def _entropy(counts):
    tot = float(sum(counts))
    if tot <= 0:
        return 0.0
    h = -sum((c / tot) * math.log(c / tot) for c in counts if c > 0)
    return h / math.log(len(counts))


def group_features(group: List[Dict[str, Any]], *, sanitize: bool, hero_only: bool, exact: bool):
    hands = [to_live_hand(h) for h in group] if sanitize else group
    counts = {t: 0 for t in _TYPES}
    total = 0
    per_hand_actions = []
    bet_bb = []
    depths = []
    vpip = pfr = present = 0
    pf_aggr = pf_tot = 0
    for hand in hands:
        hero_seat = (hand.get("metadata") or {}).get("hero_seat")
        acts = hand.get("actions") or []
        if hero_only:
            acts = [a for a in acts if a.get("actor_seat") == hero_seat]
        if not acts:
            continue
        present += 1
        per_hand_actions.append(len(acts))
        saw_v = saw_r = False
        maxs = 0
        for a in acts:
            t = a.get("action_type")
            if t in counts:
                counts[t] += 1
                total += 1
            s = _ORD.get(str(a.get("street") or "preflop").lower(), 0)
            maxs = max(maxs, s)
            if s == 0:
                pf_tot += 1
                pf_aggr += int(t in _AGGR)
                if t in ("call", "bet", "raise"):
                    saw_v = True
                if t == "raise":
                    saw_r = True
            if t in _AGGR:
                bb = float(a.get("normalized_amount_bb") or 0.0)
                if bb > 0:
                    bet_bb.append(bb)
        depths.append(maxs)
        vpip += int(saw_v)
        pfr += int(saw_r)

    n_hands = len(hands)
    if total == 0:
        return [0.0] * (13 + (3 if exact else 0))
    feats = [counts[t] / total for t in _TYPES]
    feats.append((counts["bet"] + counts["raise"]) / total)          # aggression freq
    feats.append(vpip / max(present, 1))
    feats.append(pfr / max(present, 1))
    feats.append(sum(depths) / len(depths))
    feats.append(_std(depths))
    feats.append(_entropy([counts[t] for t in _TYPES]))
    feats.append(present / max(n_hands, 1))
    feats.append(_std(bet_bb) if len(bet_bb) > 1 else 0.0)           # bet-size dispersion
    if exact:
        feats.append(sum(per_hand_actions) / len(per_hand_actions))  # mean actions/hand (UNSAFE)
        feats.append(_std(per_hand_actions))                         # variance of action count
        feats.append(sum(bet_bb) / len(bet_bb) if bet_bb else 0.0)   # exact mean bet bb (UNSAFE)
    return feats


def build(records, cfg):
    X, y, dates = [], [], []
    for rec in records:
        for gi, group in enumerate(rec.groups):
            if gi >= len(rec.ground_truth):
                continue
            X.append(group_features(group, **cfg))
            y.append(int(rec.ground_truth[gi]))
            dates.append(rec.source_date)
    return np.asarray(X, float), np.asarray(y, int), np.asarray(dates)


def run_view(name, records, cfg, holdout):
    X, y, dates = build(records, cfg)
    tr = dates != holdout
    va = dates == holdout
    if len(np.unique(y[va])) < 2 or len(np.unique(y[tr])) < 2:
        print(f"[{name}] skipped (single-class fold)")
        return
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=300, l2_regularization=1.0)
    gb.fit(X[tr], y[tr])
    p = gb.predict_proba(X[va])[:, 1]
    m = evaluate(y[va], p)
    # also best single-feature AUC on train for a signal floor
    best_auc = 0.5
    for j in range(X.shape[1]):
        col = X[tr][:, j]
        if np.all(col == col[0]):
            continue
        a = roc_auc_score(y[tr], col)
        best_auc = max(best_auc, a, 1 - a)
    print(f"[{name:34s}] GBM  reward={m['reward']:.3f} AP={m['ap']:.3f} "
          f"recall@5={m['recall_at_5pct_fpr']:.3f} AUC(val)={m['roc_auc_diagnostic']:.3f} "
          f"| best_single_feat_AUC(train)={best_auc:.3f} | dims={X.shape[1]}")


def main():
    n_dates = int(os.getenv("SN126_N_DATES", "3"))
    max_records = int(os.getenv("SN126_MAX_RECORDS", "40"))
    client = BenchmarkClient()
    by_date = client.download_recent(n_dates=n_dates, max_records_per_date=max_records)
    records = [r for d in sorted(by_date) for r in by_date[d]]
    holdout = sorted({r.source_date for r in records})[-1]
    print(f"records={len(records)} holdout_date={holdout} "
          f"dates={sorted({r.source_date for r in records})}")
    print("-" * 100)
    run_view("1 sanitized + hero (SAFE/live)", records, dict(sanitize=True, hero_only=True, exact=False), holdout)
    run_view("1b sanitized + all-seats (LIVE!)", records, dict(sanitize=True, hero_only=False, exact=False), holdout)
    run_view("2 raw + hero", records, dict(sanitize=False, hero_only=True, exact=False), holdout)
    run_view("3 raw + all-seats", records, dict(sanitize=False, hero_only=False, exact=False), holdout)
    run_view("4 raw + all-seats + exact (CEIL)", records, dict(sanitize=False, hero_only=False, exact=True), holdout)
    print("-" * 100)
    print("Read: view1 = what you actually get. view4 = best case with benchmark data if nothing were hidden.")
    print("If view4 ~ chance too, the benchmark bots are human-like at aggregate-feature level.")


if __name__ == "__main__":
    main()
