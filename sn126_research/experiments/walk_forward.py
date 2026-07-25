"""Walk-forward robustness: train on EARLIER dates, validate on later UNSEEN dates.

This is the live-mining scenario (you only have the past). LODO leaks future
dates into training and overstates generalization; walk-forward does not.

Expanding window: for each validation date d, train on ALL dates strictly before d.
Compares the 37-feature set vs the 35-feature clean set, and checks whether
big_pot_rate + cbet_freq add CONSISTENT out-of-time gains or only one-date luck.

Metrics: AP, recall@5%FPR, calibrated reward = 0.35*AP + 0.30*recall + 0.35.
Model FIXED (HistGBM, random_state=0). No new features.

Run: python -m sn126_research.experiments.walk_forward
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sn126_research.benchmark_client import BenchmarkClient  # noqa: E402
from sn126_research.evaluation import evaluate  # noqa: E402
from sn126_research.features import feature_vector  # noqa: E402
from sn126_research.features_plus import CANDIDATE_NAMES as P1, candidate_vector as p1v  # noqa: E402
from sn126_research.features_plus2 import CANDIDATE_NAMES as P2, candidate_vector as p2v  # noqa: E402
from sn126_research.sanitizer import to_live_group  # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402


def gbm():
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.08, max_iter=300, l2_regularization=1.0, random_state=0
    )


def cal(ap, rec):
    return 0.35 * ap + 0.30 * rec + 0.35


def main() -> None:
    n_dates = int(os.getenv("SN126_N_DATES", "8"))
    max_records = int(os.getenv("SN126_MAX_RECORDS", "200"))
    client = BenchmarkClient()
    by_date = client.download_recent(n_dates=n_dates, max_records_per_date=max_records)
    records = [r for d in sorted(by_date) for r in by_date[d]]

    Xb, C, y, dates = [], [], [], []
    for rec in records:
        for gi, g in enumerate(rec.groups):
            if gi >= len(rec.ground_truth):
                continue
            live = to_live_group(g)
            if not live:
                continue
            Xb.append(feature_vector(live, scope="all"))
            C.append(list(p1v(live)) + list(p2v(live)))
            y.append(int(rec.ground_truth[gi])); dates.append(rec.source_date)
    Xb = np.asarray(Xb, float); C = np.asarray(C, float)
    y = np.asarray(y, int); dates = np.asarray(dates)

    names = [f"p1:{n}" for n in P1] + [f"p2:{n}" for n in P2]
    idx = {n: i for i, n in enumerate(names)}
    common = [idx["p2:betsize_street_std_mean"], idx["p2:action_bigram_entropy"],
              idx["p1:pot_escalation_mean"], idx["p2:mean_aggressors_per_hand"],
              idx["p1:potodds_call_fold_gap"]]
    pair = [idx["p1:cbet_freq"], idx["p1:big_pot_rate"]]
    cols35 = common
    cols37 = common + pair

    def X(cols):
        return np.hstack([Xb, C[:, cols]])

    uniq = sorted(set(dates.tolist()))
    print(f"samples={len(y)} dates={uniq} (expanding walk-forward)\n")

    folds = list(range(2, len(uniq)))  # validate on dates[2:], train on all earlier
    pooled = {"35": ([], []), "37": ([], [])}  # (scores, labels)
    per_fold = {"35": [], "37": []}

    hdr = f"{'val_date':12s} {'ntr':>4s} {'nva':>4s} | {'AP35':>6s} {'rec35':>6s} {'cal35':>6s} | {'AP37':>6s} {'rec37':>6s} {'cal37':>6s} | {'dRec':>6s} {'dCal':>6s}"
    print(hdr)
    print("-" * len(hdr))
    for k in folds:
        vd = uniq[k]
        tr = np.array([d in uniq[:k] for d in dates])
        va = dates == vd
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
            continue
        row = {}
        for tag, cols in (("35", cols35), ("37", cols37)):
            Xtr, Xva = X(cols)[tr], X(cols)[va]
            m = gbm().fit(Xtr, y[tr])
            p = m.predict_proba(Xva)[:, 1]
            ev = evaluate(y[va], p)
            row[tag] = (ev["ap"], ev["recall_at_5pct_fpr"], cal(ev["ap"], ev["recall_at_5pct_fpr"]))
            per_fold[tag].append(row[tag])
            pooled[tag][0].extend(p.tolist()); pooled[tag][1].extend(y[va].tolist())
        a35, r35, c35 = row["35"]; a37, r37, c37 = row["37"]
        print(f"{vd:12s} {int(tr.sum()):4d} {int(va.sum()):4d} | {a35:6.3f} {r35:6.3f} {c35:6.3f} | "
              f"{a37:6.3f} {r37:6.3f} {c37:6.3f} | {r37-r35:+6.3f} {c37-c35:+6.3f}")

    def agg(tag):
        arr = np.array(per_fold[tag])
        return arr.mean(0), arr.std(0)

    print("\n=== per-fold mean +/- std (out-of-time) ===")
    for tag in ("35", "37"):
        mu, sd = agg(tag)
        print(f"  set-{tag}: AP={mu[0]:.4f}+/-{sd[0]:.3f}  recall={mu[1]:.4f}+/-{sd[1]:.3f}  cal_reward={mu[2]:.4f}+/-{sd[2]:.3f}")

    print("\n=== POOLED out-of-time (all walk-forward val predictions) ===")
    for tag in ("35", "37"):
        s = np.array(pooled[tag][0]); yy = np.array(pooled[tag][1])
        ev = evaluate(yy, s)
        print(f"  set-{tag}: AP={ev['ap']:.4f}  recall@5%={ev['recall_at_5pct_fpr']:.4f}  "
              f"cal_reward={cal(ev['ap'], ev['recall_at_5pct_fpr']):.4f}  (n={len(yy)})")

    # consistency of the big_pot_rate + cbet_freq pair
    d_rec = [f37[1] - f35[1] for f35, f37 in zip(per_fold["35"], per_fold["37"])]
    d_cal = [f37[2] - f35[2] for f35, f37 in zip(per_fold["35"], per_fold["37"])]
    pos_cal = sum(1 for x in d_cal if x > 0)
    print("\n=== big_pot_rate + cbet_freq: out-of-time consistency (37 - 35) ===")
    print(f"  per-fold ΔcalReward: {[round(x,3) for x in d_cal]}")
    print(f"  per-fold Δrecall   : {[round(x,3) for x in d_rec]}")
    print(f"  folds improved by pair: {pos_cal}/{len(d_cal)} | mean ΔcalReward={np.mean(d_cal):+.4f} "
          f"(std {np.std(d_cal):.4f}) | mean Δrecall={np.mean(d_rec):+.4f}")
    verdict = ("CONSISTENT -> keep 37" if pos_cal >= max(2, int(0.7 * len(d_cal))) and np.mean(d_cal) > 0
               else "INCONSISTENT / one-date -> prefer 35 (safer)")
    print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()
