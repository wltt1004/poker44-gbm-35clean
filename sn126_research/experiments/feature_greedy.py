"""Greedy feature selection on the ACTUAL SN126 reward (calibration-solved).

Start set (current best):
    baseline 30  +  action_bigram_entropy  +  betsize_street_std_mean
Candidate pool:
    all remaining plus1 (17) + plus2 (8 unused) = 25 features.

Objective = post-calibration reward = 0.35*AP + 0.30*recall@5%FPR + 0.35.
(Because the batch-anchor calibration forces TSQ=1.0 in production, this is the
reward the miner actually earns. Optimizing the raw uncalibrated reward() would
chase 0.5-gate noise that calibration removes -> overfitting.)

Rules: add one feature at a time; keep only if it improves validation reward by
> EPS; stop when nothing improves. Validation split unchanged (LODO over the same
6 release dates). Model FIXED (same HistGBM, random_state=0).

Run: python -m sn126_research.experiments.feature_greedy
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
from sn126_research.features_plus import CANDIDATE_NAMES as P1  # noqa: E402
from sn126_research.features_plus import candidate_vector as p1_vec  # noqa: E402
from sn126_research.features_plus2 import CANDIDATE_NAMES as P2  # noqa: E402
from sn126_research.features_plus2 import candidate_vector as p2_vec  # noqa: E402
from sn126_research.sanitizer import to_live_group  # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

EPS = 1e-4  # minimum calibrated-reward gain to accept a feature (guards against noise)


def _gbm():
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.08, max_iter=300, l2_regularization=1.0, random_state=0
    )


def lodo(X, y, dates):
    """LODO mean AP, recall@5%FPR, raw reward()."""
    aps, recs, rews = [], [], []
    for d in sorted(set(dates.tolist())):
        tr, va = dates != d, dates == d
        if len(np.unique(y[va])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        gb = _gbm().fit(X[tr], y[tr])
        p = gb.predict_proba(X[va])[:, 1]
        m = evaluate(y[va], p)
        aps.append(m["ap"]); recs.append(m["recall_at_5pct_fpr"]); rews.append(m["reward"])
    ap, rec, rew = float(np.mean(aps)), float(np.mean(recs)), float(np.mean(rews))
    cal = 0.35 * ap + 0.30 * rec + 0.35  # post-calibration reward
    return ap, rec, rew, cal


def main() -> None:
    n_dates = int(os.getenv("SN126_N_DATES", "6"))
    max_records = int(os.getenv("SN126_MAX_RECORDS", "200"))
    client = BenchmarkClient()
    by_date = client.download_recent(n_dates=n_dates, max_records_per_date=max_records)
    records = [r for d in sorted(by_date) for r in by_date[d]]

    Xb, C, y, dates = [], [], [], []
    for rec in records:
        for gi, group in enumerate(rec.groups):
            if gi >= len(rec.ground_truth):
                continue
            live = to_live_group(group)
            if not live:
                continue
            Xb.append(feature_vector(live, scope="all"))
            C.append(list(p1_vec(live)) + list(p2_vec(live)))  # 27 candidate cols
            y.append(int(rec.ground_truth[gi]))
            dates.append(rec.source_date)
    Xb = np.asarray(Xb, float); C = np.asarray(C, float)
    y = np.asarray(y, int); dates = np.asarray(dates)

    names = [f"p1:{n}" for n in P1] + [f"p2:{n}" for n in P2]
    idx = {n: i for i, n in enumerate(names)}
    start = [idx["p2:betsize_street_std_mean"], idx["p2:action_bigram_entropy"]]
    pool = [i for i in range(len(names)) if i not in start]

    def X_of(cols):
        return Xb if not cols else np.hstack([Xb, C[:, cols]])

    ap0, rec0, rew0, cal0 = lodo(X_of(start), y, dates)
    print(f"samples={len(y)} dates={sorted(set(dates.tolist()))}")
    print(f"START (base30 + 2): AP={ap0:.4f} recall@5%={rec0:.4f} raw_reward={rew0:.4f} "
          f"calibrated_reward={cal0:.4f}")
    print(f"\ngreedy on CALIBRATED reward (accept if gain > {EPS}); model fixed; LODO unchanged\n")

    selected = list(start)
    cur_ap, cur_rec, cur_rew, cur_cal = ap0, rec0, rew0, cal0
    accepted = []
    while pool:
        best = None  # (cal, ap, rec, rew, j)
        for j in pool:
            ap, rec, rew, cal = lodo(X_of(selected + [j]), y, dates)
            if best is None or cal > best[0]:
                best = (cal, ap, rec, rew, j)
        cal, ap, rec, rew, j = best
        if cal - cur_cal <= EPS:
            print(f"stop: best remaining ({names[j]}) gives Δcal_reward={cal-cur_cal:+.4f} <= {EPS}")
            break
        # correlation of the accepted feature with already-selected candidates (redundancy check)
        if len(selected) > len(start):
            prior = C[:, [s for s in selected if s not in start]]
            corr = float(np.max(np.abs([np.corrcoef(C[:, j], prior[:, k])[0, 1] for k in range(prior.shape[1])]))) if prior.shape[1] else 0.0
        else:
            corr = 0.0
        accepted.append((names[j], ap - cur_ap, rec - cur_rec, rew - cur_rew, cal - cur_cal, corr))
        print(f"+ {names[j]:28s} ΔAP={ap-cur_ap:+.4f} Δrecall={rec-cur_rec:+.4f} "
              f"Δraw_reward={rew-cur_rew:+.4f} Δcal_reward={cal-cur_cal:+.4f} | maxcorr_prior={corr:.2f}")
        selected.append(j); pool.remove(j)
        cur_ap, cur_rec, cur_rew, cur_cal = ap, rec, rew, cal

    print("\n=== FINAL SET ===")
    added = [names[j] for j in selected if j not in start]
    print(f"base30 + action_bigram_entropy + betsize_street_std_mean + {len(added)} more")
    print(f"added by greedy: {added}")
    print(f"FINAL: AP={cur_ap:.4f} (+{cur_ap-ap0:.4f})  recall@5%={cur_rec:.4f} (+{cur_rec-rec0:.4f})  "
          f"raw_reward={cur_rew:.4f}  calibrated_reward={cur_cal:.4f} (+{cur_cal-cal0:.4f})")
    print(f"total features = {Xb.shape[1] + len(selected)} "
          f"({Xb.shape[1]} base + {len(selected)} candidates)")

    print("\n--- per accepted feature ---")
    print(f"{'feature':30s} {'dAP':>7s} {'dRecall':>8s} {'dRawRew':>8s} {'dCalRew':>8s} {'corr':>5s}")
    for nm, dap, drec, drew, dcal, corr in accepted:
        print(f"{nm:30s} {dap:+7.4f} {drec:+8.4f} {drew:+8.4f} {dcal:+8.4f} {corr:5.2f}")


if __name__ == "__main__":
    main()
