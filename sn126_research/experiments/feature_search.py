"""Rank candidate all-table interaction features by MARGINAL AP/recall lift.

Model is held FIXED (same HistGBM) as the measuring instrument. We do not tune or
replace it. We measure how much each candidate feature adds on top of the current
30, using leave-one-date-out (LODO) to be robust to the small data.

Run: python -m sn126_research.experiments.feature_search
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sn126_research.benchmark_client import BenchmarkClient  # noqa: E402
from sn126_research.evaluation import evaluate  # noqa: E402
from sn126_research.features import FEATURE_NAMES, feature_vector  # noqa: E402
from sn126_research.features_plus import CANDIDATE_NAMES, candidate_vector  # noqa: E402
from sn126_research.sanitizer import to_live_group  # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402


def _gbm():
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.08, max_iter=300, l2_regularization=1.0, random_state=0
    )


def lodo(Xb, Xc, y, dates, cols):
    """Leave-one-date-out mean metrics for base + selected candidate cols."""
    aps, recs, rews = [], [], []
    uniq = sorted(set(dates.tolist()))
    for d in uniq:
        tr = dates != d
        va = dates == d
        if len(np.unique(y[va])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        X = Xb if not cols else np.hstack([Xb, Xc[:, cols]])
        gb = _gbm().fit(X[tr], y[tr])
        p = gb.predict_proba(X[va])[:, 1]
        m = evaluate(y[va], p)
        aps.append(m["ap"]); recs.append(m["recall_at_5pct_fpr"]); rews.append(m["reward"])
    return float(np.mean(aps)), float(np.mean(recs)), float(np.mean(rews))


def main() -> None:
    n_dates = int(os.getenv("SN126_N_DATES", "3"))
    max_records = int(os.getenv("SN126_MAX_RECORDS", "60"))
    client = BenchmarkClient()
    by_date = client.download_recent(n_dates=n_dates, max_records_per_date=max_records)
    records = [r for d in sorted(by_date) for r in by_date[d]]

    Xb, Xc, y, dates = [], [], [], []
    for rec in records:
        for gi, group in enumerate(rec.groups):
            if gi >= len(rec.ground_truth):
                continue
            live = to_live_group(group)
            if not live:
                continue
            Xb.append(feature_vector(live, scope="all"))
            Xc.append(candidate_vector(live))
            y.append(int(rec.ground_truth[gi]))
            dates.append(rec.source_date)
    Xb = np.asarray(Xb, float); Xc = np.asarray(Xc, float)
    y = np.asarray(y, int); dates = np.asarray(dates)
    print(f"samples={len(y)} base_dims={Xb.shape[1]} candidate_dims={Xc.shape[1]} "
          f"dates={sorted(set(dates.tolist()))} bots={int(y.sum())} humans={int((y==0).sum())}")

    base_ap, base_rec, base_rew = lodo(Xb, Xc, y, dates, cols=[])
    print(f"\nBASE-30 (LODO mean): AP={base_ap:.4f} recall@5%={base_rec:.4f} reward={base_rew:.4f}")

    print("\n--- per-candidate MARGINAL lift (base-30 + one candidate, LODO) ---")
    print(f"{'candidate':26s} {'dAP':>7s} {'dRecall':>8s} {'uniAP':>6s} {'bot>hum?':>8s}")
    rows = []
    for j, name in enumerate(CANDIDATE_NAMES):
        ap, rec, _ = lodo(Xb, Xc, y, dates, cols=[j])
        col = Xc[:, j]
        uni = max(average_precision_score(y, col), average_precision_score(y, -col)) if not np.all(col == col[0]) else 0.5
        direction = "bot_hi" if (col[y == 1].mean() > col[y == 0].mean()) else "bot_lo"
        rows.append((name, ap - base_ap, rec - base_rec, uni, direction, j))
    rows.sort(key=lambda r: r[1], reverse=True)
    for name, dap, drec, uni, direction, _ in rows:
        print(f"{name:26s} {dap:+7.4f} {drec:+8.4f} {uni:6.3f} {direction:>8s}")

    print("\n--- greedy forward selection (add candidates that improve LODO AP) ---")
    selected: list[int] = []
    cur_ap = base_ap
    remaining = [r[5] for r in rows]  # candidate col indices in marginal-lift order
    while remaining:
        best_gain, best_j = 0.0, None
        for j in remaining:
            ap, _, _ = lodo(Xb, Xc, y, dates, cols=selected + [j])
            if ap - cur_ap > best_gain:
                best_gain, best_j = ap - cur_ap, j
        if best_j is None or best_gain < 1e-4:
            break
        selected.append(best_j)
        remaining.remove(best_j)
        cur_ap += best_gain
        print(f"  + {CANDIDATE_NAMES[best_j]:26s} -> LODO AP={cur_ap:.4f} (gain {best_gain:+.4f})")

    if selected:
        ap, rec, rew = lodo(Xb, Xc, y, dates, cols=selected)
        print(f"\nBASE-30 + {len(selected)} selected: AP={ap:.4f} (+{ap-base_ap:.4f}) "
              f"recall@5%={rec:.4f} (+{rec-base_rec:.4f}) reward={rew:.4f} (+{rew-base_rew:.4f})")
        print("selected:", [CANDIDATE_NAMES[j] for j in selected])
    else:
        print("no candidate improved LODO AP over base-30.")


if __name__ == "__main__":
    main()
