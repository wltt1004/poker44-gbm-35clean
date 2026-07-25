"""Ablation for the conditional/determinism features (features_plus2).

Baseline = current features.py (30). New = features_plus2 (10). Model FIXED
(same HistGBM). LODO over all downloaded release dates.

Reports AP, recall@5%FPR, reward() for:
  baseline
  baseline + new(all 10)
  baseline + new(greedy-selected)
  baseline + plus1 + plus2 (everything, for the aggregate ceiling)

Run: python -m sn126_research.experiments.feature_search2
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
from sn126_research.features_plus import candidate_vector as plus1_vec  # noqa: E402
from sn126_research.features_plus2 import CANDIDATE_NAMES as NEW_NAMES  # noqa: E402
from sn126_research.features_plus2 import candidate_vector as plus2_vec  # noqa: E402
from sn126_research.sanitizer import to_live_group  # noqa: E402

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402


def _gbm():
    return HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.08, max_iter=300, l2_regularization=1.0, random_state=0
    )


def lodo(X, y, dates):
    aps, recs, rews = [], [], []
    for d in sorted(set(dates.tolist())):
        tr, va = dates != d, dates == d
        if len(np.unique(y[va])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        gb = _gbm().fit(X[tr], y[tr])
        p = gb.predict_proba(X[va])[:, 1]
        m = evaluate(y[va], p)
        aps.append(m["ap"]); recs.append(m["recall_at_5pct_fpr"]); rews.append(m["reward"])
    return float(np.mean(aps)), float(np.mean(recs)), float(np.mean(rews))


def _cat(*blocks):
    return np.hstack([b for b in blocks if b.shape[1] > 0])


def main() -> None:
    n_dates = int(os.getenv("SN126_N_DATES", "6"))
    max_records = int(os.getenv("SN126_MAX_RECORDS", "200"))
    client = BenchmarkClient()
    by_date = client.download_recent(n_dates=n_dates, max_records_per_date=max_records)
    records = [r for d in sorted(by_date) for r in by_date[d]]

    Xb, X1, X2, y, dates = [], [], [], [], []
    for rec in records:
        for gi, group in enumerate(rec.groups):
            if gi >= len(rec.ground_truth):
                continue
            live = to_live_group(group)
            if not live:
                continue
            Xb.append(feature_vector(live, scope="all"))
            X1.append(plus1_vec(live))
            X2.append(plus2_vec(live))
            y.append(int(rec.ground_truth[gi]))
            dates.append(rec.source_date)
    Xb = np.asarray(Xb, float); X1 = np.asarray(X1, float); X2 = np.asarray(X2, float)
    y = np.asarray(y, int); dates = np.asarray(dates)
    print(f"samples={len(y)} base={Xb.shape[1]} plus1={X1.shape[1]} plus2(new)={X2.shape[1]} "
          f"dates={sorted(set(dates.tolist()))} bots={int(y.sum())} humans={int((y==0).sum())}")

    b_ap, b_rec, b_rew = lodo(Xb, y, dates)
    print(f"\nBASELINE (features.py, 30):        AP={b_ap:.4f}  recall@5%={b_rec:.4f}  reward={b_rew:.4f}")

    print("\n--- per NEW feature: marginal lift (baseline + one, LODO) ---")
    print(f"{'feature':26s} {'dAP':>7s} {'dRecall':>8s} {'uniAP':>6s} {'dir':>7s}")
    rows = []
    for j, name in enumerate(NEW_NAMES):
        ap, rec, _ = lodo(_cat(Xb, X2[:, [j]]), y, dates)
        col = X2[:, j]
        uni = max(average_precision_score(y, col), average_precision_score(y, -col)) if not np.all(col == col[0]) else 0.5
        direction = "bot_hi" if col[y == 1].mean() > col[y == 0].mean() else "bot_lo"
        rows.append((name, ap - b_ap, rec - b_rec, uni, direction, j))
    rows.sort(key=lambda r: r[1], reverse=True)
    for name, dap, drec, uni, direction, _ in rows:
        print(f"{name:26s} {dap:+7.4f} {drec:+8.4f} {uni:6.3f} {direction:>7s}")

    print("\n--- greedy forward selection over NEW features (LODO AP) ---")
    selected, cur = [], b_ap
    remaining = [r[5] for r in rows]
    while remaining:
        best_gain, best_j = 0.0, None
        for j in remaining:
            ap, _, _ = lodo(_cat(Xb, X2[:, selected + [j]]), y, dates)
            if ap - cur > best_gain:
                best_gain, best_j = ap - cur, j
        if best_j is None or best_gain < 1e-4:
            break
        selected.append(best_j); remaining.remove(best_j); cur += best_gain
        print(f"  + {NEW_NAMES[best_j]:26s} -> LODO AP={cur:.4f} ({best_gain:+.4f})")

    print("\n=== ABLATION SUMMARY (LODO mean) ===")
    print(f"baseline (30)                 AP={b_ap:.4f}  recall@5%={b_rec:.4f}  reward={b_rew:.4f}")
    ap, rec, rew = lodo(_cat(Xb, X2), y, dates)
    print(f"baseline + NEW all 10         AP={ap:.4f}  recall@5%={rec:.4f}  reward={rew:.4f}")
    if selected:
        ap, rec, rew = lodo(_cat(Xb, X2[:, selected]), y, dates)
        print(f"baseline + NEW selected({len(selected)})     AP={ap:.4f} (+{ap-b_ap:.4f})  "
              f"recall@5%={rec:.4f} (+{rec-b_rec:.4f})  reward={rew:.4f} (+{rew-b_rew:.4f})")
        print("  selected:", [NEW_NAMES[j] for j in selected])
    ap, rec, rew = lodo(_cat(Xb, X1, X2), y, dates)
    print(f"baseline + plus1 + plus2 (all) AP={ap:.4f}  recall@5%={rec:.4f}  reward={rew:.4f}")
    ceil_b = 0.35 * b_ap + 0.30 * b_rec + 0.35
    ceil_n = 0.35 * ap + 0.30 * rec + 0.35
    print(f"\npost-calibration reward ceiling: baseline={ceil_b:.4f} -> all-features={ceil_n:.4f} "
          f"(0.35*AP+0.30*recall+0.35)")


if __name__ == "__main__":
    main()
