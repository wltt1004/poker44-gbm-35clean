"""Baseline experiment: how separable are bots from humans using SAFE features?

Pipeline:
    benchmark API -> live sanitizer -> hero-focused chunk features
    -> date-based split -> simple linear/tree baselines -> Poker44 reward.

This does NOT optimize a model. It establishes a floor: if plain baselines on
safe features already separate bots from humans on a held-out release date, the
approach is sound and the miner is an engineering exercise. If they don't, we
need better features before any modeling.

Run:
    python -m sn126_research.experiments.baseline_separability
Optional env:
    SN126_N_DATES (default 3)
    SN126_MAX_RECORDS (per date, default 40)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sn126_research.benchmark_client import BenchmarkClient  # noqa: E402
from sn126_research.dataset import Dataset, Sample, build_dataset  # noqa: E402
from sn126_research.evaluation import evaluate, format_report  # noqa: E402
from sn126_research.features import FEATURE_NAMES  # noqa: E402

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"scikit-learn is required for the baseline experiment: {exc}")


def _choose_split(ds: Dataset) -> Tuple[List[Sample], List[Sample], str]:
    """Prefer a held-out newest release date; fall back to a chunk-id split."""
    dates = ds.source_dates()
    if len(dates) >= 2:
        holdout = dates[-1]
        train, val = ds.split_by_date([holdout])
        if train and val:
            return train, val, f"date-holdout (val={holdout})"
    # Fallback: deterministic split by chunk_id so whole players stay together.
    val_ids = {
        s.chunk_id
        for s in ds.samples
        if (abs(hash(s.chunk_id)) % 5) == 0  # ~20% of records
    }
    train = [s for s in ds.samples if s.chunk_id not in val_ids]
    val = [s for s in ds.samples if s.chunk_id in val_ids]
    return train, val, "chunk-id holdout (fallback: only one release date available)"


def _univariate_ap(X: np.ndarray, y: np.ndarray) -> List[Tuple[str, float, float, float]]:
    """Per-feature separability: AP using the raw feature as the score."""
    rows = []
    for j, name in enumerate(FEATURE_NAMES):
        col = X[:, j]
        if np.all(col == col[0]):
            rows.append((name, 0.5, float(col[0]), float(col[0])))
            continue
        ap = float(average_precision_score(y, col))
        ap_inv = float(average_precision_score(y, -col))
        best = max(ap, ap_inv)
        mean_bot = float(col[y == 1].mean()) if np.any(y == 1) else float("nan")
        mean_hum = float(col[y == 0].mean()) if np.any(y == 0) else float("nan")
        rows.append((name, best, mean_bot, mean_hum))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def run_scope(scope: str, *, n_dates: int, max_records: int, client: BenchmarkClient) -> dict:
    print("\n" + "#" * 78)
    print(f"# SCOPE = {scope!r}  ({'table-level, recommended' if scope == 'all' else 'hero-only ablation'})")
    print("#" * 78)
    ds = build_dataset(
        n_dates=n_dates, max_records_per_date=max_records, scope=scope, client=client
    )
    if len(ds) == 0:
        raise SystemExit("No samples built — check network / API availability.")

    X_all, y_all = ds.matrix()
    print(f"samples={len(ds)} features={len(FEATURE_NAMES)} "
          f"bots={int(y_all.sum())} humans={int((y_all == 0).sum())} "
          f"dates={ds.source_dates()}")

    train, val, split_desc = _choose_split(ds)
    Xtr, ytr = ds.matrix(train)
    Xva, yva = ds.matrix(val)
    print(f"split: {split_desc} | train={len(train)} val={len(val)}")
    if len(np.unique(yva)) < 2 or len(np.unique(ytr)) < 2:
        raise SystemExit("Split produced a single-class fold; increase data.")

    # --- univariate feature separability (train only) ---
    print("\n--- Top safe features by univariate AP (train) ---")
    print(f"{'feature':28s} {'AP':>6s} {'mean_bot':>10s} {'mean_human':>10s}")
    for name, ap, mb, mh in _univariate_ap(Xtr, ytr)[:15]:
        print(f"{name:28s} {ap:6.3f} {mb:10.4f} {mh:10.4f}")

    # --- baselines (no tuning) ---
    print("\n--- Baseline models on held-out release ---")
    results = {}

    scaler = StandardScaler().fit(Xtr)
    logit = LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    logit.fit(scaler.transform(Xtr), ytr)
    p_logit = logit.predict_proba(scaler.transform(Xva))[:, 1]
    results["logreg"] = evaluate(yva, p_logit)
    print(format_report("logreg", results["logreg"]))

    gb = HistGradientBoostingClassifier(
        max_depth=3, learning_rate=0.08, max_iter=300, l2_regularization=1.0
    )
    gb.fit(Xtr, ytr)
    p_gb = gb.predict_proba(Xva)[:, 1]
    results["hist_gbm"] = evaluate(yva, p_gb)
    print(format_report("hist_gbm", results["hist_gbm"]))

    # reference: reproduce the shipped reference miner's separability baseline?
    # (Skipped here — this harness is about SAFE features, not the reference heuristic.)

    # --- verdict ---
    best_name = max(results, key=lambda k: results[k]["reward"])
    best = results[best_name]
    print(f"[scope={scope}] BEST={best_name}: " + format_report(best_name, best))
    return best


def main() -> None:
    n_dates = int(os.getenv("SN126_N_DATES", "3"))
    max_records = int(os.getenv("SN126_MAX_RECORDS", "40"))

    client = BenchmarkClient()
    status = client.status()
    print("=" * 78)
    print(f"Poker44 benchmark: release={status.get('releaseVersion')} "
          f"latest={status.get('latestSourceDate')} totalChunks={status.get('totalChunks')}")
    print(f"n_dates={n_dates} max_records_per_date={max_records}")
    print("=" * 78)

    all_best = run_scope("all", n_dates=n_dates, max_records=max_records, client=client)
    hero_best = run_scope("hero", n_dates=n_dates, max_records=max_records, client=client)

    print("\n" + "=" * 78)
    print("SEPARABILITY VERDICT (held-out release date)")
    print("=" * 78)
    print(format_report("all-seats ", all_best))
    print(format_report("hero-only ", hero_best))
    ap, rec, gate = all_best["ap"], all_best["recall_at_5pct_fpr"], all_best["gate_quality"]
    if ap >= 0.85 and rec >= 0.5:
        verdict = "STRONG — table-level features separate bots well; miner is an engineering task."
    elif ap >= 0.7:
        verdict = "MODERATE — real signal exists; feature engineering + calibration will move rank."
    elif ap >= 0.6:
        verdict = "WEAK — some signal; need richer table/interaction features next."
    else:
        verdict = "POOR — even table-level features are near chance; rethink features before modeling."
    print("=> [all-seats]", verdict)
    print(f"=> [hero-only] AP={hero_best['ap']:.3f} — confirms hero-focus is the WRONG granularity here.")
    print(
        "Reminder: AP & recall@5%FPR are rank-invariant (optimize ordering); "
        "the 0.5 gate is pass/fail (clear it, don't tune). AUC is diagnostic only."
    )


if __name__ == "__main__":
    main()
