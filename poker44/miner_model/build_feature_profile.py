"""Generate poker44/miner_model/training_feature_profile.json.

Built ONLY from public Poker44 benchmark releases, projected through the exact
production sanitizer-equivalent path (to_live_group) and the frozen 35-feature
extractor. Stores per-feature distribution statistics + fixed histogram bins for
drift/PSI. NO raw examples and NO labels are stored.

Run: python -m poker44.miner_model.build_feature_profile
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from poker44.miner_model.features import PROD_FEATURE_ORDER, chunk_feature_vector
from poker44.miner_model.predictor import Poker44Predictor
from sn126_research.benchmark_client import BenchmarkClient
from sn126_research.sanitizer import to_live_group

_OUT = Path(__file__).resolve().parent / "training_feature_profile.json"
_NBINS = 20


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _feature_stats(col: np.ndarray) -> dict:
    col = np.asarray(col, float)
    qs = {f"p{p:02d}": float(np.percentile(col, p)) for p in (1, 5, 10, 25, 75, 90, 95, 99)}
    lo, hi = float(col.min()), float(col.max())
    if hi > lo:
        edges = list(np.linspace(lo, hi, _NBINS + 1))
    else:  # constant feature
        edges = [lo - 1e-9, lo + 1e-9]
    counts, _ = np.histogram(col, bins=edges)
    probs = list((counts / max(counts.sum(), 1)).astype(float))
    return {
        "count": int(col.size), "mean": float(col.mean()), "std": float(col.std()),
        "median": float(np.median(col)), "iqr": float(qs["p75"] - qs["p25"]),
        **qs, "min": lo, "max": hi,
        "hist_bin_edges": [float(e) for e in edges], "hist_bin_probs": [float(p) for p in probs],
    }


def main() -> None:
    n_dates = int(os.getenv("SN126_N_DATES", "8"))
    max_records = int(os.getenv("SN126_MAX_RECORDS", "200"))
    predictor = Poker44Predictor()
    client = BenchmarkClient()
    by_date = client.download_recent(n_dates=n_dates, max_records_per_date=max_records)
    dates = sorted(by_date)

    rows = []
    for date in dates:
        for rec in by_date[date]:
            for gi, group in enumerate(rec.groups):
                if gi >= len(rec.ground_truth):
                    continue
                live = to_live_group(group)                 # production sanitizer-equivalent
                if not live:
                    continue
                rows.append(chunk_feature_vector(live))     # frozen 35-feature extractor
    X = np.asarray(rows, float)
    assert X.shape[1] == len(PROD_FEATURE_ORDER) == 35, "feature count mismatch"
    print(f"profiled {X.shape[0]} chunks x {X.shape[1]} features over {dates}")

    features = {}
    for j, name in enumerate(PROD_FEATURE_ORDER):
        features[name] = {"index": j, **_feature_stats(X[:, j])}

    profile = {
        "model_version": predictor.model_version,
        "feature_version": predictor.feature_version,
        "feature_names": list(PROD_FEATURE_ORDER),
        "source_release_dates": dates,
        "n_chunks_profiled": int(X.shape[0]),
        "n_bins": _NBINS,
        "profile_generation_commit": _git_commit(),
        "features": features,
    }
    # profile sha256 over the content excluding the sha256 field itself
    digest = hashlib.sha256(json.dumps(profile, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    profile["profile_sha256"] = digest
    _OUT.write_text(json.dumps(profile, indent=2), "utf-8")
    print(f"wrote {_OUT} | sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
