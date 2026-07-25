"""Train the validated 35-clean GBM and export production artifacts.

Outputs (into poker44/miner_model/):
  * model.joblib      -- HistGradientBoostingClassifier (validated params)
  * model_meta.json   -- feature version, exact 35-feature order, calibration, model version

Features are extracted through the PRODUCTION path (chunk_feature_vector on
to_live_group hands) so the saved model is trained on exactly what the miner
will feed it at inference. No new model architecture.

Run: python -m sn126_research.experiments.train_export
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from poker44.miner_model.features import PROD_FEATURE_ORDER, chunk_feature_vector  # noqa: E402
from sn126_research.benchmark_client import BenchmarkClient  # noqa: E402
from sn126_research.sanitizer import to_live_group  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[2] / "poker44" / "miner_model"
MODEL_PARAMS = dict(max_depth=3, learning_rate=0.08, max_iter=300, l2_regularization=1.0, random_state=0)
CALIB_P0 = 0.85
MIN_CALIB_BATCH = 8


def main() -> None:
    n_dates = int(os.getenv("SN126_N_DATES", "8"))
    max_records = int(os.getenv("SN126_MAX_RECORDS", "200"))
    client = BenchmarkClient()
    by_date = client.download_recent(n_dates=n_dates, max_records_per_date=max_records)
    records = [r for d in sorted(by_date) for r in by_date[d]]

    X, y, dates = [], [], []
    for rec in records:
        for gi, group in enumerate(rec.groups):
            if gi >= len(rec.ground_truth):
                continue
            live = to_live_group(group)               # sanitize+dedupe (benchmark path)
            if not live:
                continue
            X.append(chunk_feature_vector(live))       # SAME extractor the miner uses
            y.append(int(rec.ground_truth[gi]))
            dates.append(rec.source_date)
    X = np.asarray(X, float); y = np.asarray(y, int)
    print(f"training samples={len(y)} features={X.shape[1]} dates={sorted(set(dates))} "
          f"bots={int(y.sum())} humans={int((y==0).sum())}")
    assert X.shape[1] == len(PROD_FEATURE_ORDER) == 35, "feature count mismatch"

    model = HistGradientBoostingClassifier(**MODEL_PARAMS).fit(X, y)
    raw = model.predict_proba(X)[:, 1]
    global_anchor = float(np.quantile(raw, CALIB_P0))  # small-batch fallback cutoff

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, OUT_DIR / "model.joblib")

    meta = {
        "model_version": "gbm-35clean-v1",
        "feature_version": "sn126-35clean-v1",
        "feature_count": 35,
        "feature_names": list(PROD_FEATURE_ORDER),
        "model": {
            "type": "HistGradientBoostingClassifier",
            "params": MODEL_PARAMS,
            "sklearn_version": sklearn.__version__,
            "n_features_in": int(model.n_features_in_),
        },
        "calibration": {
            "method": "batch_percentile_anchor",
            "p0": CALIB_P0,
            "min_calib_batch": MIN_CALIB_BATCH,
            "global_anchor_raw": global_anchor,
        },
        "training": {
            "n_samples": int(len(y)),
            "dates": sorted(set(dates)),
        },
    }
    (OUT_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2), "utf-8")
    print(f"wrote {OUT_DIR/'model.joblib'} and model_meta.json")
    print(f"global_anchor_raw(p85)={global_anchor:.4f} sklearn={sklearn.__version__}")


if __name__ == "__main__":
    main()
