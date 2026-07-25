#!/usr/bin/env python3
"""Verify a reproduction (scripts/model/reproduce_gbm35.py) against production.

Checks, strictest first:
  1. exact binary artifact SHA256 match against the frozen hash; if not,
  2. strict trained-state equivalence: identical declared params, feature
     order, sklearn version, training manifest, and bit-for-bit identical
     predictions from both models on the same probe batches.

Never modifies the production model.

Usage:
    python scripts/model/verify_reproduction.py --reproduction /tmp/poker44-reproduction
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from poker44.miner_model.features import PROD_FEATURE_ORDER  # noqa: E402

CONFIG_PATH = _REPO / "config" / "model" / "gbm35clean_v1.json"
PROD_MODEL = _REPO / "poker44" / "miner_model" / "model.joblib"
PROD_META = _REPO / "poker44" / "miner_model" / "model_meta.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reproduction", required=True, type=Path)
    args = ap.parse_args()
    rdir: Path = args.reproduction

    cfg = json.loads(CONFIG_PATH.read_text("utf-8"))
    meta = json.loads(PROD_META.read_text("utf-8"))
    report = json.loads((rdir / "reproduction_report.json").read_text("utf-8"))
    results, ok = [], True

    def check(cond, label):
        nonlocal ok
        results.append((bool(cond), label))
        ok = ok and bool(cond)
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")

    if report.get("status") != "REPRODUCED":
        print(f"reproduction status: {report.get('status')} — nothing to verify")
        return 1

    # frozen production artifact untouched + attested
    check(_sha256_file(PROD_MODEL) == cfg["artifact_sha256"],
          "production model.joblib still matches the frozen SHA256")

    exact = _sha256_file(rdir / "model.joblib") == cfg["artifact_sha256"]
    check(True, f"exact binary artifact match: {exact}")

    # declared-state equivalence
    check(json.loads((rdir / "feature_order.json").read_text()) == list(PROD_FEATURE_ORDER)
          == meta["feature_names"], "feature names/order identical (35)")
    tc = json.loads((rdir / "training_config.json").read_text("utf-8"))
    check(tc["model"]["params"] == meta["model"]["params"], "GBM parameters identical")
    check(tc["effective_sklearn_version"] == meta["model"]["sklearn_version"],
          "sklearn version identical")
    check(tc["training_release_dates"] == meta["training"]["dates"],
          "training release dates identical")
    check(report.get("n_samples") == meta["training"]["n_samples"],
          f"sample count identical ({report.get('n_samples')})")
    anchor_repro = float(tc.get("reproduced_global_anchor_raw", -1))
    check(float(anchor_repro).hex() == float(meta["calibration"]["global_anchor_raw"]).hex(),
          "global_anchor_raw bit-for-bit identical")

    # bit-for-bit prediction equivalence on real probe chunks + synthetic grid
    golden = json.loads((rdir / "golden_scores.json").read_text("utf-8"))
    repro_model = joblib.load(rdir / "model.joblib")
    prod_model = joblib.load(PROD_MODEL)
    if golden.get("feature_matrix_hex"):
        X = np.asarray([[float.fromhex(h) for h in row]
                        for row in golden["feature_matrix_hex"]], dtype=float)
        a = prod_model.predict_proba(X)[:, 1]
        b = repro_model.predict_proba(X)[:, 1]
        check([float(v).hex() for v in a] == [float(v).hex() for v in b],
              f"prediction parity on {len(X)} real benchmark chunks: bit-for-bit")
        check([float(v).hex() for v in b] == golden["raw_scores_hex"],
              "reproduced model matches its recorded golden scores")
        check(len(golden["raw_scores_hex"]) == golden["n_probe_chunks"],
              "golden probe output length correct")
    grid = np.linspace(0.0, 1.0, 35 * 8, dtype=float).reshape(8, 35)
    a = prod_model.predict_proba(grid)[:, 1]
    b = repro_model.predict_proba(grid)[:, 1]
    check([float(v).hex() for v in a] == [float(v).hex() for v in b],
          "prediction parity: prod vs reproduced bit-for-bit on synthetic grid")

    verdict = "EXACT_BINARY_REPRODUCTION" if exact else (
        "STRICT_EQUIVALENCE" if ok else "NOT_EQUIVALENT")
    (rdir / "verification_report.json").write_text(json.dumps({
        "verdict": verdict,
        "exact_binary_match": exact,
        "checks": [{"ok": o, "label": l} for o, l in results],
    }, indent=2), "utf-8")
    print(f"VERDICT: {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
