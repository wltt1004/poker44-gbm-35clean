#!/usr/bin/env python3
"""Reproduce the served gbm-35clean-v1 model from PUBLIC benchmark data only.

Public, standalone, deterministic: uses this repository's production feature
code and the recipe pinned in config/model/gbm35clean_v1.json. Never imports
private research, never uses validator-only labels, never touches the served
poker44/miner_model/model.joblib.

Usage:
    python scripts/model/reproduce_gbm35.py \
        --output /tmp/poker44-reproduction \
        --cache  /tmp/poker44-benchmark-cache

--cache holds/receives the content-addressed benchmark download
(records/<chunkHash>.json + manifests/<date>__all.json). If a declared release
date is already cached it is used offline; otherwise it is fetched from the
public API. NOTE: the public benchmark is a rolling window — if a declared
training date has cycled out of the API and is not cached, exact reproduction
is impossible and the script reports which dates are unavailable.

Outputs (in --output):
    training_config.json     exact recipe used
    release_manifest.json    per-date record chunkHashes + SHA256 checksums
    feature_order.json       the frozen 35-name order
    model.joblib             the reproduced artifact
    reproduction_report.json artifact SHA256, sample counts, golden scores
    golden_scores.json       float-hex predictions on fixed probe batches
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

import numpy as np  # noqa: E402

from poker44.miner_model.build_feature_profile import _api_get, to_live_group  # noqa: E402
from poker44.miner_model.features import PROD_FEATURE_ORDER, chunk_feature_vector  # noqa: E402

CONFIG_PATH = _REPO / "config" / "model" / "gbm35clean_v1.json"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fetch_date(date: str, cache: Path, max_records: int | None) -> list:
    """Return raw records for one release date, cache-first, API fallback."""
    manifest = cache / "manifests" / f"{date}__all.json"
    records_dir = cache / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    if manifest.exists():
        out = []
        for h in json.loads(manifest.read_text("utf-8")):
            p = records_dir / f"{h}.json"
            if p.exists():
                out.append(json.loads(p.read_text("utf-8")))
            if max_records is not None and len(out) >= max_records:
                break
        return out

    hashes, out, cursor, emitted = [], [], None, 0
    while True:
        data = _api_get("/chunks", sourceDate=date, limit=24, cursor=cursor)
        recs = data.get("chunks", []) if isinstance(data, dict) else []
        for rec in recs:
            ch = str(rec.get("chunkHash") or "")
            if not ch:
                continue
            (records_dir / f"{ch}.json").write_text(
                json.dumps(rec, sort_keys=True), "utf-8")
            hashes.append(ch)
            out.append(rec)
            emitted += 1
            if max_records is not None and emitted >= max_records:
                manifest.write_text(json.dumps(hashes), "utf-8")
                return out
        cursor = data.get("nextCursor") if isinstance(data, dict) else None
        if not cursor:
            break
    manifest.write_text(json.dumps(hashes), "utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--cache", required=True, type=Path)
    args = ap.parse_args()

    import sklearn
    from sklearn.ensemble import HistGradientBoostingClassifier

    cfg = json.loads(CONFIG_PATH.read_text("utf-8"))
    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir.resolve() == (_REPO / "poker44" / "miner_model").resolve():
        raise SystemExit("refusing to write into the production model directory")

    dates = list(cfg["training_release_dates"])
    max_records = cfg.get("max_records_per_date")
    params = dict(cfg["model"]["params"])
    p0 = float(cfg["calibration"]["anchor_quantile"])

    # ---- gather + checksum the declared releases ---------------------------
    release_manifest, unavailable = {}, []
    by_date = {}
    for date in sorted(dates):
        recs = _fetch_date(date, args.cache, max_records)
        if not recs:
            unavailable.append(date)
            continue
        by_date[date] = recs
        release_manifest[date] = {
            "record_count": len(recs),
            "chunk_hashes": [str(r.get("chunkHash") or "") for r in recs],
            "record_sha256": [
                _sha256_bytes(json.dumps(r, sort_keys=True).encode("utf-8")) for r in recs
            ],
        }
    if unavailable:
        (out_dir / "reproduction_report.json").write_text(json.dumps({
            "status": "BLOCKED_MISSING_RELEASES",
            "unavailable_dates": unavailable,
            "reason": "public benchmark is a rolling window; these declared training "
                      "dates are neither cached nor available from the API",
        }, indent=2), "utf-8")
        print(f"BLOCKED: missing release dates {unavailable}")
        return 1

    # ---- exact training-set assembly (row order pinned by config) ----------
    X, y = [], []
    for date in sorted(dates):
        for rec in by_date[date]:
            groups = list(rec.get("chunks") or [])
            truth = list(rec.get("groundTruth") or [])
            for gi, group in enumerate(groups):
                if gi >= len(truth):
                    continue
                live = to_live_group(group)
                if not live:
                    continue
                X.append(chunk_feature_vector(live))
                y.append(int(truth[gi]))
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    assert X.shape[1] == len(PROD_FEATURE_ORDER) == 35, "feature contract violated"

    model = HistGradientBoostingClassifier(**params).fit(X, y)
    raw = model.predict_proba(X)[:, 1]
    anchor = float(np.quantile(raw, p0))

    import joblib
    joblib.dump(model, out_dir / "model.joblib")
    artifact_sha = _sha256_file(out_dir / "model.joblib")

    # ---- golden probe scores (fixed, deterministic batches) ----------------
    first_date = sorted(dates)[0]
    probe_chunks = []
    for rec in by_date[first_date]:
        for group in rec.get("chunks") or []:
            live = to_live_group(group)
            if live:
                probe_chunks.append(live)
            if len(probe_chunks) >= 16:
                break
        if len(probe_chunks) >= 16:
            break
    probe_X = np.asarray([chunk_feature_vector(c) for c in probe_chunks], float)
    probe_raw = model.predict_proba(probe_X)[:, 1]
    golden = {
        "probe_source_date": first_date,
        "n_probe_chunks": len(probe_chunks),
        "raw_scores_hex": [float(v).hex() for v in probe_raw],
        "feature_matrix_hex": [[float(v).hex() for v in row] for row in probe_X.tolist()],
    }

    (out_dir / "training_config.json").write_text(json.dumps({
        **cfg, "effective_sklearn_version": sklearn.__version__,
        "reproduced_global_anchor_raw": anchor,
    }, indent=2), "utf-8")
    (out_dir / "release_manifest.json").write_text(
        json.dumps(release_manifest, indent=1, sort_keys=True), "utf-8")
    (out_dir / "feature_order.json").write_text(
        json.dumps(list(PROD_FEATURE_ORDER), indent=1), "utf-8")
    (out_dir / "golden_scores.json").write_text(json.dumps(golden, indent=1), "utf-8")
    (out_dir / "reproduction_report.json").write_text(json.dumps({
        "status": "REPRODUCED",
        "n_samples": int(len(y)),
        "n_bots": int(y.sum()),
        "expected_n_samples": cfg.get("expected_n_samples"),
        "artifact_sha256": artifact_sha,
        "frozen_artifact_sha256": cfg["artifact_sha256"],
        "exact_binary_match": artifact_sha == cfg["artifact_sha256"],
        "sklearn_version": sklearn.__version__,
        "global_anchor_raw": anchor,
    }, indent=2), "utf-8")

    print(f"reproduced: samples={len(y)} artifact_sha256={artifact_sha}")
    print(f"exact_binary_match={artifact_sha == cfg['artifact_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
