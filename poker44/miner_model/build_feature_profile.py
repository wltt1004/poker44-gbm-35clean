"""Generate poker44/miner_model/training_feature_profile.json.

Built ONLY from public Poker44 benchmark releases (the no-auth API below),
projected through the exact production sanitizer-equivalent path
(prepare_hand_for_miner + dedupe_padding) and the frozen 35-feature extractor.
Stores per-feature distribution statistics + fixed histogram bins for
drift/PSI. NO raw examples and NO labels are stored.

Standalone: this offline tool depends only on this repository + numpy.

Run: python -m poker44.miner_model.build_feature_profile
Env: SN126_N_DATES, SN126_MAX_RECORDS, SN126_BENCHMARK_CACHE (optional record cache dir)
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from poker44.miner_model.feature_core.sanitizer_core import dedupe_padding
from poker44.miner_model.features import PROD_FEATURE_ORDER, chunk_feature_vector
from poker44.miner_model.predictor import Poker44Predictor
from poker44.validator.payload_view import prepare_hand_for_miner

_OUT = Path(__file__).resolve().parent / "training_feature_profile.json"
_NBINS = 20
_API = "https://api.poker44.net/api/v1/benchmark"


def to_live_group(group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Project one benchmark group through the official live sanitizer + dedupe."""
    return [dedupe_padding(prepare_hand_for_miner(h)) for h in group]


def _api_get(path: str, **params: Any) -> Any:
    url = f"{_API}{path}"
    query = {k: v for k, v in params.items() if v is not None}
    if query:
        url = f"{url}?{urllib.parse.urlencode(query)}"
    req = urllib.request.Request(url, headers={"accept": "application/json",
                                               "user-agent": "poker44-feature-profile/1.0"})
    with urllib.request.urlopen(req, timeout=90) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload.get("data", payload) if isinstance(payload, dict) else payload


def _recent_source_dates(n: int) -> List[str]:
    data = _api_get("/releases", limit=max(n * 2, 10))
    rels = data.get("releases") or data.get("items") or [] if isinstance(data, dict) else (data or [])
    dates: List[str] = []
    for rel in rels:
        d = str(rel.get("sourceDate") or "")
        if d and d not in dates:
            dates.append(d)
        if len(dates) >= n:
            break
    if not dates:
        dates = [str(_api_get("")["latestSourceDate"])]
    return dates


def _iter_records(source_date: str, max_records: Optional[int]) -> Iterator[Dict[str, Any]]:
    """Yield raw benchmark chunk-records, using a local cache dir when provided."""
    cache_env = os.getenv("SN126_BENCHMARK_CACHE")
    if cache_env:
        mpath = Path(cache_env) / "manifests" / f"{source_date}__all.json"
        if mpath.exists():
            emitted = 0
            for h in json.loads(mpath.read_text("utf-8")):
                rpath = Path(cache_env) / "records" / f"{h}.json"
                if rpath.exists():
                    yield json.loads(rpath.read_text("utf-8"))
                    emitted += 1
                    if max_records is not None and emitted >= max_records:
                        return
            return
    cursor: Optional[str] = None
    emitted = 0
    while True:
        data = _api_get("/chunks", sourceDate=source_date, limit=24, cursor=cursor)
        for rec in (data.get("chunks", []) if isinstance(data, dict) else []):
            yield rec
            emitted += 1
            if max_records is not None and emitted >= max_records:
                return
        cursor = data.get("nextCursor") if isinstance(data, dict) else None
        if not cursor:
            return


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
    dates = sorted(_recent_source_dates(n_dates))

    rows = []
    for date in dates:
        for rec in _iter_records(date, max_records):
            groups = list(rec.get("chunks") or [])
            truth = list(rec.get("groundTruth") or [])
            for gi, group in enumerate(groups):
                if gi >= len(truth):
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
