"""Read-only Synapse Intelligence analyzer.

Consumes the ALREADY-computed 35-feature matrix, raw/calibrated scores, timings,
fallback/exception state, and privacy-safe fingerprints from a PredictionDiagnostics.
It NEVER re-extracts features (no chunk_feature_vector / prepare_hand_for_miner
call), never parses raw hands, and never stores raw hands/actions/players/labels.

Output is a JSON-safe metadata+aggregate record. It never claims hidden labels:
the >=0.5 fraction is reported as `predicted_positive_fraction`, NOT bot fraction.
"""

from __future__ import annotations

import json
import math
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_PROFILE = Path(__file__).resolve().parent / "training_feature_profile.json"
_EPS = 1e-9
_PSI_EPS = 1e-4
_MIN_PSI_SAMPLES = 20          # below this, PSI/drift confidence is "low"
_REPETITION_CAP = 200_000      # bound the local fingerprint memory


def _pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if a.size else 0.0


def _entropy_hist(scores: np.ndarray, bins: int = 10) -> float:
    if scores.size == 0:
        return 0.0
    counts, _ = np.histogram(np.clip(scores, 0.0, 1.0), bins=bins, range=(0.0, 1.0))
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    h = -sum(x * math.log(x) for x in p if x > 0)
    return float(h / math.log(bins))  # normalized [0,1]


class SynapseAnalyzer:
    def __init__(self, profile_path: Path | str = _PROFILE) -> None:
        self.profile: Optional[Dict[str, Any]] = None
        try:
            self.profile = json.loads(Path(profile_path).read_text("utf-8"))
        except Exception:
            self.profile = None
        self._lock = threading.Lock()
        self._seen_requests: "OrderedDict[str, int]" = OrderedDict()
        self._seen_chunks: "OrderedDict[str, int]" = OrderedDict()

    # ---- repetition tracking (bounded, thread-safe) ----------------------
    def _bump(self, store: "OrderedDict[str, int]", key: str) -> int:
        c = store.get(key, 0) + 1
        store[key] = c
        store.move_to_end(key)
        if len(store) > _REPETITION_CAP:
            store.popitem(last=False)
        return c

    def _record_fingerprints(self, req_fp: Optional[str], chunk_fps: Optional[Sequence[str]]):
        req_seen = 0
        chunk_repeats = 0
        with self._lock:
            if req_fp:
                req_seen = self._bump(self._seen_requests, req_fp)
            for c in (chunk_fps or []):
                if self._bump(self._seen_chunks, c) > 1:
                    chunk_repeats += 1
        return req_seen, chunk_repeats

    # ---- drift ------------------------------------------------------------
    def _feature_drift(self, X: np.ndarray) -> Dict[str, Any]:
        if self.profile is None or X.size == 0:
            return {"available": False, "reason": "no_profile" if self.profile is None else "empty_request"}
        feats = self.profile["features"]
        names = self.profile["feature_names"]
        n = X.shape[0]
        confident = n >= _MIN_PSI_SAMPLES
        per_feature = {}
        for j, name in enumerate(names):
            tp = feats[name]
            col = X[:, j]
            live_median = float(np.median(col))
            live_mean = float(col.mean())
            median_shift = (live_median - tp["median"]) / max(abs(tp["iqr"]), _EPS)
            mean_shift = (live_mean - tp["mean"]) / max(tp["std"], _EPS)
            oor = float(np.mean((col < tp["min"]) | (col > tp["max"])))
            psi = None
            if confident and tp.get("hist_bin_edges") and len(tp["hist_bin_edges"]) >= 2:
                counts, _ = np.histogram(col, bins=tp["hist_bin_edges"])
                lp = np.clip(counts / max(counts.sum(), 1), _PSI_EPS, None)
                tprob = np.clip(np.asarray(tp["hist_bin_probs"], float), _PSI_EPS, None)
                lp = lp / lp.sum(); tprob = tprob / tprob.sum()
                psi = float(np.sum((lp - tprob) * np.log(lp / tprob)))
            per_feature[name] = {
                "median_shift_over_iqr": round(median_shift, 5),
                "standardized_mean_shift": round(mean_shift, 5),
                "psi": round(psi, 5) if psi is not None else None,
                "out_of_range_fraction": round(oor, 5),
            }
        ranked = sorted(per_feature.items(),
                        key=lambda kv: abs(kv[1]["standardized_mean_shift"]), reverse=True)
        top10 = [{"feature": k, **v} for k, v in ranked[:10]]
        psis = [v["psi"] for v in per_feature.values() if v["psi"] is not None]
        max_psi = max(psis) if psis else None
        max_abs_mean_shift = max(abs(v["standardized_mean_shift"]) for v in per_feature.values())
        if not confident:
            severity = "normal"  # do not over-claim on small samples
        elif (max_psi is not None and max_psi >= 0.25) or max_abs_mean_shift >= 3.0:
            severity = "critical"
        elif (max_psi is not None and max_psi >= 0.10) or max_abs_mean_shift >= 1.5:
            severity = "warning"
        else:
            severity = "normal"
        return {
            "available": True,
            "confidence": "high" if n >= 50 else ("medium" if confident else "low"),
            "n_chunks": int(n),
            "max_psi": round(max_psi, 5) if max_psi is not None else None,
            "max_abs_standardized_mean_shift": round(max_abs_mean_shift, 5),
            "severity": severity,
            "top10_drifting_features": top10,
        }

    # ---- main -------------------------------------------------------------
    def analyze(self, diagnostics, *, hand_counts: Optional[Sequence[int]] = None,
                caller_hotkey: Optional[str] = None, manifest_digest: Optional[str] = None,
                model_version: Optional[str] = None, feature_version: Optional[str] = None,
                request_fingerprint: Optional[str] = None,
                chunk_fingerprints: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        """Build one privacy-safe analyzer record from a PredictionDiagnostics. Never raises."""
        try:
            return self._analyze(diagnostics, hand_counts, caller_hotkey, manifest_digest,
                                 model_version, feature_version, request_fingerprint, chunk_fingerprints)
        except Exception as exc:  # analyzer must never break the caller
            return {"ts_utc": datetime.now(timezone.utc).isoformat(),
                    "analyzer_error": type(exc).__name__,
                    "request_id": getattr(diagnostics, "request_id", None)}

    def _analyze(self, diag, hand_counts, caller_hotkey, manifest_digest, model_version,
                 feature_version, request_fingerprint, chunk_fingerprints) -> Dict[str, Any]:
        X = np.asarray(diag.feature_matrix, float) if diag.feature_matrix is not None else np.empty((0, 35))
        cal = np.asarray(diag.final_scores, float)
        raw = np.asarray(diag.raw_scores, float) if diag.raw_scores else np.empty(0)
        hc = [int(x) for x in (hand_counts or [])]
        n_chunks = len(diag.final_scores)

        # A. request shape
        hc_arr = np.asarray(hc, float) if hc else np.empty(0)
        malformed = int(np.sum(np.all(X == 0.0, axis=1))) if X.size else 0
        shape = {
            "chunk_count": n_chunks,
            "total_hands": int(sum(hc)) if hc else None,
            "hands_per_chunk": ({"min": int(hc_arr.min()), "mean": round(float(hc_arr.mean()), 3),
                                 "median": float(np.median(hc_arr)), "p90": _pct(hc_arr, 90),
                                 "max": int(hc_arr.max())} if hc else None),
            "empty_chunk_count": int(sum(1 for x in hc if x == 0)) if hc else 0,
            "malformed_chunk_count": malformed,
        }

        # B. feature summaries (35), from the ALREADY-computed matrix
        feature_summaries = None
        if X.size:
            fs = {}
            names = self.profile["feature_names"] if self.profile else [f"f{j}" for j in range(X.shape[1])]
            for j, name in enumerate(names[:X.shape[1]]):
                col = X[:, j]
                invalid = int(np.sum(~np.isfinite(col)))
                fs[name] = {"mean": round(float(np.nanmean(col)), 5), "std": round(float(np.nanstd(col)), 5),
                            "median": round(float(np.median(col)), 5), "p10": round(_pct(col, 10), 5),
                            "p90": round(_pct(col, 90), 5), "min": round(float(np.min(col)), 5),
                            "max": round(float(np.max(col)), 5), "invalid_count": invalid,
                            "invalid_rate": round(invalid / col.size, 5)}
            feature_summaries = fs

        # C. drift
        drift = self._feature_drift(X)

        # D. score diagnostics
        def _stats(a):
            if a.size == 0:
                return None
            return {"min": round(float(a.min()), 6), "mean": round(float(a.mean()), 6),
                    "median": round(float(np.median(a)), 6), "std": round(float(a.std()), 6),
                    "p10": round(_pct(a, 10), 6), "p90": round(_pct(a, 90), 6), "max": round(float(a.max()), 6)}
        srt = np.sort(cal)[::-1] if cal.size else np.empty(0)
        top = float(srt[0]) if srt.size >= 1 else None
        second = float(srt[1]) if srt.size >= 2 else None
        ties = int(cal.size - np.unique(np.round(cal, 6)).size) if cal.size else 0
        scores = {
            "raw_score_stats": _stats(raw),
            "calibrated_score_stats": _stats(cal),
            "n_ge_0.5": int(np.sum(cal >= 0.5)) if cal.size else 0,
            # NOTE: this is the model's predicted-positive fraction, NOT the true bot fraction.
            "predicted_positive_fraction": round(float(np.mean(cal >= 0.5)), 6) if cal.size else None,
            "top_score": round(top, 6) if top is not None else None,
            "second_highest_score": round(second, 6) if second is not None else None,
            "top_score_margin": round(top - second, 6) if (top is not None and second is not None) else None,
            "calibrated_score_entropy": round(_entropy_hist(cal), 6),
            "rank_ties": ties,
        }

        # F. repetition (privacy-safe)
        req_seen, chunk_repeats = self._record_fingerprints(request_fingerprint, chunk_fingerprints)

        # E. operational + assemble
        return {
            "ts_utc": datetime.now(timezone.utc).isoformat(),
            "request_id": diag.request_id,
            "caller_hotkey": str(caller_hotkey) if caller_hotkey else None,
            "model_version": model_version, "feature_version": feature_version,
            "manifest_digest": manifest_digest,
            "fallback_used": bool(diag.fallback_used), "exception_type": diag.exception_type,
            "timings_ms": {"feature": round(diag.feature_duration_ms, 3),
                           "model": round(diag.model_duration_ms, 3),
                           "calibration": round(diag.calibration_duration_ms, 3),
                           "total": round(diag.total_duration_ms, 3)},
            "request_shape": shape,
            "feature_summaries": feature_summaries,
            "feature_drift": drift,
            "score_diagnostics": scores,
            "repetition": {"request_fingerprint": request_fingerprint,
                           "chunk_fingerprint_count": len(chunk_fingerprints) if chunk_fingerprints else 0,
                           "request_seen_count_local": req_seen,
                           "repeated_chunk_count_local": chunk_repeats},
        }
