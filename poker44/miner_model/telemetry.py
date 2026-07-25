"""Low-overhead, failure-isolated miner telemetry (metadata + aggregate stats only).

NEVER stores raw hands, player identifiers, full action sequences, private cards,
validator labels, or secrets/wallet data. It logs only chunk counts, hand-count
distribution, score *statistics*, timing, fallback flag, and version metadata.

Guarantees:
  * telemetry failure can never raise into the caller (every path is guarded);
  * JSONL output with size-based rotation;
  * enabled by default; disable via POKER44_TELEMETRY_ENABLED=0 (or
    POKER44_TELEMETRY_DISABLED=1);
  * negligible latency (a few list reductions + one appended line).
"""

from __future__ import annotations

import json
import math
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_TRUE = {"1", "true", "yes", "on"}


def _num_stats(arr: Optional[Sequence[Any]]) -> Optional[Dict[str, float]]:
    if arr is None:
        return None
    try:
        vals = [float(x) for x in arr]
    except Exception:
        return None
    if not vals:
        return None
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return {"min": round(min(vals), 6), "mean": round(mean, 6),
            "max": round(max(vals), 6), "std": round(math.sqrt(var), 6)}


def _hand_count_dist(hand_counts: Optional[Sequence[Any]]) -> Optional[Dict[str, float]]:
    if hand_counts is None:
        return None
    try:
        hc = [int(x) for x in hand_counts]
    except Exception:
        return None
    if not hc:
        return None
    n = len(hc)
    return {"n_chunks": n, "min": min(hc), "mean": round(sum(hc) / n, 3),
            "max": max(hc), "total": sum(hc)}


class MinerTelemetry:
    def __init__(self, path: Optional[Path | str] = None, *, enabled: Optional[bool] = None,
                 max_bytes: int = 5_000_000, backups: int = 3) -> None:
        self.enabled = self._resolve_enabled(enabled)
        default = Path(__file__).resolve().parents[2] / "logs" / "miner_telemetry.jsonl"
        self.path = Path(path or os.getenv("POKER44_TELEMETRY_PATH") or default)
        try:
            self.max_bytes = int(os.getenv("POKER44_TELEMETRY_MAX_BYTES", str(max_bytes)))
        except Exception:
            self.max_bytes = max_bytes
        self.backups = max(1, int(backups))
        self._lock = threading.Lock()
        if self.enabled:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                self.enabled = False  # cannot create dir -> silently disable

    @staticmethod
    def _resolve_enabled(enabled: Optional[bool]) -> bool:
        if enabled is not None:
            return bool(enabled)
        raw = os.getenv("POKER44_TELEMETRY_ENABLED")
        if raw is not None:
            return raw.strip().lower() in _TRUE
        if os.getenv("POKER44_TELEMETRY_DISABLED", "").strip().lower() in _TRUE:
            return False
        return True  # default enabled

    def log_inference(self, *, chunk_count: Optional[int] = None,
                      hand_counts: Optional[Sequence[Any]] = None,
                      raw_scores: Optional[Sequence[Any]] = None,
                      calibrated_scores: Optional[Sequence[Any]] = None,
                      duration_s: Optional[float] = None, fallback_used: bool = False,
                      caller_hotkey: Optional[str] = None, model_version: Optional[str] = None,
                      feature_version: Optional[str] = None, manifest_digest: Optional[str] = None,
                      exception_type: Optional[str] = None,
                      request_fingerprint: Optional[str] = None,
                      chunk_fingerprints: Optional[Sequence[str]] = None) -> None:
        """Write one metadata-only record. Never raises."""
        if not self.enabled:
            return
        try:
            pos = None
            if calibrated_scores is not None:
                try:
                    vals = [float(x) for x in calibrated_scores]
                    npos = sum(1 for v in vals if v >= 0.5)
                    pos = {"n_ge_0.5": npos,
                           "frac_ge_0.5": round(npos / len(vals), 6) if vals else 0.0}
                except Exception:
                    pos = None
            record = {
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "caller_hotkey": str(caller_hotkey) if caller_hotkey else None,
                "chunk_count": int(chunk_count) if chunk_count is not None else None,
                "hand_count_dist": _hand_count_dist(hand_counts),
                "raw_score_stats": _num_stats(raw_scores),
                "calibrated_score_stats": _num_stats(calibrated_scores),
                "positive_at_0.5": pos,
                "inference_seconds": round(float(duration_s), 6) if duration_s is not None else None,
                "fallback_used": bool(fallback_used),
                "model_version": model_version,
                "feature_version": feature_version,
                "manifest_digest": manifest_digest,
                "exception_type": exception_type,
                # privacy-safe one-way hashes of the ALREADY-sanitized payload (no raw
                # hands stored). For later comparison with public benchmark chunkHash.
                "request_fingerprint": str(request_fingerprint) if request_fingerprint else None,
                "chunk_fingerprints": ([str(c) for c in chunk_fingerprints]
                                       if chunk_fingerprints else None),
            }
            self._write(record)
        except Exception:
            return  # telemetry must NEVER break inference

    def _write(self, record: Dict[str, Any]) -> None:
        try:
            line = json.dumps(record, separators=(",", ":"), ensure_ascii=True)
        except Exception:
            return
        with self._lock:
            try:
                self._rotate_if_needed()
                with open(self.path, "a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except Exception:
                return

    def _rotate_if_needed(self) -> None:
        try:
            if self.path.exists() and self.path.stat().st_size >= self.max_bytes:
                for i in range(self.backups - 1, 0, -1):
                    src = Path(str(self.path) + f".{i}")
                    dst = Path(str(self.path) + f".{i + 1}")
                    if src.exists():
                        src.replace(dst)
                self.path.replace(Path(str(self.path) + ".1"))
        except Exception:
            return
