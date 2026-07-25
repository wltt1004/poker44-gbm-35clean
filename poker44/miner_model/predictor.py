"""SN126 model inference layer: chunks -> features -> GBM -> calibrated risk_scores.

Loads the serialized GBM and metadata ONCE at construction, verifies the feature
contract, then for each request:
    chunk_feature_vector() -> GBM.predict_proba() -> batch percentile calibration.

`predict_detailed()` returns a request-scoped, immutable PredictionDiagnostics so
that concurrent validator queries never cross diagnostics. `predict()` is a thin
compatibility wrapper returning diagnostics.final_scores -- its output is
bit-for-bit identical to the previous implementation.

Framework-agnostic (no bittensor import) so it is unit-testable in isolation.
Never raises from predict()/predict_detailed(): on any inference error it returns
a safe, correctly-sized fallback so the miner cannot emit a wrong-length response.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np

from .calibration import batch_percentile_calibrate
from .features import PROD_FEATURE_ORDER, chunk_feature_vector

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).resolve().parent
_DEFAULT_MODEL = _MODEL_DIR / "model.joblib"
_DEFAULT_META = _MODEL_DIR / "model_meta.json"
_EPS = 1e-9


@dataclass(frozen=True)
class PredictionDiagnostics:
    """Immutable, request-scoped diagnostics. Never shared between requests."""

    request_id: str
    final_scores: Tuple[float, ...]
    raw_scores: Optional[Tuple[float, ...]]
    feature_matrix: np.ndarray            # read-only (n_chunks x 35); empty on fallback
    fallback_used: bool
    exception_type: Optional[str]
    feature_duration_ms: float
    model_duration_ms: float
    calibration_duration_ms: float
    total_duration_ms: float


class Poker44Predictor:
    def __init__(
        self,
        model_path: Path | str = _DEFAULT_MODEL,
        meta_path: Path | str = _DEFAULT_META,
    ) -> None:
        self.meta: Dict[str, Any] = json.loads(Path(meta_path).read_text("utf-8"))
        self.model = joblib.load(model_path)

        # ---- verify the feature contract (fail fast at load, not at inference) ----
        expected = list(self.meta.get("feature_names", []))
        if expected != list(PROD_FEATURE_ORDER):
            raise ValueError(
                "Feature-order mismatch between model_meta.json and PROD_FEATURE_ORDER "
                f"(meta n={len(expected)}, code n={len(PROD_FEATURE_ORDER)}). "
                "The model was trained on a different feature order; retrain/export."
            )
        n_in = int(getattr(self.model, "n_features_in_", len(PROD_FEATURE_ORDER)))
        if n_in != len(PROD_FEATURE_ORDER):
            raise ValueError(
                f"Model expects {n_in} features but PROD_FEATURE_ORDER has {len(PROD_FEATURE_ORDER)}."
            )

        calib = self.meta.get("calibration", {})
        self.p0 = float(calib.get("p0", 0.85))
        self.min_calib_batch = int(calib.get("min_calib_batch", 8))
        self.global_anchor_raw = float(calib.get("global_anchor_raw", 0.5))
        self.model_version = str(self.meta.get("model_version", "unknown"))
        self.feature_version = str(self.meta.get("feature_version", "unknown"))

        # DEPRECATED best-effort last-call snapshot (NOT concurrency-safe). Prefer
        # predict_detailed(); retained only for backward compatibility.
        self.last_raw_scores = None
        self.last_calibrated_scores = None
        self.last_fallback_used = False
        self.last_exception_type = None

    # ---- calibration ------------------------------------------------------
    def _calibrate(self, raw: np.ndarray) -> np.ndarray:
        """Batch percentile anchor: place 0.5 at the p0-th percentile of THIS request.
        Rank-invariant (preserves AP/recall@FPR); only moves the 0.5 gate.
        """
        return batch_percentile_calibrate(
            raw, p0=self.p0, min_calib_batch=self.min_calib_batch,
            global_anchor_raw=self.global_anchor_raw,
        )

    def _empty_matrix(self) -> np.ndarray:
        m = np.empty((0, len(PROD_FEATURE_ORDER)), dtype=float)
        m.setflags(write=False)
        return m

    # ---- inference (request-scoped) ---------------------------------------
    def predict_detailed(
        self,
        chunks: Sequence[List[Dict[str, Any]]],
        *,
        fallback: Optional[Callable[[List[Dict[str, Any]]], float]] = None,
    ) -> PredictionDiagnostics:
        """Compute scores + immutable per-request diagnostics. Never raises."""
        rid = uuid.uuid4().hex
        n = len(chunks)
        t_start = time.perf_counter()
        if n == 0:
            return PredictionDiagnostics(rid, (), (), self._empty_matrix(), False, None,
                                         0.0, 0.0, 0.0, 0.0)
        self.last_fallback_used = False
        self.last_exception_type = None
        try:
            tf0 = time.perf_counter()
            X = np.asarray([chunk_feature_vector(c) for c in chunks], dtype=float)
            tf1 = time.perf_counter()
            if X.shape != (n, len(PROD_FEATURE_ORDER)):
                raise ValueError(f"feature matrix shape {X.shape} != ({n}, {len(PROD_FEATURE_ORDER)})")
            raw = self.model.predict_proba(X)[:, 1]
            tm1 = time.perf_counter()
            scores = self._calibrate(raw)
            out = [float(min(1.0, max(0.0, s))) for s in scores]   # IDENTICAL to legacy predict()
            tc1 = time.perf_counter()
            if len(out) != n:
                raise ValueError("score length mismatch")
            self.last_raw_scores = raw
            self.last_calibrated_scores = np.asarray(out, dtype=float)
            xro = X.copy()
            xro.setflags(write=False)
            return PredictionDiagnostics(
                request_id=rid, final_scores=tuple(out),
                raw_scores=tuple(float(r) for r in raw), feature_matrix=xro,
                fallback_used=False, exception_type=None,
                feature_duration_ms=(tf1 - tf0) * 1000.0,
                model_duration_ms=(tm1 - tf1) * 1000.0,
                calibration_duration_ms=(tc1 - tm1) * 1000.0,
                total_duration_ms=(time.perf_counter() - t_start) * 1000.0,
            )
        except Exception as exc:  # noqa: BLE001 - inference must never crash the axon
            self.last_fallback_used = True
            self.last_exception_type = type(exc).__name__
            self.last_raw_scores = None
            self.last_calibrated_scores = None
            logger.warning("Poker44Predictor.predict_detailed failed (%s); using safe fallback.", exc)
            fb = self._safe_fallback(chunks, fallback)
            return PredictionDiagnostics(
                request_id=rid, final_scores=tuple(fb), raw_scores=None,
                feature_matrix=self._empty_matrix(), fallback_used=True,
                exception_type=type(exc).__name__, feature_duration_ms=0.0,
                model_duration_ms=0.0, calibration_duration_ms=0.0,
                total_duration_ms=(time.perf_counter() - t_start) * 1000.0,
            )

    def predict(
        self,
        chunks: Sequence[List[Dict[str, Any]]],
        *,
        fallback: Optional[Callable[[List[Dict[str, Any]]], float]] = None,
    ) -> List[float]:
        """Compatibility wrapper: returns exactly diagnostics.final_scores as a list."""
        return list(self.predict_detailed(chunks, fallback=fallback).final_scores)

    @staticmethod
    def _safe_fallback(
        chunks: Sequence[List[Dict[str, Any]]],
        fallback: Optional[Callable[[List[Dict[str, Any]]], float]],
    ) -> List[float]:
        n = len(chunks)
        if fallback is not None:
            try:
                out = [float(min(1.0, max(0.0, fallback(c)))) for c in chunks]
                if len(out) == n:
                    return out
            except Exception as exc:  # noqa: BLE001
                logger.warning("fallback scorer failed (%s); emitting neutral 0.5.", exc)
        return [0.5] * n
