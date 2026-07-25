"""SN126 model inference layer: chunks -> features -> GBM -> calibrated risk_scores.

Loads the serialized GBM and metadata ONCE at construction, verifies the feature
contract, then for each request:
    chunk_feature_vector() -> GBM.predict_proba() -> batch percentile calibration.

Framework-agnostic (no bittensor import) so it is unit-testable in isolation.
Never raises from predict(): on any inference error it returns a safe,
correctly-sized fallback so the miner cannot emit a wrong-length response
(which the validator would discard -> 0 reward).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import joblib
import numpy as np

from .calibration import batch_percentile_calibrate
from .features import PROD_FEATURE_ORDER, chunk_feature_vector

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).resolve().parent
_DEFAULT_MODEL = _MODEL_DIR / "model.joblib"
_DEFAULT_META = _MODEL_DIR / "model_meta.json"
_EPS = 1e-9


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

    # ---- calibration ------------------------------------------------------
    def _calibrate(self, raw: np.ndarray) -> np.ndarray:
        """Batch percentile anchor: place 0.5 at the p0-th percentile of THIS request.

        Rank-invariant (preserves AP / recall@FPR); only moves the 0.5 gate.
        Small batches fall back to an absolute anchor learned at train time.
        """
        return batch_percentile_calibrate(
            raw,
            p0=self.p0,
            min_calib_batch=self.min_calib_batch,
            global_anchor_raw=self.global_anchor_raw,
        )

    # ---- inference --------------------------------------------------------
    def predict(
        self,
        chunks: Sequence[List[Dict[str, Any]]],
        *,
        fallback: Optional[Callable[[List[Dict[str, Any]]], float]] = None,
    ) -> List[float]:
        """Return one calibrated risk score per chunk. Guaranteed len == len(chunks)."""
        n = len(chunks)
        if n == 0:
            return []
        try:
            X = np.asarray([chunk_feature_vector(c) for c in chunks], dtype=float)
            if X.shape != (n, len(PROD_FEATURE_ORDER)):
                raise ValueError(f"feature matrix shape {X.shape} != ({n}, {len(PROD_FEATURE_ORDER)})")
            raw = self.model.predict_proba(X)[:, 1]
            scores = self._calibrate(raw)
            out = [float(min(1.0, max(0.0, s))) for s in scores]
            if len(out) != n:
                raise ValueError("score length mismatch")
            return out
        except Exception as exc:  # noqa: BLE001 - inference must never crash the axon
            logger.warning("Poker44Predictor.predict failed (%s); using safe fallback.", exc)
            return self._safe_fallback(chunks, fallback)

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
