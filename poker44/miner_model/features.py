"""Production 35-feature extractor with the dedupe_padding fix baked in.

Pipeline (per validator request):

    DetectionSynapse.chunks  (already sanitized by the validator)
        -> dedupe_chunk()        # collapse the 1-action -> 12-copy padding
        -> _assemble()           # 35 features in the frozen training order
        -> feature vector

We deliberately reuse the exact research implementations so production features
are bit-for-bit identical to what the GBM was trained on:

  * dedupe_padding  -- from sn126_research.sanitizer   (same logic as training)
  * base 30         -- sn126_research.features.extract_features(scope="all")
  * +5 additions    -- sn126_research.features_plus / features_plus2

We do NOT call prepare_hand_for_miner here: the validator has already applied it
(poker44/validator/forward.py:121). Re-applying it would re-alias seats and
re-bucket amounts (double-sanitization).
"""

from __future__ import annotations

from typing import Any, Dict, List

from sn126_research.features import FEATURE_NAMES, extract_features
from sn126_research.features_plus import extract_interaction_features as _p1_extract
from sn126_research.features_plus2 import extract as _p2_extract
from sn126_research.sanitizer import dedupe_padding

# Frozen 35-feature order = base-30 (FEATURE_NAMES) + the 5 validated additions.
# This MUST match the column order model.joblib is trained on. Persist it in
# model_meta.json and assert equality at model load.
_EXTRA_FEATURES = (
    "betsize_street_std_mean",
    "action_bigram_entropy",
    "pot_escalation_mean",
    "mean_aggressors_per_hand",
    "potodds_call_fold_gap",
)
PROD_FEATURE_ORDER = tuple(FEATURE_NAMES) + _EXTRA_FEATURES


def dedupe_chunk(chunk: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse the sanitizer's 1-action -> 12-copy padding on every hand.

    This is the production analogue of research's ``to_live_hand(dedupe=True)``,
    minus the prepare_hand_for_miner step (chunks are already sanitized).
    """
    return [dedupe_padding(hand) for hand in chunk]


def _assemble(hands: List[Dict[str, Any]]) -> List[float]:
    """Compute the 35-vector from a list of (deduped) sanitized hands."""
    base = extract_features(hands, scope="all")   # 30
    p1 = _p1_extract(hands)                        # 17 (we use 2)
    p2 = _p2_extract(hands)                        # 10 (we use 3)
    merged: Dict[str, float] = dict(base)
    merged.update(p1)
    merged.update(p2)
    return [float(merged[name]) for name in PROD_FEATURE_ORDER]


def chunk_feature_vector(chunk: List[Dict[str, Any]], *, dedupe: bool = True) -> List[float]:
    """Production entry point: one sanitized chunk -> the 35-feature vector.

    dedupe=True is the correct production setting (matches training). dedupe=False
    exists only so tests can demonstrate the train/production skew the padding
    would otherwise cause.
    """
    hands = dedupe_chunk(chunk) if dedupe else list(chunk)
    return _assemble(hands)
