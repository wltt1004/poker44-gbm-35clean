"""Production inference feature path for the Poker44 miner.

This package hosts the *production* feature pipeline that runs inside
neurons/miner.py. It consumes chunks that the validator has ALREADY sanitized
(via prepare_hand_for_miner) and must therefore:

  * NOT call prepare_hand_for_miner again (no double-sanitization), and
  * apply the same dedupe_padding used during research, so the single-action
    -> 12-copy sanitizer artifact is collapsed identically to training.

Feature implementations are reused verbatim from the validated research modules
to guarantee train/production parity. For final deployment these can be vendored
into this package so `sn126_research` is not a runtime dependency; behavior must
remain identical.
"""

from .features import (
    PROD_FEATURE_ORDER,
    chunk_feature_vector,
    dedupe_chunk,
)

__all__ = ["PROD_FEATURE_ORDER", "chunk_feature_vector", "dedupe_chunk"]
