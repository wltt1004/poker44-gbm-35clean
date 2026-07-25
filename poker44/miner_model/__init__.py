"""Production inference feature path for the Poker44 miner.

This package hosts the *production* feature pipeline that runs inside
neurons/miner.py. It consumes chunks that the validator has ALREADY sanitized
(via prepare_hand_for_miner) and must therefore:

  * NOT call prepare_hand_for_miner again (no double-sanitization), and
  * apply the same dedupe_padding used during research, so the single-action
    -> 12-copy sanitizer artifact is collapsed identically to training.

Feature implementations are vendored VERBATIM from the validated research
modules (at commit 10c039b) into the production-owned `feature_core` subpackage,
so `sn126_research` is NOT a runtime dependency. Behavior is bit-for-bit
identical, proven by golden parity fixtures captured before the vendoring.
"""

from .features import (
    PROD_FEATURE_ORDER,
    chunk_feature_vector,
    dedupe_chunk,
)

__all__ = ["PROD_FEATURE_ORDER", "chunk_feature_vector", "dedupe_chunk"]
