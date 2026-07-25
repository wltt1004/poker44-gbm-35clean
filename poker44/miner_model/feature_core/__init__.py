"""Production-owned feature implementations for the SN126 miner.

Vendored VERBATIM from the validated research modules at commit 10c039b so the
public miner has zero runtime dependency on the private research repository.
Function bodies are byte-identical to the code the GBM was trained against;
only import paths and module headers changed. Golden parity tests assert
bit-for-bit identical outputs against fixtures captured pre-vendoring.

Module map (research origin -> vendored module):
    sn126_research/sanitizer.py       -> sanitizer_core.py  (dedupe_padding only)
    sn126_research/features.py        -> base30.py          (30 base features)
    sn126_research/features_plus.py   -> plus.py            (17 interaction candidates)
    sn126_research/features_plus2.py  -> plus2.py           (10 determinism candidates)
"""

from .base30 import FEATURE_NAMES, extract_features
from .plus import extract_interaction_features
from .plus2 import extract as extract_determinism_features
from .sanitizer_core import VISIBLE_BB_BUCKETS, dedupe_padding

__all__ = [
    "FEATURE_NAMES",
    "extract_features",
    "extract_interaction_features",
    "extract_determinism_features",
    "VISIBLE_BB_BUCKETS",
    "dedupe_padding",
]
