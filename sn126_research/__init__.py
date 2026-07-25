"""SN126 Poker44 offline research harness.

A reproducible pipeline for miner development:

    benchmark API -> download chunks -> apply official prepare_hand_for_miner()
    -> hero-focused chunk features -> Poker44 reward metric.

This package is research-only. It does NOT implement a miner. It exists to answer
one question first: how separable are bots from humans using only features that
survive the live validator sanitizer?
"""

__all__ = [
    "benchmark_client",
    "sanitizer",
    "features",
    "dataset",
    "evaluation",
]
