"""Five-round (5 x 24h) competition simulator inside one 120h epoch.

Each of the 5 rounds draws a per-cycle reward from that round's own release
reward distribution (production p0, representative scenario). The composite is
combined under an ASSUMED aggregation (arithmetic mean) — the true backend
aggregation is private and NOT confirmed here. Fully reproducible from a seed.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

_ZERO = 1e-9


def simulate_competitions(
    round_reward_arrays: List[np.ndarray],
    *,
    n_competitions: int,
    thresholds: List[float],
    seed: int,
    aggregation: str = "mean",
) -> Dict[str, object]:
    """Monte-Carlo composite distribution across the 5 rounds.

    round_reward_arrays: 5 arrays of per-request rewards (one per round/release).
    aggregation: only 'mean' is supported and it is an EXPLICIT ASSUMPTION.
    """
    if aggregation != "mean":
        raise ValueError("only the 'mean' aggregation hypothesis is implemented (assumption)")
    rng = np.random.RandomState(seed)
    n_rounds = len(round_reward_arrays)
    per_round = np.empty((n_competitions, n_rounds))
    for r, arr in enumerate(round_reward_arrays):
        # one independent reward draw per round per competition
        per_round[:, r] = rng.choice(arr, size=n_competitions, replace=True)
    composite = per_round.mean(axis=1)                 # ASSUMED aggregation
    worst_round = per_round.min(axis=1)
    any_zero_round = (per_round <= _ZERO).any(axis=1)

    def stats(a):
        return {"mean": float(a.mean()), "median": float(np.median(a)),
                "std": float(a.std()), "p10": float(np.percentile(a, 10)),
                "p90": float(np.percentile(a, 90)), "min": float(a.min()), "max": float(a.max())}

    return {
        "n_competitions": n_competitions,
        "n_rounds": n_rounds,
        "aggregation": aggregation,
        "composite": composite,
        "per_round": per_round,
        "composite_stats": stats(composite),
        "worst_round_stats": stats(worst_round),
        "prob_any_zero_round": float(any_zero_round.mean()),
        "prob_exceed": {float(t): float((composite > t).mean()) for t in thresholds},
        "round_mean_reward": [float(a.mean()) for a in round_reward_arrays],
    }
