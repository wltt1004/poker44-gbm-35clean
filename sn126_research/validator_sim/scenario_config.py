"""Scenario configuration for the validator-faithful simulators.

All knobs live here so tests can assert exact class counts and reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass

# Request shapes to sweep.
CHUNK_COUNTS = [8, 16, 24, 32, 64, 100]
# Bot fractions; "one_bot" = exactly one bot in the request.
BOT_FRACTIONS = ["one_bot", 0.05, 0.10, 0.15, 0.25, 0.50]

# Offline-only calibration candidates (production p0 is NOT changed).
P0_CANDIDATES = [0.70, 0.75, 0.80, 0.85, 0.88, 0.90, 0.95]
PRODUCTION_P0 = 0.85

# Simulation sizes.
N_REQUESTS = 1000          # per (release, scenario) when data permits
N_REQUESTS_P0 = 500        # per (release, p0) for the walk-forward p0 sweep
N_COMPETITIONS = 5000      # Monte-Carlo competitions for PART 3
SEED = 12345

# Data pull.
N_DATES = 8
MAX_RECORDS_PER_DATE = 200

# Representative scenario used for the p0 walk-forward and the competition sim
# (a plausible live shape; documented as an assumption, not confirmed backend behavior).
REPRESENTATIVE_CHUNK_COUNT = 32
REPRESENTATIVE_BOT_FRACTION = 0.15

COMPOSITE_THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80]


def n_bots_for(chunk_count: int, bot_frac) -> int:
    if bot_frac == "one_bot":
        return 1
    return int(round(chunk_count * float(bot_frac)))


def is_valid_scenario(chunk_count: int, bot_frac) -> bool:
    """Mixed request required: at least one bot AND at least one human."""
    nb = n_bots_for(chunk_count, bot_frac)
    return 1 <= nb <= chunk_count - 1


def valid_scenarios():
    for cc in CHUNK_COUNTS:
        for bf in BOT_FRACTIONS:
            if is_valid_scenario(cc, bf):
                yield cc, bf


@dataclass(frozen=True)
class Scenario:
    chunk_count: int
    bot_fraction: object
    n_bots: int

    @property
    def label(self) -> str:
        bf = "one_bot" if self.bot_fraction == "one_bot" else f"{float(self.bot_fraction):.2f}"
        return f"cc{self.chunk_count}_bf{bf}"
