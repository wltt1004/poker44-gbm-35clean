"""Sanitizer-robust chunk features.

EMPIRICAL CORRECTION (see experiments/diagnostic_ceiling.py):
  The discriminative signal is TABLE-LEVEL, not hero-level. Restricting to the
  hero's own actions (actor_seat == hero_seat) collapses to chance (AUC ~0.51),
  while aggregating over ALL seats reaches AUC ~0.79 / AP ~0.84 on a held-out
  release date -- and that all-seats signal fully survives the live sanitizer.
  So ``scope="all"`` is the default. ``scope="hero"`` is retained only for
  ablation/comparison.

Contract (constrained to survive the live view either way):
  * aggregate over the whole group (one labeled sample)
  * amounts used at BUCKET resolution only (the live jitter makes sub-bucket
    precision meaningless)
  * NEVER use: hand_id, absolute seat numbers, outcome/cards/timing, raw per-hand
    action counts, or the number of hands n (differs 30 train -> 100 live)

Input to :func:`extract_features` is a group already projected through the live
sanitizer (see ``sanitizer.to_live_group``).
"""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any, Dict, List, Sequence, Tuple

from .sanitizer import VISIBLE_BB_BUCKETS

_ACTION_TYPES = ("fold", "check", "call", "bet", "raise")
_AGGRESSIVE = {"bet", "raise"}
_VOLUNTARY_PF = {"call", "bet", "raise"}
_STREET_ORD = {"preflop": 0, "flop": 1, "turn": 2, "river": 3, "showdown": 3}
_VISIBLE_BB = 0.02  # sanitizer's fixed visible big blind (currency = bb * 0.02)
_EPS = 1e-9

# Coarse bet-size bands (bb), aligned with the sanitizer's jitter tiers (1.5 / 8).
_BAND_SMALL_MAX = 2.0
_BAND_MID_MAX = 8.0

# Ordered, stable feature schema. Keep this as the single source of truth.
FEATURE_NAMES: Tuple[str, ...] = (
    # --- hero action rates (pooled) ---
    "fold_rate",
    "check_rate",
    "call_rate",
    "bet_rate",
    "raise_rate",
    "aggression_ratio",
    "aggression_freq",
    # --- street ---
    "preflop_action_rate",
    "postflop_action_rate",
    "preflop_aggression_freq",
    "postflop_aggression_freq",
    "street_depth_mean",
    "street_depth_std",
    "reach_flop_rate",
    "reach_turn_rate",
    "reach_river_rate",
    "vpip",
    "pfr",
    # --- consistency ---
    "action_entropy",
    "perhand_entropy_mean",
    "action_dist_dispersion",
    "aggression_dispersion",
    # --- bet sizing (bucket level only) ---
    "bet_bucket_mean",
    "bet_bucket_std",
    "bet_small_frac",
    "bet_mid_frac",
    "bet_large_frac",
    "bet_to_pot_mean",
    "bet_overbet_frac",
    # --- presence (mildly window-sensitive; aggregate is stable) ---
    "hero_presence_rate",
)


def _bucket_index(bb_value: float) -> int:
    """Nearest index in the sanitizer's visible bb ladder (bucket resolution)."""
    v = max(0.0, float(bb_value or 0.0))
    best_i, best_d = 0, float("inf")
    for i, b in enumerate(VISIBLE_BB_BUCKETS):
        d = abs(b - v)
        if d < best_d:
            best_i, best_d = i, d
    return best_i


def _entropy(counts: Sequence[float]) -> float:
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log(p)
    return h / math.log(len(counts)) if len(counts) > 1 else 0.0  # normalized [0,1]


def _std(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / n)


def _select_actions(hand: Dict[str, Any], scope: str) -> List[Dict[str, Any]]:
    actions = hand.get("actions") or []
    if scope == "all":
        return list(actions)
    hero_seat = (hand.get("metadata") or {}).get("hero_seat")
    if hero_seat in (None, 0):
        return []
    return [a for a in actions if a.get("actor_seat") == hero_seat]


def extract_features(
    live_group: List[List[Dict[str, Any]]] | List[Dict[str, Any]],
    *,
    scope: str = "all",
) -> "OrderedDict[str, float]":
    """Compute the robust feature vector for one live-view group.

    scope="all"  -> table-level aggregation (recommended; carries the signal)
    scope="hero" -> hero-only aggregation (ablation; near chance on this benchmark)
    """
    pooled = {t: 0 for t in _ACTION_TYPES}
    pooled_total = 0

    per_hand_dists: List[List[float]] = []       # per-hand [fold,check,call,bet,raise] fractions
    per_hand_aggression: List[float] = []        # per-hand (bet+raise)/total
    street_depths: List[int] = []                # hero max street ordinal per hand
    reach_flop = reach_turn = reach_river = 0
    vpip_hands = pfr_hands = 0
    hands_with_hero = 0

    pf_actions = pf_aggr = 0
    post_actions = post_aggr = 0

    bet_bucket_indices: List[int] = []
    bet_bb_values: List[float] = []              # bucketed magnitudes (for band fractions)
    bet_to_pot: List[float] = []                 # coarse bet/pot ratios
    overbets = 0
    bet_events = 0

    n_hands = len(live_group)
    for hand in live_group:
        hero = _select_actions(hand, scope)
        if not hero:
            continue
        hands_with_hero += 1

        hand_counts = {t: 0 for t in _ACTION_TYPES}
        max_street = 0
        saw_pf_voluntary = False
        saw_pf_raise = False
        for a in hero:
            atype = a.get("action_type")
            if atype in hand_counts:
                hand_counts[atype] += 1
                pooled[atype] += 1
                pooled_total += 1
            street = str(a.get("street") or "preflop").lower()
            sord = _STREET_ORD.get(street, 0)
            max_street = max(max_street, sord)

            is_pf = sord == 0
            is_aggr = atype in _AGGRESSIVE
            if is_pf:
                pf_actions += 1
                pf_aggr += int(is_aggr)
                if atype in _VOLUNTARY_PF:
                    saw_pf_voluntary = True
                if atype == "raise":
                    saw_pf_raise = True
            else:
                post_actions += 1
                post_aggr += int(is_aggr)

            # bucket-level bet sizing (aggressive actions with money in)
            if is_aggr:
                bb = float(a.get("normalized_amount_bb") or 0.0)
                if bb > 0:
                    bet_events += 1
                    bet_bucket_indices.append(_bucket_index(bb))
                    snap = VISIBLE_BB_BUCKETS[_bucket_index(bb)]
                    bet_bb_values.append(snap)
                    pot_before_bb = float(a.get("pot_before") or 0.0) / _VISIBLE_BB
                    if pot_before_bb > _EPS:
                        ratio = snap / pot_before_bb
                        bet_to_pot.append(ratio)
                        overbets += int(ratio > 1.0)

        h_total = sum(hand_counts.values())
        if h_total > 0:
            per_hand_dists.append([hand_counts[t] / h_total for t in _ACTION_TYPES])
            per_hand_aggression.append(
                (hand_counts["bet"] + hand_counts["raise"]) / h_total
            )
        street_depths.append(max_street)
        reach_flop += int(max_street >= 1)
        reach_turn += int(max_street >= 2)
        reach_river += int(max_street >= 3)
        vpip_hands += int(saw_pf_voluntary)
        pfr_hands += int(saw_pf_raise)

    f: "OrderedDict[str, float]" = OrderedDict((name, 0.0) for name in FEATURE_NAMES)
    if pooled_total == 0 or hands_with_hero == 0:
        return f  # degenerate group -> all-zero vector (model still gets a row)

    # action rates
    for t in _ACTION_TYPES:
        f[f"{t}_rate"] = pooled[t] / pooled_total
    aggr = pooled["bet"] + pooled["raise"]
    passive = pooled["call"] + pooled["check"]
    f["aggression_ratio"] = aggr / (passive + _EPS)
    f["aggression_freq"] = aggr / pooled_total

    # street
    f["preflop_action_rate"] = pf_actions / pooled_total
    f["postflop_action_rate"] = post_actions / pooled_total
    f["preflop_aggression_freq"] = pf_aggr / (pf_actions + _EPS)
    f["postflop_aggression_freq"] = post_aggr / (post_actions + _EPS)
    f["street_depth_mean"] = sum(street_depths) / len(street_depths)
    f["street_depth_std"] = _std(street_depths)
    f["reach_flop_rate"] = reach_flop / hands_with_hero
    f["reach_turn_rate"] = reach_turn / hands_with_hero
    f["reach_river_rate"] = reach_river / hands_with_hero
    f["vpip"] = vpip_hands / hands_with_hero
    f["pfr"] = pfr_hands / hands_with_hero

    # consistency
    f["action_entropy"] = _entropy([pooled[t] for t in _ACTION_TYPES])
    if per_hand_dists:
        f["perhand_entropy_mean"] = sum(_entropy(d) for d in per_hand_dists) / len(per_hand_dists)
        # dispersion = mean over the 5 components of their std across hands
        comp_std = [
            _std([d[i] for d in per_hand_dists]) for i in range(len(_ACTION_TYPES))
        ]
        f["action_dist_dispersion"] = sum(comp_std) / len(comp_std)
    f["aggression_dispersion"] = _std(per_hand_aggression)

    # bet sizing (bucket-level)
    if bet_events:
        f["bet_bucket_mean"] = sum(bet_bucket_indices) / bet_events
        f["bet_bucket_std"] = _std(bet_bucket_indices)
        small = sum(1 for v in bet_bb_values if v <= _BAND_SMALL_MAX)
        mid = sum(1 for v in bet_bb_values if _BAND_SMALL_MAX < v <= _BAND_MID_MAX)
        large = sum(1 for v in bet_bb_values if v > _BAND_MID_MAX)
        f["bet_small_frac"] = small / bet_events
        f["bet_mid_frac"] = mid / bet_events
        f["bet_large_frac"] = large / bet_events
    if bet_to_pot:
        f["bet_to_pot_mean"] = sum(bet_to_pot) / len(bet_to_pot)
        f["bet_overbet_frac"] = overbets / len(bet_to_pot)

    # presence (aggregate over the chunk; individual-hand presence is window-noisy)
    f["hero_presence_rate"] = hands_with_hero / max(n_hands, 1)
    return f


def feature_vector(live_group: List[List[Dict[str, Any]]], *, scope: str = "all") -> List[float]:
    """Ordered numeric vector aligned to FEATURE_NAMES."""
    f = extract_features(live_group, scope=scope)
    return [float(f[name]) for name in FEATURE_NAMES]
