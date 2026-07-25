# Vendored verbatim from sn126_research/features_plus2.py @ 10c039b
# (no import changes; all function bodies byte-identical).
"""Conditional / determinism interaction features (research; not yet in miner).

Motivation: the current features.py (30) and features_plus.py (17) are marginal
aggregates + simple transitions. The last search showed the untapped signal is
CONDITIONAL (potodds_call_fold_gap: weak alone, strong in combination). Bots are
deterministic policies: given the same STATE they take the same ACTION and the
same SIZE. Humans mix. So we measure state-conditioned determinism.

All features:
  * computed on the sanitized live view (survive live scoring),
  * pooled over ALL seats / ALL hands (robust to per-hand seat re-aliasing),
  * amounts at BUCKET / ratio resolution only (jitter-robust),
  * state keys use only surviving fields: street, faced-bet flag, pot buckets.

Survival notes vs prepare_hand_for_miner:
  * street, action_type, actor_seat(aliased), normalized_amount_bb(bucketed),
    pot_before/after(bucketed), call_to(bucketed) all SURVIVE.
  * the 5-8 action window adds noise to per-spot counts; we mitigate by pooling
    all seats over 30-100 hands and using coarse spot keys.
"""

from __future__ import annotations

import math
from collections import Counter, OrderedDict, defaultdict
from typing import Any, Dict, List

_AGGR = {"bet", "raise"}
_VISIBLE_BB = 0.02
_EPS = 1e-9

CANDIDATE_NAMES = (
    "spot_action_entropy",       # determinism: low => bot
    "spot_dominant_frac",        # determinism: high => bot
    "betsize_vocab_ratio",       # sizing menu size: low => bot
    "betsize_street_std_mean",   # conditional sizing consistency: low => bot
    "action_bigram_entropy",     # transition predictability: low => bot
    "repeat_action_rate",        # consecutive same-action tendency
    "aggr_slope_pf_flop",        # barreling shape
    "aggr_slope_flop_turn",      # barreling shape
    "postflop_fold_share",       # fold-timing signature
    "mean_aggressors_per_hand",  # cross-seat aggression concentration
)


def _entropy(counts) -> float:
    tot = float(sum(counts))
    if tot <= 0:
        return 0.0
    h = -sum((c / tot) * math.log(c / tot) for c in counts if c > 0)
    k = sum(1 for c in counts if c > 0)
    return h / math.log(k) if k > 1 else 0.0


def _std(xs) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / n)


def _pot_bb(v) -> float:
    return float(v or 0.0) / _VISIBLE_BB


def extract(live_group: List[List[Dict[str, Any]]]) -> "OrderedDict[str, float]":
    spots: Dict[tuple, Counter] = defaultdict(Counter)          # (street, faced) -> action counts
    bigrams: Counter = Counter()
    pair_total = repeat_total = 0
    bet_buckets: List[int] = []
    street_betpot: Dict[str, List[float]] = defaultdict(list)
    street_actions: Counter = Counter()
    street_aggr: Counter = Counter()
    fold_total = fold_post = 0
    aggressors_per_hand: List[int] = []
    n_hands = 0

    for hand in live_group:
        acts = hand.get("actions") or []
        if not acts:
            continue
        n_hands += 1
        prev = None
        faced: Dict[str, bool] = {}
        hand_aggr_seats = set()
        for a in acts:
            st = str(a.get("street") or "preflop")
            t = a.get("action_type")
            amt = float(a.get("normalized_amount_bb") or 0.0)
            pot_b = _pot_bb(a.get("pot_before"))

            spots[(st, faced.get(st, False))][t] += 1
            street_actions[st] += 1
            if t in _AGGR:
                street_aggr[st] += 1
                hand_aggr_seats.add(a.get("actor_seat"))
                if amt > 0:
                    bet_buckets.append(min(int(round(amt)), 200))
                    if pot_b > _EPS:
                        street_betpot[st].append(amt / pot_b)
            if t == "fold":
                fold_total += 1
                if st != "preflop":
                    fold_post += 1
            if prev is not None:
                bigrams[(prev, t)] += 1
                pair_total += 1
                repeat_total += int(prev == t)
            if t in _AGGR:
                faced[st] = True
            prev = t
        aggressors_per_hand.append(len(hand_aggr_seats))

    f: "OrderedDict[str, float]" = OrderedDict((n, 0.0) for n in CANDIDATE_NAMES)
    if n_hands == 0 or not spots:
        return f

    # determinism over (street, faced) spots
    total = sum(sum(c.values()) for c in spots.values())
    ent_w = dom_num = 0.0
    for c in spots.values():
        w = sum(c.values())
        ent_w += w * _entropy(list(c.values()))
        if w > 0 and max(c.values()) / w > 0.70:
            dom_num += w
    f["spot_action_entropy"] = ent_w / max(total, 1)
    f["spot_dominant_frac"] = dom_num / max(total, 1)

    # sizing vocabulary + conditional consistency
    if bet_buckets:
        f["betsize_vocab_ratio"] = len(set(bet_buckets)) / len(bet_buckets)
    per_street_std = [_std(v) for v in street_betpot.values() if len(v) >= 2]
    f["betsize_street_std_mean"] = sum(per_street_std) / len(per_street_std) if per_street_std else 0.0

    # transition regularity
    f["action_bigram_entropy"] = _entropy(list(bigrams.values()))
    f["repeat_action_rate"] = repeat_total / max(pair_total, 1)

    # street progression shape
    def _af(st):
        return street_aggr.get(st, 0) / street_actions.get(st, 1) if street_actions.get(st, 0) else 0.0
    f["aggr_slope_pf_flop"] = _af("flop") - _af("preflop")
    f["aggr_slope_flop_turn"] = _af("turn") - _af("flop")
    f["postflop_fold_share"] = fold_post / max(fold_total, 1)

    # cross-seat
    f["mean_aggressors_per_hand"] = sum(aggressors_per_hand) / len(aggressors_per_hand)
    return f


def candidate_vector(live_group: List[List[Dict[str, Any]]]) -> List[float]:
    f = extract(live_group)
    return [float(f[n]) for n in CANDIDATE_NAMES]
