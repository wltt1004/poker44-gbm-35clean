# Vendored verbatim from sn126_research/features_plus.py @ 10c039b (only the
# VISIBLE_BB_BUCKETS import path changed; all function bodies byte-identical).
"""Candidate ALL-TABLE INTERACTION features (research; not yet in the miner).

The current 30 features (features.py) are first-order marginal aggregates. They
ignore conditional/relational/sequential structure, which is exactly where bot
rigidity lives. These candidates condition one signal on another and are all:
  * computed over the sanitized live view (survive live scoring),
  * pooled over ALL seats and ALL hands (robust to per-hand seat re-aliasing;
    they exploit the ~6x action volume that made all-seats >> hero-only),
  * amounts used at bucket/ratio resolution only (jitter-robust).

Categories:
  A. sizing RIGIDITY (bots size formulaically given pot/street)
  B. SEQUENTIAL transitions (c-bet, 3-bet, check-raise, fold-to-bet, aggr->aggr)
  C. pot-odds RATIONALITY (sharp call/fold threshold)
  D. pot DYNAMICS (how pots escalate)
  E. action-ORDER position proxies (button is dead; order is the only position)
"""

from __future__ import annotations

import math
from collections import Counter, OrderedDict
from typing import Any, Dict, List, Tuple

from .sanitizer_core import VISIBLE_BB_BUCKETS

_AGGR = {"bet", "raise"}
_VISIBLE_BB = 0.02
_EPS = 1e-9
_BIG_POT_BB = 30.0

CANDIDATE_NAMES: Tuple[str, ...] = (
    # A. sizing rigidity
    "betpot_std",
    "betpot_modal_frac",
    "betsize_bucket_entropy",
    "pf_open_size_std",
    "flop_betpot_std",
    # B. sequential transitions
    "cbet_freq",
    "checkraise_freq",
    "threebet_freq",
    "fold_to_bet_freq",
    "aggr_after_aggr_freq",
    # C. pot-odds rationality
    "potodds_call_fold_gap",
    "facing_bet_rate",
    # D. pot dynamics
    "mean_final_pot_bb",
    "pot_escalation_mean",
    "big_pot_rate",
    # E. action-order position
    "first_actor_aggr_rate",
    "last_actor_aggr_rate",
)


def _bucket_idx(bb: float) -> int:
    v = max(0.0, float(bb or 0.0))
    return min(range(len(VISIBLE_BB_BUCKETS)), key=lambda i: abs(VISIBLE_BB_BUCKETS[i] - v))


def _std(xs) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / n)


def _entropy(counts) -> float:
    tot = float(sum(counts))
    if tot <= 0:
        return 0.0
    h = -sum((c / tot) * math.log(c / tot) for c in counts if c > 0)
    k = sum(1 for c in counts if c > 0)
    return h / math.log(k) if k > 1 else 0.0


def _pot_bb(currency: Any) -> float:
    return float(currency or 0.0) / _VISIBLE_BB


def extract_interaction_features(live_group: List[List[Dict[str, Any]]]) -> "OrderedDict[str, float]":
    betpot: List[float] = []
    bet_buckets: List[int] = []
    pf_open: List[float] = []
    flop_betpot: List[float] = []

    cbet_num = cbet_den = 0
    cr_num = 0
    tb_num = tb_den = 0
    facing_actions = facing_folds = 0
    aa_num = aa_den = 0
    po_call: List[float] = []
    po_fold: List[float] = []
    final_pots: List[float] = []
    escal: List[float] = []
    max_pots: List[float] = []
    first_aggr = last_aggr = 0
    n_hands = 0

    for hand in live_group:
        acts = hand.get("actions") or []
        if not acts:
            continue
        n_hands += 1

        pf_raised = any(
            str(a.get("street")) == "preflop" and a.get("action_type") == "raise" for a in acts
        )
        flop_bet = any(
            str(a.get("street")) == "flop" and a.get("action_type") in _AGGR for a in acts
        )
        if pf_raised:
            cbet_den += 1
            cbet_num += int(flop_bet)
        first_aggr += int(acts[0].get("action_type") in _AGGR)
        last_aggr += int(acts[-1].get("action_type") in _AGGR)

        prev_type = None
        pf_raise_count = 0
        street_aggr_before: Dict[str, bool] = {}
        max_pot = 0.0
        for a in acts:
            st = str(a.get("street") or "preflop")
            t = a.get("action_type")
            amt = float(a.get("normalized_amount_bb") or 0.0)
            pot_b = _pot_bb(a.get("pot_before"))
            pot_a = _pot_bb(a.get("pot_after"))
            max_pot = max(max_pot, pot_a)

            faced = street_aggr_before.get(st, False)
            if faced:
                facing_actions += 1
                if t == "fold":
                    facing_folds += 1
                call_amt = _pot_bb(a.get("call_to")) if a.get("call_to") else amt
                denom = pot_b + call_amt
                po = call_amt / denom if denom > _EPS else 0.0
                if t == "call":
                    po_call.append(po)
                elif t == "fold":
                    po_fold.append(po)

            if prev_type is not None:
                aa_den += 1
                if prev_type in _AGGR and t in _AGGR:
                    aa_num += 1
            if t == "raise" and prev_type == "check":
                cr_num += 1

            if t in _AGGR and amt > 0:
                bet_buckets.append(_bucket_idx(amt))
                if pot_b > _EPS:
                    betpot.append(amt / pot_b)
                    escal.append(pot_a / (pot_b + _EPS))
                if st == "preflop":
                    pf_open.append(amt)
                if st == "flop" and pot_b > _EPS:
                    flop_betpot.append(amt / pot_b)

            if st == "preflop" and t == "raise":
                pf_raise_count += 1
                tb_den += 1
                if pf_raise_count >= 2:
                    tb_num += 1

            if t in _AGGR:
                street_aggr_before[st] = True
            prev_type = t

        final_pots.append(_pot_bb(acts[-1].get("pot_after")))
        max_pots.append(max_pot)

    f: "OrderedDict[str, float]" = OrderedDict((n, 0.0) for n in CANDIDATE_NAMES)
    if n_hands == 0:
        return f

    # A. sizing rigidity
    f["betpot_std"] = _std(betpot)
    if betpot:
        bins = Counter(round(min(x, 3.0) / 0.25) for x in betpot)  # 0.25-pot-width bins
        f["betpot_modal_frac"] = max(bins.values()) / len(betpot)
    if bet_buckets:
        f["betsize_bucket_entropy"] = _entropy(list(Counter(bet_buckets).values()))
    f["pf_open_size_std"] = _std(pf_open)
    f["flop_betpot_std"] = _std(flop_betpot)

    # B. sequential transitions
    f["cbet_freq"] = cbet_num / max(cbet_den, 1)
    f["checkraise_freq"] = cr_num / n_hands
    f["threebet_freq"] = tb_num / max(tb_den, 1)
    f["fold_to_bet_freq"] = facing_folds / max(facing_actions, 1)
    f["aggr_after_aggr_freq"] = aa_num / max(aa_den, 1)

    # C. pot-odds rationality
    mean_call = sum(po_call) / len(po_call) if po_call else 0.0
    mean_fold = sum(po_fold) / len(po_fold) if po_fold else 0.0
    f["potodds_call_fold_gap"] = mean_call - mean_fold
    total_actions = aa_den + n_hands  # ~ number of actions
    f["facing_bet_rate"] = facing_actions / max(total_actions, 1)

    # D. pot dynamics
    f["mean_final_pot_bb"] = sum(final_pots) / len(final_pots)
    f["pot_escalation_mean"] = sum(escal) / len(escal) if escal else 0.0
    f["big_pot_rate"] = sum(1 for p in max_pots if p > _BIG_POT_BB) / len(max_pots)

    # E. action-order position
    f["first_actor_aggr_rate"] = first_aggr / n_hands
    f["last_actor_aggr_rate"] = last_aggr / n_hands
    return f


def candidate_vector(live_group: List[List[Dict[str, Any]]]) -> List[float]:
    f = extract_interaction_features(live_group)
    return [float(f[n]) for n in CANDIDATE_NAMES]
