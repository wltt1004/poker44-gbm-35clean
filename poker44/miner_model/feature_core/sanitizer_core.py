"""Padding dedupe for already-sanitized chunks (production runtime).

Vendored verbatim from sn126_research/sanitizer.py @ 10c039b. Only the parts the
SERVED miner needs are kept: ``dedupe_padding`` and the visible bet-bucket
ladder. Projection helpers (to_live_hand / to_live_group) stay in research —
production chunks arrive ALREADY sanitized by the validator, so the miner must
never call prepare_hand_for_miner itself.

``VISIBLE_BB_BUCKETS`` is imported from the official validator sanitizer inside
this same repository, exactly as research did, so the ladder has one source of
truth and identical values.
"""

from __future__ import annotations

from typing import Any, Dict, List

from poker44.validator.payload_view import _VISIBLE_BB_BUCKETS as VISIBLE_BB_BUCKETS

# Fields that identify a duplicated action (everything except the reassigned action_id).
_IDENTITY_KEYS = (
    "street",
    "actor_seat",
    "action_type",
    "normalized_amount_bb",
    "pot_before",
    "pot_after",
)


def dedupe_padding(hand: Dict[str, Any]) -> Dict[str, Any]:
    """Collapse actions that are identical except for action_id.

    Safe for normal hands: two genuinely distinct actions differ in at least one
    identity key (actor_seat / street / amount / pot). This only removes the
    sanitizer's 1->12 padding and any exact repeats.
    """
    actions = hand.get("actions") or []
    if len(actions) <= 1:
        return hand
    seen = set()
    deduped: List[Dict[str, Any]] = []
    for act in actions:
        key = tuple(act.get(k) for k in _IDENTITY_KEYS)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(act)
    out = dict(hand)
    out["actions"] = deduped
    return out
