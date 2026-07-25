"""Live-view adapter.

We NEVER reimplement the sanitizer. We import the official
``prepare_hand_for_miner`` from the validator package so the offline training
view is byte-identical (in distribution) to what miners receive on-chain.

The only thing we add is ``dedupe_padding`` to collapse the sanitizer's
single-action -> 12-identical-copies artifact, which would otherwise over-weight
those hands in pooled features.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

# Make ``poker44`` importable whether or not the repo is pip-installed.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from poker44.validator.payload_view import prepare_hand_for_miner  # noqa: E402
    from poker44.validator.payload_view import _VISIBLE_BB_BUCKETS as VISIBLE_BB_BUCKETS  # noqa: E402
except Exception as exc:  # pragma: no cover - explicit, actionable failure
    raise ImportError(
        "Could not import the official sanitizer from poker44.validator.payload_view. "
        f"Ensure the repo root ({_REPO_ROOT}) is on PYTHONPATH or run `pip install -e .`. "
        f"Original error: {exc}"
    ) from exc


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


def to_live_hand(raw_hand: Dict[str, Any], *, dedupe: bool = True) -> Dict[str, Any]:
    """Project a benchmark/raw hand through the exact live sanitizer."""
    view = prepare_hand_for_miner(raw_hand)
    return dedupe_padding(view) if dedupe else view


def to_live_group(group: List[Dict[str, Any]], *, dedupe: bool = True) -> List[Dict[str, Any]]:
    """Project a whole benchmark group (one player) to its live view."""
    return [to_live_hand(h, dedupe=dedupe) for h in group]
