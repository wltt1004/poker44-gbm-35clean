"""Verify the production dedupe_padding fix matches research feature behavior.

Scenario: a sanitized chunk containing a hand whose single surviving action was
padded to 12 identical copies by prepare_hand_for_miner, mixed with a normal
multi-action hand. Without dedupe the 12 copies over-weight pooled features
(train/production skew). With dedupe, production features exactly equal the
research pipeline (to_live_group), which is what the GBM was trained on.
"""

import unittest

import numpy as np

from poker44.miner_model.features import (
    PROD_FEATURE_ORDER,
    chunk_feature_vector,
    dedupe_chunk,
)
from poker44.miner_model.feature_core.sanitizer_core import dedupe_padding
from poker44.validator.payload_view import prepare_hand_for_miner


def to_live_group(group):
    """Research-equivalent projection (prepare_hand_for_miner + dedupe per hand)."""
    return [dedupe_padding(prepare_hand_for_miner(h)) for h in group]


def _raw_single_action_hand():
    """Raw hand that sanitizes to ONE action -> padded to 12 identical copies."""
    return {
        "metadata": {
            "game_type": "Hold'em", "limit_type": "No Limit", "max_seats": 2,
            "hero_seat": 1, "button_seat": 2, "sb": 0.01, "bb": 0.02, "ante": 0.0,
        },
        "players": [
            {"player_uid": "seat_1", "seat": 1, "starting_stack": 8.0},
            {"player_uid": "seat_2", "seat": 2, "starting_stack": 8.0},
        ],
        "streets": [],
        "actions": [
            {"action_id": "1", "street": "preflop", "actor_seat": 1, "action_type": "small_blind",
             "amount": 0.01, "raise_to": None, "call_to": None, "normalized_amount_bb": 0.5,
             "pot_before": 0.0, "pot_after": 0.01},
            {"action_id": "2", "street": "preflop", "actor_seat": 2, "action_type": "big_blind",
             "amount": 0.02, "raise_to": None, "call_to": None, "normalized_amount_bb": 1.0,
             "pot_before": 0.01, "pot_after": 0.03},
            {"action_id": "3", "street": "preflop", "actor_seat": 1, "action_type": "raise",
             "amount": 0.16, "raise_to": 0.16, "call_to": None, "normalized_amount_bb": 8.0,
             "pot_before": 0.03, "pot_after": 0.19},
        ],
        "outcome": {"showdown": False},
    }


def _raw_normal_hand():
    """Raw hand with a mix of actions across seats (a real chunk-mate)."""
    return {
        "metadata": {
            "game_type": "Hold'em", "limit_type": "No Limit", "max_seats": 2,
            "hero_seat": 2, "button_seat": 1, "sb": 0.01, "bb": 0.02, "ante": 0.0,
        },
        "players": [
            {"player_uid": "seat_1", "seat": 1, "starting_stack": 10.0},
            {"player_uid": "seat_2", "seat": 2, "starting_stack": 10.0},
        ],
        "streets": [{"street": "flop", "board_cards": []}],
        "actions": [
            {"action_id": "1", "street": "preflop", "actor_seat": 1, "action_type": "small_blind",
             "amount": 0.01, "raise_to": None, "call_to": None, "normalized_amount_bb": 0.5,
             "pot_before": 0.0, "pot_after": 0.01},
            {"action_id": "2", "street": "preflop", "actor_seat": 2, "action_type": "big_blind",
             "amount": 0.02, "raise_to": None, "call_to": None, "normalized_amount_bb": 1.0,
             "pot_before": 0.01, "pot_after": 0.03},
            {"action_id": "3", "street": "preflop", "actor_seat": 1, "action_type": "raise",
             "amount": 0.06, "raise_to": 0.08, "call_to": None, "normalized_amount_bb": 3.0,
             "pot_before": 0.03, "pot_after": 0.09},
            {"action_id": "4", "street": "preflop", "actor_seat": 2, "action_type": "call",
             "amount": 0.06, "raise_to": None, "call_to": 0.08, "normalized_amount_bb": 3.0,
             "pot_before": 0.09, "pot_after": 0.15},
            {"action_id": "5", "street": "flop", "actor_seat": 2, "action_type": "check",
             "amount": 0.0, "raise_to": None, "call_to": None, "normalized_amount_bb": 0.0,
             "pot_before": 0.15, "pot_after": 0.15},
            {"action_id": "6", "street": "flop", "actor_seat": 1, "action_type": "bet",
             "amount": 0.10, "raise_to": None, "call_to": None, "normalized_amount_bb": 5.0,
             "pot_before": 0.15, "pot_after": 0.25},
            {"action_id": "7", "street": "flop", "actor_seat": 2, "action_type": "fold",
             "amount": 0.0, "raise_to": None, "call_to": None, "normalized_amount_bb": 0.0,
             "pot_before": 0.25, "pot_after": 0.25},
        ],
        "outcome": {"showdown": False},
    }


class DedupePaddingParityTests(unittest.TestCase):
    def setUp(self):
        self.raw_A = _raw_single_action_hand()
        self.raw_B = _raw_normal_hand()
        # What the validator actually sends the miner (sanitized, NOT deduped):
        self.validator_chunk = [prepare_hand_for_miner(self.raw_A),
                                prepare_hand_for_miner(self.raw_B)]
        # Research pipeline (sanitize + dedupe) — what the GBM was trained on:
        self.research_hands = to_live_group([self.raw_A, self.raw_B])

    def test_padding_is_present_and_collapses(self):
        sanitized_A = prepare_hand_for_miner(self.raw_A)
        self.assertEqual(len(sanitized_A["actions"]), 12,
                         "single-action hand should be padded to 12 copies by the sanitizer")
        self.assertEqual(len(dedupe_padding(sanitized_A)["actions"]), 1,
                         "dedupe_padding should collapse the 12 identical copies to 1")

    def test_dedupe_matches_research_and_fixes_skew(self):
        vec_research = np.asarray(chunk_feature_vector(self.research_hands, dedupe=True))
        vec_fixed = np.asarray(chunk_feature_vector(self.validator_chunk, dedupe=True))
        vec_buggy = np.asarray(chunk_feature_vector(self.validator_chunk, dedupe=False))

        self.assertEqual(len(vec_fixed), 35)
        self.assertEqual(len(PROD_FEATURE_ORDER), 35)

        # (1) The fix makes production features EXACTLY match the research pipeline.
        self.assertTrue(np.allclose(vec_fixed, vec_research, atol=1e-9),
                        "deduped production features must equal research (training) features")

        # (2) Without dedupe, the 12-copy padding SKEWS features (the real bug).
        self.assertFalse(np.allclose(vec_buggy, vec_research, atol=1e-6),
                         "un-deduped features should differ (padding skew)")

        # Show exactly which features the padding corrupts (informative).
        diff_idx = np.where(~np.isclose(vec_buggy, vec_research, atol=1e-6))[0]
        diff_names = [PROD_FEATURE_ORDER[i] for i in diff_idx]
        self.assertIn("raise_rate", diff_names,
                      "raise_rate must be inflated by the 12 padded raise copies")
        print("\nfeatures corrupted by padding (buggy vs research):")
        for i in diff_idx:
            print(f"  {PROD_FEATURE_ORDER[i]:26s} buggy={vec_buggy[i]:.4f} "
                  f"research/fixed={vec_research[i]:.4f}")

    def test_dedupe_chunk_is_idempotent(self):
        once = dedupe_chunk(self.validator_chunk)
        twice = dedupe_chunk(once)
        self.assertEqual([len(h["actions"]) for h in once],
                         [len(h["actions"]) for h in twice])


if __name__ == "__main__":
    unittest.main()
