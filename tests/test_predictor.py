"""Tests for the SN126 model inference layer (Poker44Predictor)."""

import unittest

from poker44.miner_model.features import PROD_FEATURE_ORDER
from poker44.miner_model.predictor import Poker44Predictor
from poker44.validator.payload_view import prepare_hand_for_miner


def _raw_hand(hero_seat, opener, size_bb):
    """A small raw hand; sanitized -> valid miner-visible hand."""
    return {
        "metadata": {"game_type": "Hold'em", "limit_type": "No Limit", "max_seats": 2,
                     "hero_seat": hero_seat, "button_seat": 2, "sb": 0.01, "bb": 0.02, "ante": 0.0},
        "players": [{"player_uid": "seat_1", "seat": 1, "starting_stack": 10.0},
                    {"player_uid": "seat_2", "seat": 2, "starting_stack": 10.0}],
        "streets": [{"street": "flop", "board_cards": []}],
        "actions": [
            {"action_id": "1", "street": "preflop", "actor_seat": 1, "action_type": "small_blind",
             "amount": 0.01, "raise_to": None, "call_to": None, "normalized_amount_bb": 0.5,
             "pot_before": 0.0, "pot_after": 0.01},
            {"action_id": "2", "street": "preflop", "actor_seat": 2, "action_type": "big_blind",
             "amount": 0.02, "raise_to": None, "call_to": None, "normalized_amount_bb": 1.0,
             "pot_before": 0.01, "pot_after": 0.03},
            {"action_id": "3", "street": "preflop", "actor_seat": opener, "action_type": "raise",
             "amount": size_bb * 0.02, "raise_to": size_bb * 0.02, "call_to": None,
             "normalized_amount_bb": size_bb, "pot_before": 0.03, "pot_after": 0.03 + size_bb * 0.02},
            {"action_id": "4", "street": "preflop", "actor_seat": 3 - opener, "action_type": "call",
             "amount": size_bb * 0.02, "raise_to": None, "call_to": size_bb * 0.02,
             "normalized_amount_bb": size_bb, "pot_before": 0.05, "pot_after": 0.05 + size_bb * 0.02},
            {"action_id": "5", "street": "flop", "actor_seat": 2, "action_type": "check",
             "amount": 0.0, "raise_to": None, "call_to": None, "normalized_amount_bb": 0.0,
             "pot_before": 0.1, "pot_after": 0.1},
            {"action_id": "6", "street": "flop", "actor_seat": 1, "action_type": "bet",
             "amount": 0.06, "raise_to": None, "call_to": None, "normalized_amount_bb": 3.0,
             "pot_before": 0.1, "pot_after": 0.16},
        ],
        "outcome": {"showdown": False},
    }


def _chunk(variant):
    """A sanitized chunk (list of hands) — what the miner receives per scoring unit."""
    hands = [_raw_hand(hero_seat=1 + (i % 2), opener=1 + ((i + variant) % 2), size_bb=2 + (i % 4))
             for i in range(6)]
    return [prepare_hand_for_miner(h) for h in hands]


def _batch(n):
    return [_chunk(v) for v in range(n)]


class _BoomModel:
    n_features_in_ = 35

    def predict_proba(self, X):
        raise RuntimeError("boom")


class PredictorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pred = Poker44Predictor()  # loads model.joblib + model_meta.json

    def test_feature_contract(self):
        self.assertEqual(self.pred.meta["feature_names"], list(PROD_FEATURE_ORDER))
        self.assertEqual(len(PROD_FEATURE_ORDER), 35)
        self.assertEqual(self.pred.p0, 0.85)

    def test_fixed_chunk_deterministic(self):
        batch = _batch(10)
        a = self.pred.predict(batch)
        b = self.pred.predict(batch)
        self.assertEqual(a, b)  # identical input -> identical scores
        self.assertTrue(all(0.0 <= s <= 1.0 for s in a))

    def test_output_length_equals_input(self):
        for n in (0, 1, 5, 10):
            out = self.pred.predict(_batch(n))
            self.assertEqual(len(out), n)

    def test_inference_failure_falls_back_safely(self):
        batch = _batch(5)
        broken = Poker44Predictor()
        broken.model = _BoomModel()
        # default fallback -> neutral 0.5, correct length, valid range
        out = broken.predict(batch)
        self.assertEqual(len(out), len(batch))
        self.assertTrue(all(0.0 <= s <= 1.0 for s in out))
        self.assertTrue(all(s == 0.5 for s in out))
        # custom fallback scorer is used
        out2 = broken.predict(batch, fallback=lambda c: 0.3)
        self.assertEqual(len(out2), len(batch))
        self.assertTrue(all(abs(s - 0.3) < 1e-9 for s in out2))

    def test_malformed_chunk_does_not_crash(self):
        out = self.pred.predict([None, None], fallback=lambda c: 0.5)
        self.assertEqual(out, [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
