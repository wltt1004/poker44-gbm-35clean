"""Tests for the validator-faithful simulator (PART 5)."""

import unittest

import numpy as np

import poker44.score.scoring as scoring
from poker44.miner_model.calibration import batch_percentile_calibrate
from poker44.miner_model.features import chunk_feature_vector
from poker44.miner_model.predictor import Poker44Predictor
from poker44.validator.payload_view import prepare_hand_for_miner
from sn126_research.validator_sim import request_simulator as RS
from sn126_research.validator_sim.competition_simulator import simulate_competitions
from sn126_research.validator_sim.report import walkforward_select


def _raw_hand(op=1, sz=3):
    return {"metadata": {"game_type": "Hold'em", "limit_type": "No Limit", "max_seats": 2,
            "hero_seat": 1, "button_seat": 2, "sb": 0.01, "bb": 0.02, "ante": 0.0},
            "players": [{"player_uid": "seat_1", "seat": 1, "starting_stack": 10.0},
                        {"player_uid": "seat_2", "seat": 2, "starting_stack": 10.0}],
            "streets": [{"street": "flop", "board_cards": []}],
            "actions": [
              {"action_id": "1", "street": "preflop", "actor_seat": 1, "action_type": "small_blind", "amount": 0.01, "raise_to": None, "call_to": None, "normalized_amount_bb": 0.5, "pot_before": 0.0, "pot_after": 0.01},
              {"action_id": "2", "street": "preflop", "actor_seat": 2, "action_type": "big_blind", "amount": 0.02, "raise_to": None, "call_to": None, "normalized_amount_bb": 1.0, "pot_before": 0.01, "pot_after": 0.03},
              {"action_id": "3", "street": "preflop", "actor_seat": op, "action_type": "raise", "amount": sz * 0.02, "raise_to": sz * 0.02, "call_to": None, "normalized_amount_bb": sz, "pot_before": 0.03, "pot_after": 0.03 + sz * 0.02},
              {"action_id": "4", "street": "flop", "actor_seat": 1, "action_type": "bet", "amount": 0.06, "raise_to": None, "call_to": None, "normalized_amount_bb": 3.0, "pot_before": 0.1, "pot_after": 0.16}]}


class SimulatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pred = Poker44Predictor()

    def test_1_simulator_uses_official_reward(self):
        # the simulator must import — not reimplement — poker44.score.scoring.reward
        self.assertIs(RS.official_reward, scoring.reward)

    def test_2_no_future_release_leaks_into_selection(self):
        dates = ["d0", "d1", "d2", "d3"]
        cands = [0.70, 0.90]
        mean_reward = {
            ("d0", 0.70): 0.8, ("d0", 0.90): 0.1,
            ("d1", 0.70): 0.8, ("d1", 0.90): 0.1,
            ("d2", 0.70): 0.0, ("d2", 0.90): 0.99,   # future/held-out strongly prefers 0.90
            ("d3", 0.70): 0.0, ("d3", 0.90): 0.99,
        }
        rows = walkforward_select(dates, mean_reward, cands, production_p0=0.70)
        by_held = {r["held_out_release"]: r["selected_p0_from_past"] for r in rows}
        # if d2/d3 leaked, selection for d2 would be 0.90; correct walk-forward keeps 0.70
        self.assertEqual(by_held["d2"], 0.70)
        self.assertEqual(by_held["d1"], 0.70)

    def test_3_request_class_counts_match_scenario(self):
        raw = np.concatenate([np.linspace(0.5, 1.0, 40), np.linspace(0.0, 0.6, 40)])
        label = np.array([1] * 40 + [0] * 40)
        pool = RS.ReleasePool("t", raw, label, self.pred.model_version)
        rng = np.random.RandomState(3)
        for cc, nb in [(8, 1), (16, 2), (32, 5), (100, 50)]:
            idx = RS.sample_request(pool, cc, nb, rng)
            labels = pool.label[idx]
            self.assertEqual(len(idx), cc)
            self.assertEqual(int((labels == 1).sum()), nb)
            self.assertEqual(int((labels == 0).sum()), cc - nb)

    def test_4_output_scores_are_exactly_production(self):
        # simulator raw+p0=0.85 calibration must equal Poker44Predictor.predict()
        chunks = [[prepare_hand_for_miner(_raw_hand(1 + i % 2, 2 + i % 4)) for _ in range(3)]
                  for i in range(12)]
        production = np.asarray(self.pred.predict(chunks))
        X = np.asarray([chunk_feature_vector(c) for c in chunks], float)
        raw = self.pred.model.predict_proba(X)[:, 1]
        sim = batch_percentile_calibrate(raw, p0=self.pred.p0,
                                         min_calib_batch=self.pred.min_calib_batch,
                                         global_anchor_raw=self.pred.global_anchor_raw)
        self.assertTrue(np.allclose(production, sim, atol=1e-9))

    def test_5_five_round_simulation_reproducible_from_seed(self):
        arrs = [np.random.RandomState(i).uniform(0, 1, 500) for i in range(5)]
        a = simulate_competitions(arrs, n_competitions=3000, thresholds=[0.6, 0.7], seed=99)
        b = simulate_competitions(arrs, n_competitions=3000, thresholds=[0.6, 0.7], seed=99)
        self.assertTrue(np.allclose(a["composite"], b["composite"]))
        self.assertEqual(a["composite_stats"], b["composite_stats"])
        c = simulate_competitions(arrs, n_competitions=3000, thresholds=[0.6], seed=100)
        self.assertFalse(np.allclose(a["composite"], c["composite"]))  # different seed differs


if __name__ == "__main__":
    unittest.main()
