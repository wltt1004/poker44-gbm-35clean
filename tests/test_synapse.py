"""Synapse Intelligence Layer tests (PART 7)."""

import json
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from poker44.miner_model import synapse_analyzer as SA
from poker44.miner_model.calibration import batch_percentile_calibrate
from poker44.miner_model.features import chunk_feature_vector
from poker44.miner_model.predictor import Poker44Predictor, PredictionDiagnostics
from poker44.miner_model.synapse_analyzer import SynapseAnalyzer
from poker44.miner_model.synapse_sink import AnalyzerSink
from poker44.validator.payload_view import prepare_hand_for_miner

_FORBIDDEN = ["action_type", "hole_cards", "board_cards", "player_uid", "pot_before",
              "pot_after", "raise_to", "call_to", "\"actions\"", "\"players\"", "is_bot"]


def _raw_hand(op=1, sz=3):
    return {"metadata": {"game_type": "Hold'em", "limit_type": "No Limit", "max_seats": 2,
            "hero_seat": 1, "button_seat": 2, "sb": 0.01, "bb": 0.02, "ante": 0.0},
            "players": [{"player_uid": "seat_1", "seat": 1, "starting_stack": 10.0},
                        {"player_uid": "seat_2", "seat": 2, "starting_stack": 10.0}],
            "streets": [{"street": "flop", "board_cards": []}],
            "actions": [{"action_id": "1", "street": "preflop", "actor_seat": 1, "action_type": "raise", "amount": sz * 0.02, "raise_to": sz * 0.02, "call_to": None, "normalized_amount_bb": sz, "pot_before": 0.03, "pot_after": 0.03 + sz * 0.02},
                        {"action_id": "2", "street": "flop", "actor_seat": 1, "action_type": "bet", "amount": 0.06, "raise_to": None, "call_to": None, "normalized_amount_bb": 3.0, "pot_before": 0.1, "pot_after": 0.16}]}


def _batch(seed, n=16):
    return [[prepare_hand_for_miner(_raw_hand(1 + (i + seed) % 2, 2 + (i + seed) % 4)) for _ in range(3)]
            for i in range(n)]


def _diag(n=24, mat_value=0.5, final=None):
    X = np.full((n, 35), mat_value, dtype=float)
    X.setflags(write=False)
    fs = tuple(final if final is not None else ([0.6] * (n // 2) + [0.4] * (n - n // 2)))
    return PredictionDiagnostics(request_id=f"rid_{n}_{mat_value}", final_scores=fs,
                                 raw_scores=tuple([0.7] * n), feature_matrix=X,
                                 fallback_used=False, exception_type=None,
                                 feature_duration_ms=1.0, model_duration_ms=1.0,
                                 calibration_duration_ms=1.0, total_duration_ms=3.0)


class SynapseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pred = Poker44Predictor()
        cls.az = SynapseAnalyzer()

    def test_1_predict_outputs_unchanged(self):
        chunks = _batch(0, 12)
        X = np.asarray([chunk_feature_vector(c) for c in chunks], float)
        raw = self.pred.model.predict_proba(X)[:, 1]
        ref = [float(min(1.0, max(0.0, s))) for s in batch_percentile_calibrate(
            raw, p0=self.pred.p0, min_calib_batch=self.pred.min_calib_batch,
            global_anchor_raw=self.pred.global_anchor_raw)]
        self.assertEqual(self.pred.predict(chunks), ref)
        self.assertEqual(list(self.pred.predict_detailed(chunks).final_scores), ref)

    def test_2_concurrent_requests_isolated(self):
        batches = [_batch(s, 16) for s in range(6)]
        expected = [self.pred.predict(b) for b in batches]  # serial ground truth
        with ThreadPoolExecutor(max_workers=6) as ex:
            results = list(ex.map(lambda b: self.pred.predict_detailed(b), batches))
        for i, d in enumerate(results):
            self.assertEqual(list(d.final_scores), expected[i], f"batch {i} diagnostics crossed")
        # request ids all unique
        self.assertEqual(len({d.request_id for d in results}), len(results))

    def test_3_analyzer_uses_precomputed_matrix(self):
        rec = self.az.analyze(_diag(n=24, mat_value=0.5), hand_counts=[3] * 24)
        # feature summaries must reflect the PASSED matrix (all 0.5), not a re-extraction
        for fname, s in rec["feature_summaries"].items():
            self.assertAlmostEqual(s["mean"], 0.5, places=5)
            self.assertAlmostEqual(s["median"], 0.5, places=5)

    def test_4_no_feature_extraction_in_analyzer(self):
        # not imported into the analyzer namespace -> cannot be called
        self.assertFalse(hasattr(SA, "prepare_hand_for_miner"))
        self.assertFalse(hasattr(SA, "chunk_feature_vector"))
        # and no call sites in the source (docstring mentions the names without "(")
        src = Path(SA.__file__).read_text("utf-8")
        self.assertNotIn("prepare_hand_for_miner(", src)
        self.assertNotIn("chunk_feature_vector(", src)

    def test_5_analyzer_record_has_no_raw_payload(self):
        rec = self.az.analyze(_diag(24), hand_counts=[3] * 24, caller_hotkey="5V",
                              request_fingerprint="f", chunk_fingerprints=["c"] * 24)
        blob = json.dumps(rec)
        for bad in _FORBIDDEN:
            self.assertNotIn(bad, blob)

    def test_6_queue_saturation_cannot_block(self):
        with TemporaryDirectory() as d:
            sink = AnalyzerSink(d, enabled=True, max_queue=3)
            sink.close()  # stop the consumer so the queue cannot drain
            t0 = time.perf_counter()
            results = [sink.submit({"ts_utc": "2026-07-25T00:00:00", "i": i}) for i in range(2000)]
            elapsed = time.perf_counter() - t0
            self.assertLess(elapsed, 1.0, "submit must be non-blocking even when full")
            self.assertGreater(sink.dropped, 0)
            self.assertFalse(all(results))  # some were dropped, not blocked

    def test_7_malformed_inputs_produce_safe_record(self):
        # empty feature matrix + None raw scores (fallback-like)
        X = np.empty((0, 35)); X.setflags(write=False)
        d = PredictionDiagnostics("r", (0.5, 0.5), None, X, True, "ValueError", 0, 0, 0, 1.0)
        rec = self.az.analyze(d, hand_counts=[0, 0])
        self.assertIn("request_shape", rec)
        self.assertEqual(rec["request_shape"]["empty_chunk_count"], 2)
        # a broken diagnostics object -> analyzer_error record, no raise
        class Broken: pass
        rec2 = self.az.analyze(Broken())
        self.assertIn("analyzer_error", rec2)

    def test_8_drift_is_deterministic(self):
        d = _diag(30, 0.4)
        r1 = self.az.analyze(d, hand_counts=[3] * 30)
        r2 = self.az.analyze(d, hand_counts=[3] * 30)
        self.assertEqual(r1["feature_drift"], r2["feature_drift"])
        self.assertEqual(r1["score_diagnostics"], r2["score_diagnostics"])

    def test_9_small_requests_low_confidence(self):
        rec = self.az.analyze(_diag(5, 0.4), hand_counts=[3] * 5)
        self.assertEqual(rec["feature_drift"]["confidence"], "low")

    def test_10_report_never_claims_labels_or_true_reward(self):
        from poker44.miner_model import synapse_report as SR
        recs = [self.az.analyze(_diag(24), hand_counts=[3] * 24, caller_hotkey="5V")]
        agg = SR.aggregate(recs)
        with TemporaryDirectory() as d:
            p = Path(d) / "r.md"
            SR.write_report("2026-07-25", agg, {}, p)
            txt = p.read_text()
        # required disclaimers present
        self.assertIn("UNKNOWN WITHOUT LABELS", txt)
        self.assertIn("NOT LIVE SCORE", txt)
        self.assertIn("NOT the true bot fraction", txt)
        # must not present a computed live AP/Recall/TSQ value
        for banned in ["live AP =", "Recall@5%FPR =", "TSQ =", "true bot fraction ="]:
            self.assertNotIn(banned, txt)


if __name__ == "__main__":
    unittest.main()
