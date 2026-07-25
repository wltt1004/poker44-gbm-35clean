"""Tests for miner telemetry: logging, failure isolation, no raw-payload leakage (PART 5)."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from poker44.miner_model.telemetry import MinerTelemetry

_FORBIDDEN_SUBSTRINGS = [
    "action_type", "hole_cards", "board_cards", "player_uid", "pot_before",
    "pot_after", "raise_to", "call_to", "actions", "\"players\"", "is_bot", "label",
]
_ALLOWED_KEYS = {
    "ts_utc", "caller_hotkey", "chunk_count", "hand_count_dist", "raw_score_stats",
    "calibrated_score_stats", "positive_at_0.5", "inference_seconds", "fallback_used",
    "model_version", "feature_version", "manifest_digest", "exception_type",
    "request_fingerprint", "chunk_fingerprints",
}


class TelemetryTests(unittest.TestCase):
    def test_logging_writes_metadata_record(self):
        with TemporaryDirectory() as d:
            t = MinerTelemetry(Path(d) / "t.jsonl", enabled=True)
            t.log_inference(chunk_count=3, hand_counts=[30, 30, 12],
                            raw_scores=[0.1, 0.9, 0.5], calibrated_scores=[0.2, 0.8, 0.5],
                            duration_s=0.01, fallback_used=False, caller_hotkey="5Vali",
                            model_version="gbm-35clean-v1", feature_version="sn126-35clean-v1",
                            manifest_digest="abc123")
            lines = (Path(d) / "t.jsonl").read_text().strip().splitlines()
            self.assertEqual(len(lines), 1)
            rec = json.loads(lines[0])
            self.assertEqual(set(rec.keys()), _ALLOWED_KEYS)
            self.assertEqual(rec["chunk_count"], 3)
            self.assertEqual(rec["hand_count_dist"]["total"], 72)
            self.assertAlmostEqual(rec["calibrated_score_stats"]["max"], 0.8)
            self.assertEqual(rec["positive_at_0.5"]["n_ge_0.5"], 2)
            self.assertEqual(rec["model_version"], "gbm-35clean-v1")

    def test_no_raw_hand_payload_is_written(self):
        with TemporaryDirectory() as d:
            t = MinerTelemetry(Path(d) / "t.jsonl", enabled=True)
            t.log_inference(chunk_count=2, hand_counts=[5, 5],
                            raw_scores=[0.3, 0.7], calibrated_scores=[0.3, 0.7])
            blob = (Path(d) / "t.jsonl").read_text()
            for bad in _FORBIDDEN_SUBSTRINGS:
                self.assertNotIn(bad, blob, f"telemetry leaked forbidden token: {bad}")
            rec = json.loads(blob.strip())
            self.assertLessEqual(set(rec.keys()), _ALLOWED_KEYS)

    def test_fingerprints_are_privacy_safe_hashes(self):
        import hashlib
        with TemporaryDirectory() as d:
            t = MinerTelemetry(Path(d) / "t.jsonl", enabled=True)
            cfps = [hashlib.sha256(f"chunk{i}".encode()).hexdigest() for i in range(3)]
            rfp = hashlib.sha256("|".join(cfps).encode()).hexdigest()
            t.log_inference(chunk_count=3, hand_counts=[5, 5, 5], calibrated_scores=[0.2, 0.8, 0.5],
                            request_fingerprint=rfp, chunk_fingerprints=cfps)
            rec = json.loads((Path(d) / "t.jsonl").read_text().strip())
            self.assertEqual(rec["request_fingerprint"], rfp)
            self.assertEqual(rec["chunk_fingerprints"], cfps)
            # fingerprints are 64-hex one-way hashes (no raw hands recoverable)
            self.assertTrue(all(len(c) == 64 and all(ch in "0123456789abcdef" for ch in c)
                                for c in rec["chunk_fingerprints"]))
            blob = (Path(d) / "t.jsonl").read_text()
            for bad in _FORBIDDEN_SUBSTRINGS:
                self.assertNotIn(bad, blob)

    def test_disabled_writes_nothing(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "t.jsonl"
            t = MinerTelemetry(p, enabled=False)
            t.log_inference(chunk_count=1, calibrated_scores=[0.5])
            self.assertFalse(p.exists())

    def test_failure_isolation_never_raises(self):
        # unwritable path -> must not raise
        t = MinerTelemetry("/proc/should_not_exist/telemetry.jsonl", enabled=True)
        t.log_inference(chunk_count=1, calibrated_scores=[0.5])  # no exception
        # garbage inputs -> must not raise
        class Boom:
            def __iter__(self):
                raise RuntimeError("boom")
        t2 = None
        with TemporaryDirectory() as d:
            t2 = MinerTelemetry(Path(d) / "t.jsonl", enabled=True)
            t2.log_inference(chunk_count="x", hand_counts=Boom(), raw_scores=Boom(),
                             calibrated_scores=Boom(), duration_s="nope")
        # broken _write -> still no raise
        with TemporaryDirectory() as d:
            t3 = MinerTelemetry(Path(d) / "t.jsonl", enabled=True)
            t3._write = lambda rec: (_ for _ in ()).throw(RuntimeError("write boom"))
            t3.log_inference(chunk_count=1, calibrated_scores=[0.5])

    def test_telemetry_cannot_change_inference_scores(self):
        # emulate the miner's failure-isolated call site with a raising telemetry
        class RaisingTelemetry:
            def log_inference(self, **kw):
                raise RuntimeError("telemetry blew up")

        def emit(telemetry, scores):
            try:
                telemetry.log_inference(chunk_count=len(scores), calibrated_scores=scores)
            except Exception:
                return  # miner._emit_telemetry pattern
        scores = [0.1, 0.9, 0.5]
        before = list(scores)
        emit(RaisingTelemetry(), scores)      # must not raise
        self.assertEqual(scores, before)      # inputs unchanged

    def test_env_disable(self):
        import os
        old = os.environ.get("POKER44_TELEMETRY_ENABLED")
        try:
            os.environ["POKER44_TELEMETRY_ENABLED"] = "0"
            self.assertFalse(MinerTelemetry(enabled=None).enabled)
            os.environ["POKER44_TELEMETRY_ENABLED"] = "1"
            self.assertTrue(MinerTelemetry(enabled=None).enabled)
        finally:
            if old is None:
                os.environ.pop("POKER44_TELEMETRY_ENABLED", None)
            else:
                os.environ["POKER44_TELEMETRY_ENABLED"] = old


if __name__ == "__main__":
    unittest.main()
