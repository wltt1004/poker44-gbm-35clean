"""Tests for the read-only operations monitor (PART 3)."""

import json
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from poker44.miner_model import monitor as MON


def _rec(ts, *, chunk_count=32, latency=0.02, fallback=False, exc=None, caller="5Vali",
         raw=(0.1, 0.9), cal=(0.2, 0.8), frac=0.15, hands_mean=30.0):
    return {
        "ts_utc": ts, "caller_hotkey": caller, "chunk_count": chunk_count,
        "hand_count_dist": {"n_chunks": chunk_count, "mean": hands_mean, "min": 30, "max": 30, "total": chunk_count * 30},
        "raw_score_stats": {"min": raw[0], "mean": sum(raw) / 2, "max": raw[1], "std": 0.1},
        "calibrated_score_stats": {"min": cal[0], "mean": sum(cal) / 2, "max": cal[1], "std": 0.1},
        "positive_at_0.5": {"n_ge_0.5": int(chunk_count * frac), "frac_ge_0.5": frac},
        "inference_seconds": latency, "fallback_used": fallback,
        "model_version": "gbm-35clean-v1", "feature_version": "sn126-35clean-v1",
        "manifest_digest": "d", "exception_type": exc,
        "request_fingerprint": "a" * 64, "chunk_fingerprints": ["a" * 64] * chunk_count,
    }


class MonitorTests(unittest.TestCase):
    def _write(self, d, recs):
        p = Path(d) / "tel.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
        return p

    def test_metrics_computed(self):
        now = datetime.now(timezone.utc)
        recs = [_rec((now - timedelta(minutes=i)).isoformat(), latency=0.01 * (i + 1),
                     caller=("5A" if i % 2 else "5B")) for i in range(10)]
        with TemporaryDirectory() as d:
            p = self._write(d, recs)
            s = MON.build(p, window_hours=24, pm2_name="none", host="127.0.0.1", port=59999)
            m = s["metrics"]
            self.assertEqual(m["queries_in_window"], 10)
            self.assertEqual(sorted(m["distinct_caller_hotkeys"]), ["5A", "5B"])
            self.assertEqual(m["chunk_count_distribution"]["min"], 32)
            self.assertIsNotNone(m["latency_p50"])
            self.assertLessEqual(m["latency_p50"], m["latency_p90"])
            self.assertEqual(m["fallback_count"], 0)
            self.assertEqual(m["exception_count"], 0)

    def test_alerts_fire(self):
        now = datetime.now(timezone.utc)
        recs = [_rec(now.isoformat(), fallback=True),
                _rec(now.isoformat(), exc="ValueError"),
                _rec(now.isoformat(), chunk_count=0)]
        with TemporaryDirectory() as d:
            p = self._write(d, recs)
            s = MON.build(p, window_hours=24, pm2_name="none", host="127.0.0.1", port=59999)
            joined = " ".join(s["alerts"])
            self.assertIn("FALLBACK_USED", joined)
            self.assertIn("INFERENCE_EXCEPTION", joined)
            self.assertIn("EMPTY_OR_MALFORMED", joined)
            self.assertIn("AXON_PORT_UNREACHABLE", joined)  # port 59999 closed

    def test_no_query_60min_alert(self):
        old = (datetime.now(timezone.utc) - timedelta(minutes=120)).isoformat()
        with TemporaryDirectory() as d:
            p = self._write(d, [_rec(old)])
            s = MON.build(p, window_hours=24, pm2_name="none", host="127.0.0.1", port=59999)
            self.assertTrue(any("NO_QUERY_60MIN" in a for a in s["alerts"]))

    def test_empty_telemetry_is_safe(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "none.jsonl"
            s = MON.build(p, window_hours=24, pm2_name="none", host="127.0.0.1", port=59999)
            self.assertEqual(s["records_total"], 0)
            self.assertEqual(s["metrics"]["queries_in_window"], 0)
            # report generation on empty telemetry must not crash
            MON.write_report(s, Path(d) / "r.md")
            self.assertIn("NO TELEMETRY YET", (Path(d) / "r.md").read_text())

    def test_monitor_is_read_only(self):
        # reading records must not modify the telemetry file
        now = datetime.now(timezone.utc).isoformat()
        with TemporaryDirectory() as d:
            p = self._write(d, [_rec(now)])
            before = p.read_bytes()
            MON.build(p, window_hours=24, pm2_name="none", host="127.0.0.1", port=59999)
            self.assertEqual(p.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
