"""Read-only operations monitor for the SN126 miner.

Parses the telemetry JSONL (and optionally PM2) and reports metrics + alerts.
It NEVER writes model outputs, never blocks inference, never calls private
/internal endpoints. Safe to run alongside the live miner.

Usage:
    python -m poker44.miner_model.monitor                 # 24h summary + alerts
    python -m poker44.miner_model.monitor --window 1h
    python -m poker44.miner_model.monitor --report --csv  # write artifacts/
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import socket
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_TELEMETRY = _REPO / "logs" / "miner_telemetry.jsonl"
_MODEL = _REPO / "poker44" / "miner_model" / "model.joblib"
EXPECTED_MODEL_SHA256 = "b8b7fc78586568e4480ed83a880eb2639001605215a1a6631ed6206c89a9194d"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(s: Any) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None


def _pct(vals: List[float], q: float) -> Optional[float]:
    if not vals:
        return None
    xs = sorted(vals)
    k = max(0, min(len(xs) - 1, int(round((q / 100.0) * (len(xs) - 1)))))
    return round(xs[k], 6)


def read_records(path: Path, window: Optional[timedelta]) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    cutoff = (_now() - window) if window else None
    out: List[Dict[str, Any]] = []
    for line in path.read_text("utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if cutoff is not None:
            ts = _parse_ts(rec.get("ts_utc"))
            if ts is None or ts < cutoff:
                continue
        out.append(rec)
    return out


def _dist(counts: List[int]) -> Dict[str, Any]:
    if not counts:
        return {}
    from collections import Counter
    c = Counter(counts)
    return {"min": min(counts), "max": max(counts),
            "mean": round(sum(counts) / len(counts), 3),
            "histogram": dict(sorted(c.items()))}


def compute_metrics(records: List[Dict[str, Any]], all_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(records)
    latencies = [r["inference_seconds"] for r in records if isinstance(r.get("inference_seconds"), (int, float))]
    chunk_counts = [int(r["chunk_count"]) for r in records if isinstance(r.get("chunk_count"), int)]
    callers = sorted({r.get("caller_hotkey") for r in records if r.get("caller_hotkey")})
    fallback = sum(1 for r in records if r.get("fallback_used"))
    exceptions = [r.get("exception_type") for r in records if r.get("exception_type")]
    empty = sum(1 for r in records if (r.get("chunk_count") in (0, None)))
    frac_ge = [r["positive_at_0.5"]["frac_ge_0.5"] for r in records
               if isinstance(r.get("positive_at_0.5"), dict) and "frac_ge_0.5" in r["positive_at_0.5"]]
    hands_per_chunk = []
    for r in records:
        hcd = r.get("hand_count_dist")
        if isinstance(hcd, dict) and "mean" in hcd:
            hands_per_chunk.append(round(float(hcd["mean"]), 3))

    def _agg(stat_key, field):
        vals = [r[stat_key][field] for r in records
                if isinstance(r.get(stat_key), dict) and field in r[stat_key]]
        if not vals:
            return None
        return {"min": round(min(vals), 6), "mean": round(sum(vals) / len(vals), 6),
                "max": round(max(vals), 6)}

    ts_all = [_parse_ts(r.get("ts_utc")) for r in all_records]
    ts_all = [t for t in ts_all if t]
    last_query = max(ts_all).isoformat() if ts_all else None
    first_query = min(ts_all).isoformat() if ts_all else None

    return {
        "queries_in_window": n,
        "first_query_overall": first_query,
        "last_query_overall": last_query,
        "distinct_caller_hotkeys": callers,
        "queries_by_caller": {c: sum(1 for r in records if r.get("caller_hotkey") == c) for c in callers},
        "chunk_count_distribution": _dist(chunk_counts),
        "hands_per_chunk_mean_distribution": _dist([int(round(x)) for x in hands_per_chunk]) if hands_per_chunk else {},
        "latency_p50": _pct(latencies, 50), "latency_p90": _pct(latencies, 90), "latency_p99": _pct(latencies, 99),
        "raw_score_stats_agg": {"min": _agg("raw_score_stats", "min"),
                                "mean": _agg("raw_score_stats", "mean"),
                                "max": _agg("raw_score_stats", "max"),
                                "std": _agg("raw_score_stats", "std")},
        "calibrated_score_stats_agg": {"min": _agg("calibrated_score_stats", "min"),
                                       "mean": _agg("calibrated_score_stats", "mean"),
                                       "max": _agg("calibrated_score_stats", "max"),
                                       "std": _agg("calibrated_score_stats", "std")},
        "frac_ge_0.5_mean": round(sum(frac_ge) / len(frac_ge), 6) if frac_ge else None,
        "fallback_count": fallback,
        "fallback_rate": round(fallback / n, 6) if n else None,
        "exception_count": len(exceptions),
        "exception_types": dict(__import__("collections").Counter(exceptions)),
        "empty_or_malformed_count": empty,
    }


def pm2_status(name: str) -> Dict[str, Any]:
    try:
        out = subprocess.run(["pm2", "jlist"], capture_output=True, text=True, timeout=10)
        procs = json.loads(out.stdout or "[]")
        for p in procs:
            if p.get("name") == name:
                pm = p.get("pm2_env", {})
                up_ms = pm.get("pm_uptime")
                uptime_s = int((_now().timestamp() * 1000 - up_ms) / 1000) if up_ms else None
                return {"found": True, "status": pm.get("status"),
                        "restart_time": pm.get("restart_time"), "uptime_seconds": uptime_s,
                        "cpu": (p.get("monit") or {}).get("cpu"), "memory": (p.get("monit") or {}).get("memory")}
        return {"found": False}
    except Exception as exc:
        return {"found": False, "error": str(exc)}


def port_reachable(host: str, port: int, timeout: float = 6.0) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        return s.connect_ex((host, port)) == 0
    except Exception:
        return False
    finally:
        s.close()


def model_hash_ok() -> Dict[str, Any]:
    try:
        h = hashlib.sha256(_MODEL.read_bytes()).hexdigest()
        return {"sha256": h, "matches_expected": h == EXPECTED_MODEL_SHA256}
    except Exception as exc:
        return {"error": str(exc), "matches_expected": False}


def disk_status(path: Path) -> Dict[str, Any]:
    try:
        du = shutil.disk_usage(path)
        pct = round(100 * du.used / du.total, 2)
        return {"used_pct": pct, "free_gb": round(du.free / 1e9, 2)}
    except Exception as exc:
        return {"error": str(exc)}


def evaluate_alerts(metrics, pm2, disk, model_hash, port_ok, telemetry_path,
                    *, window_appears_active: Optional[bool] = None) -> List[str]:
    alerts = []
    last = _parse_ts(metrics.get("last_query_overall"))
    gap_min = (( _now() - last).total_seconds() / 60) if last else None
    if gap_min is not None and gap_min > 60 and window_appears_active is not False:
        alerts.append(f"NO_QUERY_60MIN: last query {round(gap_min)}min ago"
                      + ("" if window_appears_active is None else " while window appears active"))
    if metrics.get("fallback_count", 0) >= 1:
        alerts.append(f"FALLBACK_USED: {metrics['fallback_count']} record(s)")
    if metrics.get("exception_count", 0) >= 1:
        alerts.append(f"INFERENCE_EXCEPTION: {metrics['exception_types']}")
    if metrics.get("empty_or_malformed_count", 0) >= 1:
        alerts.append(f"EMPTY_OR_MALFORMED_REQUESTS: {metrics['empty_or_malformed_count']}")
    if pm2.get("found") and (pm2.get("restart_time") or 0) > 0:
        alerts.append(f"PM2_RESTARTS: restart_time={pm2['restart_time']}")
    if pm2.get("found") and pm2.get("status") not in (None, "online"):
        alerts.append(f"PM2_STATUS_NOT_ONLINE: {pm2.get('status')}")
    if port_ok is False:
        alerts.append("AXON_PORT_UNREACHABLE")
    if not model_hash.get("matches_expected", False):
        alerts.append(f"MODEL_HASH_MISMATCH: {model_hash.get('sha256')}")
    if isinstance(disk.get("used_pct"), (int, float)) and disk["used_pct"] > 80:
        alerts.append(f"DISK_ABOVE_80PCT: {disk['used_pct']}%")
    try:
        size = Path(telemetry_path).stat().st_size if Path(telemetry_path).exists() else 0
        if size > 100_000_000:
            alerts.append(f"TELEMETRY_LOG_LARGE: {round(size/1e6,1)}MB")
    except Exception:
        pass
    return alerts


def build(telemetry_path: Path, *, window_hours: int, pm2_name: str,
          host: str, port: int) -> Dict[str, Any]:
    all_recs = read_records(telemetry_path, None)
    win_recs = read_records(telemetry_path, timedelta(hours=window_hours))
    hour_recs = read_records(telemetry_path, timedelta(hours=1))
    metrics = compute_metrics(win_recs, all_recs)
    metrics["queries_last_1h"] = len(hour_recs)
    metrics["queries_last_24h"] = len(read_records(telemetry_path, timedelta(hours=24)))
    pm2 = pm2_status(pm2_name)
    disk = disk_status(_REPO)
    disk["telemetry_log_size_bytes"] = telemetry_path.stat().st_size if telemetry_path.exists() else 0
    mh = model_hash_ok()
    pok = port_reachable(host, port)
    alerts = evaluate_alerts(metrics, pm2, disk, mh, pok, telemetry_path)
    return {"window_hours": window_hours, "generated_utc": _now().isoformat(),
            "telemetry_path": str(telemetry_path), "records_total": len(all_recs),
            "process": pm2, "disk": disk, "model_hash": mh,
            "axon_port": {"host": host, "port": port, "reachable": pok},
            "metrics": metrics, "alerts": alerts}


def write_csv(records: List[Dict[str, Any]], path: Path) -> None:
    import csv
    fields = ["ts_utc", "caller_hotkey", "chunk_count", "inference_seconds", "fallback_used",
              "exception_type", "model_version", "feature_version", "request_fingerprint"]
    stat_fields = ["raw_min", "raw_mean", "raw_max", "raw_std",
                   "cal_min", "cal_mean", "cal_max", "cal_std", "frac_ge_0.5", "hands_mean", "n_chunks"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields + stat_fields)
        w.writeheader()
        for r in records:
            rs = r.get("raw_score_stats") or {}
            cs = r.get("calibrated_score_stats") or {}
            pos = r.get("positive_at_0.5") or {}
            hcd = r.get("hand_count_dist") or {}
            row = {k: r.get(k) for k in fields}
            row.update({
                "raw_min": rs.get("min"), "raw_mean": rs.get("mean"), "raw_max": rs.get("max"), "raw_std": rs.get("std"),
                "cal_min": cs.get("min"), "cal_mean": cs.get("mean"), "cal_max": cs.get("max"), "cal_std": cs.get("std"),
                "frac_ge_0.5": pos.get("frac_ge_0.5"), "hands_mean": hcd.get("mean"), "n_chunks": hcd.get("n_chunks"),
            })
            w.writerow(row)


def write_report(summary: Dict[str, Any], path: Path) -> None:
    m = summary["metrics"]
    L = []
    A = L.append
    A("# SN126 Live Round 1 — Operations Report\n")
    A(f"Generated (UTC): {summary['generated_utc']} | telemetry: {summary['telemetry_path']} "
      f"| records total: {summary['records_total']}\n")
    if summary["records_total"] == 0:
        A("\n> **NO TELEMETRY YET** — the miner has not recorded any inference. This report is a "
          "scaffold; re-run after a live round produces telemetry.\n")
    A("\n## A. Operational reliability")
    A(f"- process: {summary['process']}")
    A(f"- axon port reachable: {summary['axon_port']}")
    A(f"- model hash matches validated: {summary['model_hash'].get('matches_expected')}")
    A(f"- disk: {summary['disk']}")
    A(f"- queries (1h / 24h / window): {m.get('queries_last_1h')} / {m.get('queries_last_24h')} / {m['queries_in_window']}")
    A(f"- fallback rate: {m['fallback_rate']} | exceptions: {m['exception_count']} {m['exception_types']}")
    A(f"- empty/malformed: {m['empty_or_malformed_count']}")
    A("\n## B. Request & score distribution")
    A(f"- chunk-count distribution: {m['chunk_count_distribution']}")
    A(f"- hands/chunk (mean) distribution: {m['hands_per_chunk_mean_distribution']}")
    A(f"- distinct callers: {m['distinct_caller_hotkeys']}")
    A(f"- queries by caller: {m['queries_by_caller']}")
    A(f"- latency p50/p90/p99: {m['latency_p50']} / {m['latency_p90']} / {m['latency_p99']}")
    A(f"- raw score agg: {m['raw_score_stats_agg']}")
    A(f"- calibrated score agg: {m['calibrated_score_stats_agg']}")
    A(f"- fraction >= 0.5 (mean): {m['frac_ge_0.5_mean']}")
    A("\n## Alerts")
    if summary["alerts"]:
        for a in summary["alerts"]:
            A(f"- ⚠️ {a}")
    else:
        A("- none")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), "utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--telemetry", default=str(_DEFAULT_TELEMETRY))
    ap.add_argument("--window", default="24h")
    ap.add_argument("--pm2-name", default="poker44_miner")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--csv", action="store_true")
    args = ap.parse_args()
    wh = 1 if args.window == "1h" else 24
    tpath = Path(args.telemetry)
    summary = build(tpath, window_hours=wh, pm2_name=args.pm2_name, host=args.host, port=args.port)
    print(json.dumps(summary, indent=2, default=str))
    art = _REPO / "artifacts"
    if args.csv:
        write_csv(read_records(tpath, timedelta(hours=24)), art / "sn126_live_round_1_telemetry.csv")
        print(f"wrote {art/'sn126_live_round_1_telemetry.csv'}")
    if args.report:
        write_report(summary, art / "sn126_live_round_1_report.md")
        print(f"wrote {art/'sn126_live_round_1_report.md'}")


if __name__ == "__main__":
    main()
