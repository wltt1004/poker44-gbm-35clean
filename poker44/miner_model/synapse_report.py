"""Daily Synapse Intelligence report.

Reads the per-date analyzer JSONL and emits a markdown report + CSV summary.
Every conclusion is classified OBSERVED / INFERRED / UNKNOWN WITHOUT LABELS.
It NEVER computes or claims live AP, Recall@5%FPR, TSQ, bot ratio, or reward from
unlabeled requests. PART 6 maps observed request shapes to a SIMULATED reward
RANGE (not a live score) using the existing validator-simulation CSV.

Run: python -m poker44.miner_model.synapse_report --date YYYY-MM-DD
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO = Path(__file__).resolve().parents[2]
_SYNAPSE_DIR = Path(os.getenv("POKER44_SYNAPSE_DIR") or (_REPO / "logs" / "synapse"))
_SCENARIO_CSV = _REPO / "artifacts" / "sn126_request_scenarios.csv"
_ART = _REPO / "artifacts"


def _pct(vals: List[float], q: float) -> Optional[float]:
    if not vals:
        return None
    xs = sorted(vals)
    k = max(0, min(len(xs) - 1, int(round((q / 100) * (len(xs) - 1)))))
    return round(xs[k], 6)


def read_day(date: str, directory: Path = _SYNAPSE_DIR) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    if not directory.exists():
        return recs
    for p in sorted(directory.glob(f"synapse_analyzer_{date}.jsonl*")):
        for line in p.read_text("utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                continue
    return recs


def load_sim_ranges() -> Dict[int, Dict[str, float]]:
    """chunk_count -> conservative SIMULATED reward range across bot fractions (latest release)."""
    if not _SCENARIO_CSV.exists():
        return {}
    by_cc: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    latest = None
    rows = list(csv.DictReader(open(_SCENARIO_CSV)))
    dates = sorted({r["release"] for r in rows})
    latest = dates[-1] if dates else None
    for r in rows:
        if r.get("metric") == "reward" and r.get("release") == latest:
            by_cc[int(r["chunk_count"])].append(r)
    out = {}
    for cc, rs in by_cc.items():
        worsts = [float(r["worst"]) for r in rs]
        p10s = [float(r["p10"]) for r in rs]
        p90s = [float(r["p90"]) for r in rs]
        out[cc] = {"reward_low": round(min(worsts + p10s), 4), "reward_high": round(max(p90s), 4),
                   "release": latest}
    return out


def aggregate(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(recs)
    callers = Counter(r.get("caller_hotkey") for r in recs if r.get("caller_hotkey"))
    chunk_counts = [r["request_shape"]["chunk_count"] for r in recs if r.get("request_shape")]
    hands_means = [r["request_shape"]["hands_per_chunk"]["mean"] for r in recs
                   if r.get("request_shape") and r["request_shape"].get("hands_per_chunk")]
    lat = [r["timings_ms"]["total"] for r in recs if r.get("timings_ms")]
    fallback = sum(1 for r in recs if r.get("fallback_used"))
    exceptions = Counter(r.get("exception_type") for r in recs if r.get("exception_type"))
    analyzer_errs = sum(1 for r in recs if r.get("analyzer_error"))
    frac_ge = [r["score_diagnostics"]["predicted_positive_fraction"] for r in recs
               if r.get("score_diagnostics") and r["score_diagnostics"].get("predicted_positive_fraction") is not None]
    margins = [r["score_diagnostics"]["top_score_margin"] for r in recs
               if r.get("score_diagnostics") and r["score_diagnostics"].get("top_score_margin") is not None]
    severities = Counter(r["feature_drift"]["severity"] for r in recs
                         if r.get("feature_drift") and r["feature_drift"].get("available"))
    max_psis = [r["feature_drift"]["max_psi"] for r in recs
                if r.get("feature_drift") and r["feature_drift"].get("max_psi") is not None]
    top_drift = Counter()
    drift_shift = defaultdict(list)
    for r in recs:
        fd = r.get("feature_drift") or {}
        for item in (fd.get("top10_drifting_features") or []):
            top_drift[item["feature"]] += 1
            drift_shift[item["feature"]].append(abs(item.get("standardized_mean_shift", 0.0)))
    repeated_req = sum(1 for r in recs if (r.get("repetition") or {}).get("request_seen_count_local", 0) > 1)
    repeated_chunk = sum((r.get("repetition") or {}).get("repeated_chunk_count_local", 0) for r in recs)
    versions = Counter((r.get("model_version"), r.get("feature_version"), r.get("manifest_digest")) for r in recs)
    top_drift_ranked = sorted(top_drift.items(), key=lambda kv: kv[1], reverse=True)[:10]
    return {
        "query_count": n,
        "caller_distribution": dict(callers),
        "chunk_count_hist": dict(sorted(Counter(chunk_counts).items())),
        "hands_per_chunk_mean_range": [round(min(hands_means), 3), round(max(hands_means), 3)] if hands_means else None,
        "latency_ms": {"p50": _pct(lat, 50), "p90": _pct(lat, 90), "p99": _pct(lat, 99)},
        "fallback_count": fallback, "fallback_rate": round(fallback / n, 5) if n else None,
        "exception_types": dict(exceptions), "analyzer_error_count": analyzer_errs,
        "predicted_positive_fraction_mean": round(sum(frac_ge) / len(frac_ge), 5) if frac_ge else None,
        "top_score_margin_mean": round(sum(margins) / len(margins), 5) if margins else None,
        "drift_severity_hist": dict(severities),
        "max_psi_mean": round(sum(max_psis) / len(max_psis), 5) if max_psis else None,
        "top_drifting_features": [{"feature": f, "appearances": c,
                                   "mean_abs_shift": round(sum(drift_shift[f]) / len(drift_shift[f]), 4)}
                                  for f, c in top_drift_ranked],
        "repeated_request_count": repeated_req, "repeated_chunk_count": repeated_chunk,
        "versions_seen": [{"model": k[0], "feature": k[1], "manifest": k[2], "count": v}
                          for k, v in versions.items()],
    }


def write_report(date: str, agg: Dict[str, Any], sim: Dict[int, Dict[str, float]], path: Path):
    L, A = [], None
    A = L.append
    A(f"# SN126 Synapse Intelligence Report — {date}\n")
    A(f"Records: **{agg['query_count']}**. All conclusions are labeled OBSERVED / INFERRED / "
      "UNKNOWN WITHOUT LABELS. No live AP / Recall / TSQ / bot ratio / reward is computed from "
      "unlabeled requests.\n")
    if agg["query_count"] == 0:
        A("\n> **NO ANALYZER RECORDS for this date.** Scaffold only.\n")
    A("\n## Query & callers  [OBSERVED]")
    A(f"- queries: {agg['query_count']} | callers: {agg['caller_distribution']}")
    A(f"- versions seen: {agg['versions_seen']}")
    A("\n## Request shapes  [OBSERVED]")
    A(f"- chunk-count histogram: {agg['chunk_count_hist']}")
    A(f"- hands/chunk mean range: {agg['hands_per_chunk_mean_range']}")
    A("\n## Score distribution  [OBSERVED]")
    A(f"- predicted-positive fraction (>=0.5) mean: {agg['predicted_positive_fraction_mean']}  "
      "*(model prediction rate — NOT the true bot fraction)*")
    A(f"- top-score margin mean: {agg['top_score_margin_mean']}")
    A("\n## Feature drift vs public training profile  [OBSERVED shift; magnitude INFERRED]")
    A(f"- severity histogram: {agg['drift_severity_hist']} | mean max-PSI: {agg['max_psi_mean']}")
    A("- top drifting features:")
    for f in agg["top_drifting_features"]:
        A(f"  - {f['feature']}: seen in top-10 {f['appearances']}x, mean |std-shift| {f['mean_abs_shift']}")
    A("\n## Reliability  [OBSERVED]")
    A(f"- fallback rate: {agg['fallback_rate']} | exceptions: {agg['exception_types']} | "
      f"analyzer errors: {agg['analyzer_error_count']}")
    A(f"- latency p50/p90/p99 (ms): {agg['latency_ms']['p50']} / {agg['latency_ms']['p90']} / {agg['latency_ms']['p99']}")
    A(f"- repeated requests: {agg['repeated_request_count']} | repeated chunks: {agg['repeated_chunk_count']}")
    A("\n## PART 6 — SIMULATED reward range for observed shapes  [SIMULATED, NOT LIVE SCORE]")
    A("> Live bot fraction is UNKNOWN WITHOUT LABELS. The range below spans plausible class ratios "
      "from the offline simulator; it is NOT a live competition score.")
    A("| observed chunk_count | queries | SIMULATED reward range (across class ratios) | sim release |")
    A("|---|---|---|---|")
    for cc, cnt in sorted(agg["chunk_count_hist"].items()):
        rng = sim.get(int(cc))
        if rng:
            A(f"| {cc} | {cnt} | {rng['reward_low']} – {rng['reward_high']} | {rng['release']} |")
        else:
            A(f"| {cc} | {cnt} | (no simulated shape) | — |")
    A("\n## Classification summary")
    A("- OBSERVED: query counts, callers, request shapes, score/latency distributions, fallback/errors, repetition.")
    A("- INFERRED: drift magnitude/severity (depends on training-profile representativeness).")
    A("- UNKNOWN WITHOUT LABELS: true bot fraction, live AP/Recall/TSQ, actual competition reward.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), "utf-8")


def write_csv(date: str, agg: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        flat = {
            "date": date, "query_count": agg["query_count"],
            "fallback_rate": agg["fallback_rate"], "analyzer_error_count": agg["analyzer_error_count"],
            "predicted_positive_fraction_mean": agg["predicted_positive_fraction_mean"],
            "top_score_margin_mean": agg["top_score_margin_mean"], "max_psi_mean": agg["max_psi_mean"],
            "latency_p50_ms": agg["latency_ms"]["p50"], "latency_p90_ms": agg["latency_ms"]["p90"],
            "latency_p99_ms": agg["latency_ms"]["p99"], "repeated_request_count": agg["repeated_request_count"],
            "repeated_chunk_count": agg["repeated_chunk_count"],
            "drift_severity_hist": json.dumps(agg["drift_severity_hist"]),
            "chunk_count_hist": json.dumps(agg["chunk_count_hist"]),
            "caller_distribution": json.dumps(agg["caller_distribution"]),
        }
        for k, v in flat.items():
            w.writerow([k, v])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--dir", default=str(_SYNAPSE_DIR))
    args = ap.parse_args()
    recs = read_day(args.date, Path(args.dir))
    agg = aggregate(recs)
    sim = load_sim_ranges()
    write_report(args.date, agg, sim, _ART / f"sn126_synapse_report_{args.date}.md")
    write_csv(args.date, agg, _ART / f"sn126_synapse_summary_{args.date}.csv")
    print(f"records={len(recs)} | wrote artifacts/sn126_synapse_report_{args.date}.md "
          f"and sn126_synapse_summary_{args.date}.csv")


if __name__ == "__main__":
    main()
