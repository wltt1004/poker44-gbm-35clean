"""Orchestrate the request + competition simulations and emit artifacts.

Outputs:
    artifacts/sn126_request_scenarios.csv
    artifacts/sn126_competition_simulations.csv
    artifacts/sn126_simulation_report.md

Run: python -m sn126_research.validator_sim.report
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

from poker44.miner_model.predictor import Poker44Predictor
from sn126_research.validator_sim import scenario_config as C
from sn126_research.validator_sim.competition_simulator import simulate_competitions
from sn126_research.validator_sim.request_simulator import (
    build_release_pools, dist_stats, p_tsq_zero, simulate_requests,
)

ART = Path(__file__).resolve().parents[2] / "artifacts"
_WORST = {"hard_fpr_at_0.5": "max"}  # higher FPR is worse; everything else min


def _metric_rows(release, cc, bf, nb, n_req, p0, metrics):
    ptsq = p_tsq_zero(metrics["tsq"])
    for name, arr in metrics.items():
        s = dist_stats(arr, worst=_WORST.get(name, "min"))
        yield {
            "release": release, "chunk_count": cc,
            "bot_fraction": ("one_bot" if bf == "one_bot" else f"{float(bf):.2f}"),
            "n_bots": nb, "n_requests": n_req, "p0": p0, "metric": name,
            "mean": round(s["mean"], 5), "median": round(s["median"], 5),
            "std": round(s["std"], 5), "p10": round(s["p10"], 5),
            "p90": round(s["p90"], 5), "worst": round(s["worst"], 5),
            "p_tsq_zero": round(ptsq, 5),
        }


def run_scenarios(pools, predictor, dates):
    """PART 2: per (held-out release, scenario) at PRODUCTION p0."""
    rows = []
    for release in dates:
        pool = pools[release]
        if pool.bot_idx.size == 0 or pool.human_idx.size == 0:
            continue
        for cc, bf in C.valid_scenarios():
            nb = C.n_bots_for(cc, bf)
            rng = np.random.RandomState(C.SEED + hash((release, cc, str(bf))) % 10_000)
            m = simulate_requests(pool, predictor, chunk_count=cc, n_bots=nb,
                                  p0=C.PRODUCTION_P0, n_requests=C.N_REQUESTS, rng=rng)
            rows.extend(_metric_rows(release, cc, bf, nb, C.N_REQUESTS, C.PRODUCTION_P0, m))
    return rows


def run_p0_walkforward(pools, predictor, dates):
    """Select p0 on strictly-PAST releases; evaluate on the next unseen release."""
    cc, bf = C.REPRESENTATIVE_CHUNK_COUNT, C.REPRESENTATIVE_BOT_FRACTION
    nb = C.n_bots_for(cc, bf)
    # per (release, p0) mean reward at the representative scenario
    mean_reward = {}
    for release in dates:
        pool = pools[release]
        for p0 in C.P0_CANDIDATES:
            rng = np.random.RandomState(C.SEED + hash((release, p0)) % 10_000)
            m = simulate_requests(pool, predictor, chunk_count=cc, n_bots=nb, p0=p0,
                                  n_requests=C.N_REQUESTS_P0, rng=rng)
            mean_reward[(release, p0)] = float(np.mean(m["reward"]))
    rows = walkforward_select(dates, mean_reward, C.P0_CANDIDATES, C.PRODUCTION_P0)
    return rows, mean_reward


def walkforward_select(dates, mean_reward, candidates, production_p0):
    """Pure walk-forward p0 selection: for each held-out release, select p0 using
    ONLY strictly-past releases (dates[:i]); never touches the held-out or future.
    """
    rows = []
    for i in range(1, len(dates)):
        held = dates[i]
        past = dates[:i]                          # strictly past — no future leak
        past_mean = {p0: float(np.mean([mean_reward[(d, p0)] for d in past])) for p0 in candidates}
        selected = max(past_mean, key=past_mean.get)
        rows.append({
            "held_out_release": held,
            "selected_p0_from_past": selected,
            "reward_selected_on_heldout": round(mean_reward[(held, selected)], 5),
            "reward_p0_0.85_on_heldout": round(mean_reward[(held, production_p0)], 5),
            "delta_vs_production": round(mean_reward[(held, selected)] - mean_reward[(held, production_p0)], 5),
        })
    return rows


def run_competition(pools, predictor, dates):
    """PART 3: last 5 releases as 5 rounds; composite under assumed mean."""
    rounds = dates[-5:]
    cc, bf = C.REPRESENTATIVE_CHUNK_COUNT, C.REPRESENTATIVE_BOT_FRACTION
    nb = C.n_bots_for(cc, bf)
    round_arrays = []
    for r, release in enumerate(rounds):
        rng = np.random.RandomState(C.SEED + 1000 + r)
        m = simulate_requests(pools[release], predictor, chunk_count=cc, n_bots=nb,
                              p0=C.PRODUCTION_P0, n_requests=C.N_REQUESTS, rng=rng)
        round_arrays.append(m["reward"])
    res = simulate_competitions(round_arrays, n_competitions=C.N_COMPETITIONS,
                                thresholds=C.COMPOSITE_THRESHOLDS, seed=C.SEED)
    res["rounds"] = rounds
    res["scenario"] = f"cc{cc}_bf{bf}"
    return res


def _write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ART.mkdir(parents=True, exist_ok=True)
    predictor = Poker44Predictor()
    n_dates = int(os.getenv("SN126_N_DATES", str(C.N_DATES)))
    pools = build_release_pools(predictor, n_dates=n_dates, max_records=C.MAX_RECORDS_PER_DATE)
    dates = sorted(pools)
    print(f"releases={dates} | pool sizes={[int(pools[d].raw.size) for d in dates]}")

    scen = run_scenarios(pools, predictor, dates)
    _write_csv(ART / "sn126_request_scenarios.csv", scen,
               ["release", "chunk_count", "bot_fraction", "n_bots", "n_requests", "p0",
                "metric", "mean", "median", "std", "p10", "p90", "worst", "p_tsq_zero"])
    print(f"wrote {len(scen)} scenario rows")

    p0_rows, _ = run_p0_walkforward(pools, predictor, dates)
    comp = run_competition(pools, predictor, dates)

    comp_rows = [{"kind": "round", "index": i, "release": rel,
                  "value": round(comp["round_mean_reward"][i], 5), "field": "round_mean_reward"}
                 for i, rel in enumerate(comp["rounds"])]
    cs = comp["composite_stats"]
    for k, v in cs.items():
        comp_rows.append({"kind": "composite", "index": "", "release": "", "value": round(v, 5), "field": f"composite_{k}"})
    comp_rows.append({"kind": "composite", "index": "", "release": "", "value": round(comp["prob_any_zero_round"], 5), "field": "prob_any_zero_round"})
    for t, p in comp["prob_exceed"].items():
        comp_rows.append({"kind": "composite", "index": "", "release": "", "value": round(p, 5), "field": f"prob_composite_gt_{t:.2f}"})
    _write_csv(ART / "sn126_competition_simulations.csv", comp_rows,
               ["kind", "index", "release", "field", "value"])
    print("wrote competition CSV")

    _write_report(ART / "sn126_simulation_report.md", dates, pools, scen, p0_rows, comp)
    print("wrote report md")


def _reward_row(scen, release, cc, bf):
    bfl = "one_bot" if bf == "one_bot" else f"{float(bf):.2f}"
    for r in scen:
        if r["release"] == release and r["chunk_count"] == cc and r["bot_fraction"] == bfl and r["metric"] == "reward":
            return r
    return None


def _write_report(path, dates, pools, scen, p0_rows, comp):
    held = dates[-1]
    lines = []
    L = lines.append
    L("# SN126 Poker44 — Simulation Report\n")
    L("Model **gbm-35clean-v1** / features **sn126-35clean-v1** / production calibration "
      "**batch percentile anchor p0=0.85**. Metrics from the OFFICIAL `poker44.score.scoring.reward()`; "
      "calibration from the OFFICIAL `batch_percentile_calibrate`. Bootstrap requests within a single "
      "release (never mixing releases); production p0 unchanged.\n")
    L(f"Releases (chronological): {dates}\n")
    L(f"Pool sizes (chunks/release): {[int(pools[d].raw.size) for d in dates]}\n")

    L("\n## PART 2 — Request scenarios (held-out release = latest, production p0=0.85)\n")
    L("Per-request **reward** distribution by scenario on the latest release "
      f"({held}), 1000 bootstrap requests each:\n")
    L("| chunk_count | bot_fraction | n_bots | reward mean | median | std | p10 | p90 | worst | P(TSQ=0) |")
    L("|---|---|---|---|---|---|---|---|---|---|")
    for cc, bf in C.valid_scenarios():
        r = _reward_row(scen, held, cc, bf)
        if r:
            L(f"| {cc} | {r['bot_fraction']} | {r['n_bots']} | {r['mean']} | {r['median']} | "
              f"{r['std']} | {r['p10']} | {r['p90']} | {r['worst']} | {r['p_tsq_zero']} |")

    L("\n## p0 walk-forward (select on past releases, evaluate on next unseen)\n")
    L("Representative scenario "
      f"cc{C.REPRESENTATIVE_CHUNK_COUNT}/bf{C.REPRESENTATIVE_BOT_FRACTION}. "
      "**Production p0=0.85 is NOT changed** — evidence only.\n")
    L("| held-out release | p0 selected from past | reward@selected | reward@0.85 | delta |")
    L("|---|---|---|---|---|")
    for row in p0_rows:
        L(f"| {row['held_out_release']} | {row['selected_p0_from_past']} | "
          f"{row['reward_selected_on_heldout']} | {row['reward_p0_0.85_on_heldout']} | {row['delta_vs_production']} |")

    L("\n## PART 3 — Five-round competition (assumed aggregation = arithmetic MEAN)\n")
    L("> The true backend aggregation is private. Arithmetic mean is an EXPLICIT ASSUMPTION, not confirmed.\n")
    L(f"Rounds (releases): {comp['rounds']} | scenario: {comp['scenario']} | p0=0.85\n")
    L(f"Per-round mean reward: {[round(x,4) for x in comp['round_mean_reward']]}\n")
    cs = comp["composite_stats"]
    L(f"\nComposite: mean **{cs['mean']:.4f}**, median {cs['median']:.4f}, "
      f"p10 {cs['p10']:.4f}, p90 {cs['p90']:.4f}, min {cs['min']:.4f}\n")
    L(f"Worst-round mean: {comp['worst_round_stats']['mean']:.4f} "
      f"(p10 {comp['worst_round_stats']['p10']:.4f})\n")
    L(f"P(at least one zero round): {comp['prob_any_zero_round']:.4f}\n")
    L("\n| composite > | probability |")
    L("|---|---|")
    for t, p in comp["prob_exceed"].items():
        L(f"| {t:.2f} | {p:.4f} |")

    L("\n## Realistic reward range (not one optimistic number)\n")
    rr = _reward_row(scen, held, C.REPRESENTATIVE_CHUNK_COUNT, C.REPRESENTATIVE_BOT_FRACTION)
    if rr:
        L(f"At the representative live shape (cc{C.REPRESENTATIVE_CHUNK_COUNT}/bf{C.REPRESENTATIVE_BOT_FRACTION}) "
          f"on the latest release, per-cycle reward spans roughly **p10 {rr['p10']} → p90 {rr['p90']}** "
          f"(median {rr['median']}, worst {rr['worst']}). The composite over 5 rounds centers near "
          f"**{cs['mean']:.3f}** (p10 {cs['p10']:.3f}).\n")
    path.write_text("\n".join(lines), "utf-8")


if __name__ == "__main__":
    main()
