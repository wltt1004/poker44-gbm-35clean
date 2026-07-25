"""Validator-faithful request simulator.

Builds a per-release pool of (raw GBM score, label) using the EXACT production
path (chunk_feature_vector on sanitized/deduped benchmark hands + the frozen
Poker44Predictor model), then bootstraps DetectionSynapse-shaped requests with
controlled chunk counts and bot fractions. Every metric comes from the OFFICIAL
reward(); calibration comes from the OFFICIAL batch_percentile_calibrate().

Release-date walk-forward is preserved by the caller (competition/p0 selection
only ever look at strictly-past releases). Bootstrap sampling (with replacement)
within a single release is used to reach >=1000 requests; it never mixes releases.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from poker44.miner_model.calibration import batch_percentile_calibrate
from poker44.miner_model.features import chunk_feature_vector
from poker44.miner_model.predictor import Poker44Predictor
from poker44.score.scoring import reward as official_reward
from sn126_research.benchmark_client import BenchmarkClient
from sn126_research.sanitizer import to_live_group

_CACHE = Path(__file__).resolve().parent / ".cache_pools"


@dataclass
class ReleasePool:
    date: str
    raw: np.ndarray          # per-chunk raw GBM P(bot)
    label: np.ndarray        # 1=bot, 0=human
    model_version: str

    @property
    def bot_idx(self) -> np.ndarray:
        return np.flatnonzero(self.label == 1)

    @property
    def human_idx(self) -> np.ndarray:
        return np.flatnonzero(self.label == 0)


def _pool_cache_path(date: str, model_version: str) -> Path:
    key = hashlib.sha256(f"{date}|{model_version}".encode()).hexdigest()[:16]
    return _CACHE / f"pool_{date}_{key}.npz"


def build_release_pools(
    predictor: Poker44Predictor,
    *,
    n_dates: int,
    max_records: int,
    client: Optional[BenchmarkClient] = None,
) -> Dict[str, ReleasePool]:
    """Featurize + score every chunk of the most recent n_dates releases (cached)."""
    _CACHE.mkdir(parents=True, exist_ok=True)
    client = client or BenchmarkClient()
    by_date = client.download_recent(n_dates=n_dates, max_records_per_date=max_records)
    pools: Dict[str, ReleasePool] = {}
    for date in sorted(by_date):
        cache = _pool_cache_path(date, predictor.model_version)
        if cache.exists():
            data = np.load(cache)
            pools[date] = ReleasePool(date, data["raw"], data["label"], predictor.model_version)
            continue
        X, y = [], []
        for rec in by_date[date]:
            for gi, group in enumerate(rec.groups):
                if gi >= len(rec.ground_truth):
                    continue
                live = to_live_group(group)          # sanitize + dedupe (benchmark path)
                if not live:
                    continue
                X.append(chunk_feature_vector(live))  # SAME 35-feature path as production
                y.append(int(rec.ground_truth[gi]))
        if not X:
            continue
        Xarr = np.asarray(X, float)
        raw = predictor.model.predict_proba(Xarr)[:, 1]   # frozen model, raw P(bot)
        label = np.asarray(y, int)
        np.savez(cache, raw=raw, label=label)
        pools[date] = ReleasePool(date, raw, label, predictor.model_version)
    return pools


def sample_request(pool: ReleasePool, chunk_count: int, n_bots: int,
                   rng: np.random.RandomState) -> np.ndarray:
    """Bootstrap indices for ONE request: exactly n_bots bots + (chunk_count-n_bots)
    humans, sampled with replacement from THIS release only (never mixing releases)."""
    n_humans = chunk_count - n_bots
    bots, humans = pool.bot_idx, pool.human_idx
    if bots.size == 0 or humans.size == 0:
        raise ValueError(f"release {pool.date} lacks a class (bots={bots.size} humans={humans.size})")
    bi = rng.choice(bots, size=n_bots, replace=True)
    hi = rng.choice(humans, size=n_humans, replace=True)
    return np.concatenate([bi, hi])


def simulate_requests(
    pool: ReleasePool,
    predictor: Poker44Predictor,
    *,
    chunk_count: int,
    n_bots: int,
    p0: float,
    n_requests: int,
    rng: np.random.RandomState,
) -> Dict[str, np.ndarray]:
    """Bootstrap n_requests requests; return per-request metric arrays (official reward())."""
    ap = np.empty(n_requests); rec = np.empty(n_requests)
    hfpr = np.empty(n_requests); hrec = np.empty(n_requests)
    tsq = np.empty(n_requests); rew = np.empty(n_requests)
    for i in range(n_requests):
        idx = sample_request(pool, chunk_count, n_bots, rng)
        raw_batch = pool.raw[idx]
        labels = pool.label[idx]
        calib = batch_percentile_calibrate(
            raw_batch, p0=p0,
            min_calib_batch=predictor.min_calib_batch,
            global_anchor_raw=predictor.global_anchor_raw,
        )
        r, res = official_reward(calib, labels)     # OFFICIAL scoring
        ap[i] = res["ap_score"]; rec[i] = res["bot_recall"]
        hfpr[i] = res["hard_fpr"]; hrec[i] = res["hard_bot_recall"]
        tsq[i] = res["threshold_sanity_quality"]; rew[i] = r
    return {"ap": ap, "recall_at_5fpr": rec, "hard_fpr_at_0.5": hfpr,
            "hard_bot_recall_at_0.5": hrec, "tsq": tsq, "reward": rew}


def dist_stats(arr: np.ndarray, *, worst: str = "min") -> Dict[str, float]:
    a = np.asarray(arr, float)
    return {
        "mean": float(np.mean(a)), "median": float(np.median(a)), "std": float(np.std(a)),
        "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90)),
        "worst": float(np.min(a) if worst == "min" else np.max(a)),
    }


def p_tsq_zero(tsq: np.ndarray) -> float:
    return float(np.mean(np.asarray(tsq) <= 0.0))
