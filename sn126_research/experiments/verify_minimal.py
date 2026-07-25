"""Confirm the size/performance tradeoff of the greedy tail (redundancy pruning)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sn126_research.benchmark_client import BenchmarkClient
from sn126_research.evaluation import evaluate
from sn126_research.features import feature_vector
from sn126_research.features_plus import CANDIDATE_NAMES as P1, candidate_vector as p1v
from sn126_research.features_plus2 import CANDIDATE_NAMES as P2, candidate_vector as p2v
from sn126_research.sanitizer import to_live_group
from sklearn.ensemble import HistGradientBoostingClassifier

def gbm():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=300, l2_regularization=1.0, random_state=0)

client = BenchmarkClient()
by_date = client.download_recent(n_dates=6, max_records_per_date=200)
records = [r for d in sorted(by_date) for r in by_date[d]]
Xb, C, y, dates = [], [], [], []
for rec in records:
    for gi, g in enumerate(rec.groups):
        if gi >= len(rec.ground_truth):
            continue
        live = to_live_group(g)
        if not live:
            continue
        Xb.append(feature_vector(live, scope="all"))
        C.append(list(p1v(live)) + list(p2v(live)))
        y.append(int(rec.ground_truth[gi])); dates.append(rec.source_date)
Xb = np.asarray(Xb, float); C = np.asarray(C, float); y = np.asarray(y, int); dates = np.asarray(dates)
names = [f"p1:{n}" for n in P1] + [f"p2:{n}" for n in P2]
idx = {n: i for i, n in enumerate(names)}

def lodo(cols):
    aps, recs, rews = [], [], []
    X = Xb if not cols else np.hstack([Xb, C[:, cols]])
    for d in sorted(set(dates.tolist())):
        tr, va = dates != d, dates == d
        gb = gbm().fit(X[tr], y[tr]); p = gb.predict_proba(X[va])[:, 1]
        m = evaluate(y[va], p); aps.append(m["ap"]); recs.append(m["recall_at_5pct_fpr"]); rews.append(m["reward"])
    ap, rec, rew = np.mean(aps), np.mean(recs), np.mean(rews)
    return ap, rec, rew, 0.35*ap + 0.30*rec + 0.35

seeds = [idx["p2:betsize_street_std_mean"], idx["p2:action_bigram_entropy"]]
clean5 = seeds + [idx["p1:pot_escalation_mean"], idx["p2:mean_aggressors_per_hand"], idx["p1:potodds_call_fold_gap"]]
drop_bigpot = clean5 + [idx["p1:cbet_freq"]]
full = seeds + [idx["p1:pot_escalation_mean"], idx["p2:mean_aggressors_per_hand"], idx["p1:big_pot_rate"], idx["p1:potodds_call_fold_gap"], idx["p1:cbet_freq"]]

sets = [("baseline (30)", []), ("+2 seeds (32)", seeds), ("+clean5 (35)", clean5),
        ("+6 drop big_pot (36)", drop_bigpot), ("+full greedy (37)", full)]
print(f"{'set':26s} {'nfeat':>5s} {'AP':>7s} {'recall':>7s} {'rawRew':>7s} {'calRew':>7s}")
for nm, cols in sets:
    ap, rec, rew, cal = lodo(cols)
    print(f"{nm:26s} {30+len(cols):5d} {ap:7.4f} {rec:7.4f} {rew:7.4f} {cal:7.4f}")
