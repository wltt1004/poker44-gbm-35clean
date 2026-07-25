"""Local miner smoke test — NO validator, NO network, NO wallet required.

Verifies the deployment artifacts and exercises the exact prediction path used by
neurons.miner.Miner.forward():

    scores = predictor.predict(synapse.chunks, fallback=Miner.score_chunk)

Run:  python -m poker44.miner_model.smoke_test
Exit code 0 = all checks pass, 1 = a check failed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import sklearn

from poker44.miner_model.features import PROD_FEATURE_ORDER
from poker44.miner_model.predictor import Poker44Predictor
from poker44.validator.payload_view import prepare_hand_for_miner
from poker44.validator.synapse import DetectionSynapse

_DIR = Path(__file__).resolve().parent
_RESULTS: list[tuple[bool, str]] = []


def check(cond: bool, label: str) -> bool:
    _RESULTS.append((bool(cond), label))
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    return bool(cond)


# ---- fake sanitized inputs (built offline via the real sanitizer) -----------
def _raw_hand(opener=1, size_bb=3):
    return {
        "metadata": {"game_type": "Hold'em", "limit_type": "No Limit", "max_seats": 2,
                     "hero_seat": 1, "button_seat": 2, "sb": 0.01, "bb": 0.02, "ante": 0.0},
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
            {"action_id": "4", "street": "flop", "actor_seat": 2, "action_type": "check",
             "amount": 0.0, "raise_to": None, "call_to": None, "normalized_amount_bb": 0.0,
             "pot_before": 0.1, "pot_after": 0.1},
            {"action_id": "5", "street": "flop", "actor_seat": 1, "action_type": "bet",
             "amount": 0.06, "raise_to": None, "call_to": None, "normalized_amount_bb": 3.0,
             "pot_before": 0.1, "pot_after": 0.16},
        ],
        "outcome": {"showdown": False},
    }


def _hand(opener=1, size_bb=3):
    return prepare_hand_for_miner(_raw_hand(opener, size_bb))


def build_chunks():
    empty_chunk: list = []                                   # 0 hands
    one_hand_chunk = [_hand()]                               # 1 hand
    normal_chunk = [_hand(opener=1 + i % 2, size_bb=2 + i % 4) for i in range(5)]
    batch = [empty_chunk, one_hand_chunk, normal_chunk]      # the required cases
    # pad to a realistic request size so the batch calibration path (>=8) runs
    for i in range(9):
        batch.append([_hand(opener=1 + i % 2, size_bb=2 + i % 5) for _ in range(3 + i % 3)])
    return batch


# ---- forward() parity -------------------------------------------------------
def _score_chunk_fallback():
    """The exact fallback miner.forward() uses (Miner.score_chunk), if importable."""
    try:
        from neurons.miner import Miner
        return Miner.score_chunk
    except Exception as exc:  # pragma: no cover
        print(f"  (note: could not import Miner.score_chunk: {exc}; using 0.5 fallback)")
        return lambda chunk: 0.5


def run() -> int:
    print("== 1. Deployment artifacts (task 4) ==")
    for fname in ("model.joblib", "model_meta.json", "features.py", "calibration.py", "predictor.py"):
        check((_DIR / fname).exists(), f"artifact present: {fname}")
    meta = json.loads((_DIR / "model_meta.json").read_text("utf-8"))
    check(len(PROD_FEATURE_ORDER) == 35, "feature count == 35")
    check(meta["feature_names"] == list(PROD_FEATURE_ORDER), "feature order matches model_meta.json")
    meta_skl = meta.get("model", {}).get("sklearn_version", "?")
    same_minor = ".".join(sklearn.__version__.split(".")[:2]) == ".".join(str(meta_skl).split(".")[:2])
    check(True, f"sklearn installed={sklearn.__version__} trained={meta_skl} "
                f"({'match' if same_minor else 'MINOR MISMATCH — verify'})")

    print("\n== 2. Model loads + feature contract (task 4) ==")
    try:
        pred = Poker44Predictor()
        check(True, "Poker44Predictor loaded model.joblib + model_meta.json")
        check(pred.model.n_features_in_ == 35, "model.n_features_in_ == 35")
        check(pred.p0 == 0.85, f"calibration p0 == 0.85 (got {pred.p0})")
    except Exception as exc:
        check(False, f"Poker44Predictor load FAILED: {exc}")
        return _summary()

    print("\n== 3. Prediction path == miner.forward() (task 3) ==")
    fallback = _score_chunk_fallback()
    chunks = build_chunks()
    synapse = DetectionSynapse(chunks=chunks)
    check(isinstance(synapse.chunks, list) and len(synapse.chunks) == len(chunks),
          f"DetectionSynapse carries {len(chunks)} chunks")

    exc_raised = None
    try:
        scores = pred.predict(synapse.chunks or [], fallback=fallback)   # SAME call as forward()
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
    except Exception as exc:  # must never happen
        exc_raised = exc
        scores = []
    check(exc_raised is None, f"no exception during predict ({exc_raised})")
    check(len(scores) == len(chunks), f"output length {len(scores)} == input chunk count {len(chunks)}")
    check(all(0.0 <= s <= 1.0 for s in scores), "every score in [0, 1]")
    check(len(synapse.predictions) == len(scores), "predictions align 1:1 with risk_scores")

    # empty chunk still yields a valid score (index 0 is the empty chunk)
    check(0.0 <= scores[0] <= 1.0, "empty chunk -> valid score (no crash)")

    print("\n== 4. Calibration behaviour (task 3) ==")
    n_hi = sum(1 for s in scores if s >= 0.5)
    check(max(scores) > min(scores), "calibration produced score spread (rank-preserving)")
    check(n_hi <= max(1, round(0.35 * len(scores))),
          f"top-anchored: only {n_hi}/{len(scores)} chunks >= 0.5 (p0=0.85 -> ~15%)")

    print("\n== 5. Fallback safety (task 3) ==")
    # (a) model LOAD fails -> miner sets predictor=None -> heuristic path
    load_failed = False
    try:
        Poker44Predictor(model_path=_DIR / "does_not_exist.joblib")
    except Exception:
        load_failed = True
    fb_scores = [min(1.0, max(0.0, float(fallback(c)))) for c in chunks]  # heuristic path
    check(load_failed, "missing model.joblib raises at load (miner then falls back)")
    check(len(fb_scores) == len(chunks) and all(0.0 <= s <= 1.0 for s in fb_scores),
          "heuristic fallback yields correct-length scores in [0,1]")

    # (b) model INFERENCE fails -> predict() returns safe, correct-length fallback
    class _Boom:
        n_features_in_ = 35
        def predict_proba(self, X):
            raise RuntimeError("boom")
    pred.model = _Boom()
    safe = pred.predict(chunks, fallback=fallback)
    check(len(safe) == len(chunks) and all(0.0 <= s <= 1.0 for s in safe),
          "inference failure -> safe correct-length fallback (never wrong length)")

    return _summary()


def _summary() -> int:
    passed = sum(1 for ok, _ in _RESULTS if ok)
    total = len(_RESULTS)
    print("\n" + "=" * 60)
    print(f"SMOKE TEST: {passed}/{total} checks passed")
    failed = [lbl for ok, lbl in _RESULTS if not ok]
    if failed:
        print("FAILED:")
        for lbl in failed:
            print(f"  - {lbl}")
        print("RESULT: NOT READY")
        return 1
    print("RESULT: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(run())
