"""Reference Poker44 miner with simple chunk-level behavioral heuristics."""

# from __future__ import annotations

import time
from collections import Counter
from pathlib import Path
from typing import Tuple

import bittensor as bt

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse


class Miner(BaseMinerNeuron):
    """
    Reference heuristic miner.

    It aggregates simple behavior signals over each chunk and returns a bot-risk
    score per chunk. The goal is not SOTA accuracy, but a deterministic and
    explainable baseline that is meaningfully better than random.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        bt.logging.info("🤖 Heuristic Poker44 Miner started")
        repo_root = Path(__file__).resolve().parents[1]
        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            # Every file imported/read by the SERVED runtime path (prediction-critical +
            # runtime-critical + runtime-loaded observability). model.joblib is attested
            # separately via artifact_sha256, so it is intentionally NOT listed here.
            # Offline tools (monitor.py, synapse_report.py, build_feature_profile.py,
            # benchmark/dataset/validator_sim/experiments) are excluded — forward() never
            # imports them.
            implementation_files=[
                Path(__file__).resolve(),                                    # neurons/miner.py
                repo_root / "poker44" / "miner_model" / "predictor.py",
                repo_root / "poker44" / "miner_model" / "features.py",
                repo_root / "poker44" / "miner_model" / "calibration.py",
                repo_root / "poker44" / "miner_model" / "model_meta.json",
                repo_root / "poker44" / "miner_model" / "telemetry.py",
                repo_root / "poker44" / "miner_model" / "synapse_analyzer.py",
                repo_root / "poker44" / "miner_model" / "synapse_sink.py",
                repo_root / "poker44" / "miner_model" / "training_feature_profile.json",
                repo_root / "poker44" / "miner_model" / "feature_core" / "__init__.py",
                repo_root / "poker44" / "miner_model" / "feature_core" / "sanitizer_core.py",
                repo_root / "poker44" / "miner_model" / "feature_core" / "base30.py",
                repo_root / "poker44" / "miner_model" / "feature_core" / "plus.py",
                repo_root / "poker44" / "miner_model" / "feature_core" / "plus2.py",
                # feature_core.sanitizer_core imports the bet-bucket ladder from the
                # official sanitizer, so its content determines served feature values.
                repo_root / "poker44" / "validator" / "payload_view.py",
            ],
            defaults={
                "model_name": "poker44-gbm-35clean",
                "model_version": "gbm-35clean-v1",
                "framework": "scikit-learn-histgradientboosting",
                "license": "MIT",
                # Set POKER44_MODEL_REPO_URL to your published model repo (a non-reference
                # repo is required by the compliance policy for a custom model_name).
                "repo_url": "https://github.com/Poker44/Poker44-subnet",
                "notes": (
                    "GBM-35clean inference served via poker44.miner_model; batch percentile "
                    "calibration (p0=0.85). Set POKER44_MODEL_REPO_URL and "
                    "POKER44_MODEL_REPO_COMMIT to your published model repo and deploy commit."
                ),
                "open_source": True,
                "inference_mode": "remote",
                # sha256(model.joblib); regenerate if the model artifact is ever re-exported.
                "artifact_sha256": "b8b7fc78586568e4480ed83a880eb2639001605215a1a6631ed6206c89a9194d",
                "training_data_statement": (
                    "Supervised HistGradientBoosting model trained on the public Poker44 training "
                    "benchmark (https://api.poker44.net/api/v1/benchmark), projected through "
                    "prepare_hand_for_miner into the live miner-visible view. 35 behavioral "
                    "features (feature_version sn126-35clean-v1) with batch percentile calibration "
                    "(p0=0.85). No validator-only live evaluation data is used in training."
                ),
                "training_data_sources": ["poker44-public-training-benchmark"],
                "private_data_attestation": (
                    "This model trains only on the public Poker44 benchmark labels and does not "
                    "train on validator-only live evaluation data."
                ),
            },
        )
        # feature_version is not a native build_local_model_manifest field; attach it
        # explicitly so the manifest records the exact served feature contract.
        self.model_manifest["feature_version"] = "sn126-35clean-v1"
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup(repo_root)

        # Load the trained GBM inference layer once. If unavailable, the miner
        # transparently falls back to the reference heuristic (score_chunk).
        self.predictor = None
        try:
            from poker44.miner_model.predictor import Poker44Predictor
            self.predictor = Poker44Predictor()
            bt.logging.info(f"Loaded GBM predictor (model_version={self.predictor.model_version}).")
        except Exception as exc:  # pragma: no cover - deployment/artifact issues
            bt.logging.warning(
                f"GBM predictor unavailable ({exc}); falling back to heuristic scoring."
            )

        # Low-overhead, failure-isolated telemetry (metadata + aggregate stats only).
        self.telemetry = None
        try:
            from poker44.miner_model.telemetry import MinerTelemetry
            self.telemetry = MinerTelemetry()
        except Exception as exc:  # pragma: no cover
            bt.logging.warning(f"Telemetry unavailable ({exc}); continuing without it.")

        # Read-only Synapse Intelligence layer (analyzer + non-blocking sink).
        self.synapse_analyzer = None
        self.synapse_sink = None
        try:
            from poker44.miner_model.synapse_analyzer import SynapseAnalyzer
            from poker44.miner_model.synapse_sink import AnalyzerSink
            self.synapse_analyzer = SynapseAnalyzer()
            self.synapse_sink = AnalyzerSink()
        except Exception as exc:  # pragma: no cover
            bt.logging.warning(f"Synapse analyzer unavailable ({exc}); continuing without it.")

        # # Attach handlers after initialization
        # self.axon.attach(
        #     forward_fn = self.forward,
        #     blacklist_fn = self.blacklist,
        #     priority_fn = self.priority,
        # )
        # bt.logging.info("Attaching forward function to miner axon.")
        
        bt.logging.info(f"Axon created: {self.axon}")

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"open_source={self.model_manifest.get('open_source')}"
        )
        bt.logging.info(
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}"
        )
        bt.logging.info(
            "Miner prep docs available | "
            f"miner_doc={repo_root / 'docs' / 'miner.md'}"
        )

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """Assign one deterministic bot-risk score per chunk."""
        chunks = synapse.chunks or []
        diagnostics = None
        if self.predictor is not None:
            diagnostics = self.predictor.predict_detailed(chunks, fallback=self.score_chunk)
            scores = list(diagnostics.final_scores)
        else:
            scores = [self.score_chunk(chunk) for chunk in chunks]
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        self._emit_intelligence(synapse, chunks, scores, diagnostics)
        bt.logging.info(f"Miner Predctions: {synapse.predictions}")
        bt.logging.info(f"Scored {len(chunks)} chunks.")
        return synapse

    def _emit_intelligence(self, synapse, chunks, scores, diagnostics) -> None:
        """Request-scoped telemetry + synapse analyzer. Never affects inference."""
        try:
            caller = getattr(getattr(synapse, "dendrite", None), "hotkey", None)
            # privacy-safe one-way fingerprints of the ALREADY-sanitized chunks.
            req_fp, chunk_fps = None, None
            try:
                import hashlib as _hl
                from poker44.validator.integrity import chunk_fingerprint
                chunk_fps = [chunk_fingerprint(c) for c in chunks]
                req_fp = _hl.sha256("|".join(chunk_fps).encode("utf-8")).hexdigest()
            except Exception:
                req_fp, chunk_fps = None, None
            hand_counts = [len(c) for c in chunks]
            raw_scores = diagnostics.raw_scores if diagnostics is not None else None
            fallback_used = bool(diagnostics.fallback_used) if diagnostics is not None else True
            exc_type = diagnostics.exception_type if diagnostics is not None else None
            duration_s = (diagnostics.total_duration_ms / 1000.0) if diagnostics is not None else None

            # (a) compact telemetry (feeds the ops monitor); request-scoped, no shared attrs
            telemetry = getattr(self, "telemetry", None)
            if telemetry is not None:
                telemetry.log_inference(
                    chunk_count=len(chunks), hand_counts=hand_counts,
                    raw_scores=raw_scores, calibrated_scores=scores, duration_s=duration_s,
                    fallback_used=fallback_used, caller_hotkey=caller,
                    model_version=self.model_manifest.get("model_version"),
                    feature_version=self.model_manifest.get("feature_version"),
                    manifest_digest=getattr(self, "manifest_digest", None),
                    exception_type=exc_type, request_fingerprint=req_fp, chunk_fingerprints=chunk_fps,
                )
            # (b) rich synapse analyzer -> non-blocking bounded sink
            analyzer = getattr(self, "synapse_analyzer", None)
            sink = getattr(self, "synapse_sink", None)
            if analyzer is not None and sink is not None and diagnostics is not None:
                record = analyzer.analyze(
                    diagnostics, hand_counts=hand_counts, caller_hotkey=caller,
                    manifest_digest=getattr(self, "manifest_digest", None),
                    model_version=self.model_manifest.get("model_version"),
                    feature_version=self.model_manifest.get("feature_version"),
                    request_fingerprint=req_fp, chunk_fingerprints=chunk_fps,
                )
                sink.submit(record)
        except Exception:  # pragma: no cover - intelligence must never break inference
            return

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @classmethod
    def _score_hand(cls, hand: dict) -> float:
        actions = hand.get("actions") or []
        players = hand.get("players") or []
        streets = hand.get("streets") or []
        outcome = hand.get("outcome") or {}

        action_counts = Counter(action.get("action_type") for action in actions)
        meaningful_actions = max(
            1,
            sum(
                action_counts.get(kind, 0)
                for kind in ("call", "check", "bet", "raise", "fold")
            ),
        )

        call_ratio = action_counts.get("call", 0) / meaningful_actions
        check_ratio = action_counts.get("check", 0) / meaningful_actions
        fold_ratio = action_counts.get("fold", 0) / meaningful_actions
        raise_ratio = action_counts.get("raise", 0) / meaningful_actions
        street_depth = len(streets) / 3.0
        showdown_flag = 1.0 if outcome.get("showdown") else 0.0

        player_count_signal = 0.0
        if players:
            player_count_signal = (6 - min(len(players), 6)) / 4.0

        score = 0.0
        score += 0.32 * street_depth
        score += 0.22 * showdown_flag
        score += 0.18 * cls._clamp01(call_ratio / 0.35)
        score += 0.12 * cls._clamp01(check_ratio / 0.30)
        score += 0.08 * cls._clamp01(player_count_signal)
        score -= 0.18 * cls._clamp01(fold_ratio / 0.55)
        score -= 0.10 * cls._clamp01(raise_ratio / 0.20)

        return cls._clamp01(score)

    @classmethod
    def score_chunk(cls, chunk: list[dict]) -> float:
        if not chunk:
            return 0.5

        hand_scores = [cls._score_hand(hand) for hand in chunk]
        avg_score = sum(hand_scores) / len(hand_scores)

        return round(cls._clamp01(avg_score), 6)

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        """Determine whether to blacklist incoming requests."""
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        """Assign priority based on caller's stake."""
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Random miner running...")
        while True:
            bt.logging.info(f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}")
            time.sleep(5 * 60)
