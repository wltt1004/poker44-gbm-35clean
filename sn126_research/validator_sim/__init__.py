"""Validator-faithful request & competition simulators for SN126 Poker44.

These simulators import — never reimplement — the official reward() and the
frozen production Poker44Predictor. They use sanitized benchmark data through the
exact production feature path and preserve release-date walk-forward validation
(no future release ever leaks into past-release selection).
"""

__all__ = ["scenario_config", "request_simulator", "competition_simulator", "report"]
