"""Frontier Council - Multi-model deliberation for important decisions."""

__version__ = "0.1.2"

from .council import (
    run_council,
    run_blind_phase_parallel,
    detect_social_context,
    COUNCIL,
    JUDGE_MODEL,
)

__all__ = [
    "run_council",
    "run_blind_phase_parallel",
    "detect_social_context",
    "COUNCIL",
    "JUDGE_MODEL",
]
