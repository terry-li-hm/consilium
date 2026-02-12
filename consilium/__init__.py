"""Consilium - Multi-model deliberation for important decisions."""

__version__ = "0.3.0"

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
