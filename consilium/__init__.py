"""Consilium - Multi-model deliberation for important decisions."""

__version__ = "0.11.0"

from .models import (
    COUNCIL,
    JUDGE_MODEL,
    DISCUSS_MODELS,
    REDTEAM_MODELS,
    OXFORD_MODELS,
    SessionResult,
    detect_social_context,
)

from .prompts import ROLE_LIBRARY

from .council import (
    run_council,
    run_blind_phase_parallel,
)

from .quick import run_quick
from .discuss import run_discuss
from .redteam import run_redteam
from .solo import run_solo
from .oxford import run_oxford

__all__ = [
    "run_council",
    "run_quick",
    "run_discuss",
    "run_redteam",
    "run_solo",
    "run_oxford",
    "run_blind_phase_parallel",
    "detect_social_context",
    "SessionResult",
    "COUNCIL",
    "JUDGE_MODEL",
    "DISCUSS_MODELS",
    "REDTEAM_MODELS",
    "OXFORD_MODELS",
    "ROLE_LIBRARY",
]
