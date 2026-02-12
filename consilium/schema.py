from pydantic import BaseModel
from typing import Literal

class ActionItem(BaseModel):
    action: str
    priority: Literal["high", "medium", "low"] = "medium"

class Dissent(BaseModel):
    model: str
    concern: str

class CouncilMeta(BaseModel):
    timestamp: str
    models_used: list[str]
    rounds: int
    duration_seconds: float
    estimated_cost_usd: float

class CouncilOutput(BaseModel):
    schema_version: str = "1.0"
    question: str
    decision: str
    confidence: Literal["low", "medium", "high"]
    reasoning_summary: str
    dissents: list[Dissent]
    action_items: list[ActionItem]
    meta: CouncilMeta