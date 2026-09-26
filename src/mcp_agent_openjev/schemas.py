"""Decision schemas.

Standalone pydantic models for the three decision types. Two fields the abstention
and score contracts depend on are declared explicitly so the shape never depends on
a third-party package:

* ``ChoiceDecision.tentative_value`` -- the raw argmax kept aside when the decision
  abstains to UNKNOWN.
* ``ScoreDecision.tier_weights`` -- the scale ``expected_score`` was computed on.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

__all__ = ["ChoiceDecision", "NoulDecision", "ScoreDecision"]


class ChoiceDecision(BaseModel):
    """Categorical decision; value normalizes to UNKNOWN when abstained."""

    value: str = Field(description="Chosen option, or UNKNOWN when abstained")
    probabilities: dict[str, float] = Field(default_factory=dict, description="Option -> calibrated probability")
    confidence: float = Field(description="Calibrated probability of the winning option")
    abstained: bool = Field(description="True when the winner fell below the abstention threshold")
    tentative_value: str | None = Field(
        default=None, description="Raw argmax kept aside when abstained to UNKNOWN, else None"
    )
    raw_logits: dict[str, float] = Field(default_factory=dict, description="Option -> raw log-probability")


class NoulDecision(BaseModel):
    """Binary truth judgment."""

    value: bool = Field(description="TRUE or FALSE for the assertion")
    probability_true: float = Field(description="Calibrated P(TRUE)")
    confidence: float = Field(description="max(P(true), 1 - P(true))")
    abstained: bool = Field(description="True when the judgment fell below the abstention threshold")


class ScoreDecision(BaseModel):
    """Ordinal decision carrying the scale its expected score was computed on."""

    expected_score: float = Field(description="Probability-weighted expected tier index")
    level_probabilities: dict[str, float] = Field(default_factory=dict, description="Tier label -> probability")
    tier_weights: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Label to weight map used to compute expected_score, in tier order. Includes "
            "UNKNOWN (0.0) when abstention was enabled, so the expected score can be "
            "recomputed from level_probabilities."
        ),
    )
    confidence: float = Field(description="Spread of the distribution away from its mode")
    abstained: bool = Field(description="True when the decision fell below the abstention threshold")
