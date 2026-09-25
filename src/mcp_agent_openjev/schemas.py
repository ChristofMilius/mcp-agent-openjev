"""Decision schemas.

Two gaps in OpenJev's PyPI 0.1.0 schemas are closed by local subclasses, so
pydantic does not silently drop the fields the abstention and score contracts
depend on:

* ``ChoiceDecision`` omits ``tentative_value``, which its documented
  abstention contract requires (present in the upstream repo's main branch).
* ``ScoreDecision`` reports ``expected_score`` without saying which scale it
  was computed on, leaving the reader to infer the tier weights.
"""

from __future__ import annotations

from openjevpro.schemas import ChoiceDecision as _OpenJevChoiceDecision
from openjevpro.schemas import NoulDecision
from openjevpro.schemas import ScoreDecision as _OpenJevScoreDecision
from pydantic import Field

__all__ = ["ChoiceDecision", "NoulDecision", "ScoreDecision"]


class ChoiceDecision(_OpenJevChoiceDecision):
    """Categorical decision; value normalizes to UNKNOWN when abstained."""

    tentative_value: str | None = None


class ScoreDecision(_OpenJevScoreDecision):
    """Ordinal decision carrying the scale its expected score was computed on."""

    tier_weights: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Label to weight map used to compute expected_score, in tier order. Includes "
            "UNKNOWN (0.0) when abstention was enabled, so the expected score can be "
            "recomputed from level_probabilities."
        ),
    )
