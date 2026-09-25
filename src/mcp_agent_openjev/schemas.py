"""Decision schemas.

OpenJev's PyPI 0.1.0 `ChoiceDecision` omits `tentative_value` (present in the
repo's main branch and in its documented abstention contract). Rather than let
pydantic silently drop the field we set, these subclasses restore it locally
while keeping OpenJev's types as the base.
"""

from __future__ import annotations

from openjevpro.schemas import ChoiceDecision as _OpenJevChoiceDecision
from openjevpro.schemas import NoulDecision, ScoreDecision

__all__ = ["ChoiceDecision", "NoulDecision", "ScoreDecision"]


class ChoiceDecision(_OpenJevChoiceDecision):
    """Categorical decision; value normalizes to UNKNOWN when abstained."""

    tentative_value: str | None = None
