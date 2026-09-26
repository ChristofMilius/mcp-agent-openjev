"""Typed probabilistic decision client backed by OpenJev over Bionic (LM Studio).

The readout is OpenJev's: one score per option letter at the first output position of a single-token completion, then a
temperature-scaled softmax. The transport is Bionic's `/v1/responses` endpoint -- the one LM Studio surface that returns
logprobs (`/v1/completions` and `/v1/chat/completions` return `logprobs: null` and cap `top_logprobs` at 20).

The calibration constants are the released helper's: ``temp 0.85, noul_t 1.829074, noul_bias 0.0, perms 1``.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Any

from mcp_agent_openjev.bionic import BionicClient, BionicError
from mcp_agent_openjev.config import Config
from mcp_agent_openjev.openjev import (
    LETTERS,
    OpenJevReadoutError,
    build_prompt,
    choice_confidence,
    letter_scores,
    score_confidence,
    softmax,
)
from mcp_agent_openjev.schemas import ChoiceDecision, NoulDecision, ScoreDecision

Criteria = str | dict[str, str]


class DecisionError(Exception):
    """Base error for decision failures, carrying a user-facing message."""


class TooManyOptionsError(DecisionError):
    """More than 52 candidates cannot be expressed as single letter tokens."""


class InvalidTierError(DecisionError, ValueError):
    """A score tier is malformed: missing label, duplicate label, or foreign weight."""


def _tier_weight(raw: Any, index: int) -> float:
    """Validate an explicit tier weight against the tier's ordinal position."""
    try:
        weight = float(raw)
    except (TypeError, ValueError) as exc:
        raise InvalidTierError(
            f"tier score must be a number, got {raw!r} at position {index}"
        ) from exc
    if weight != float(index):
        raise InvalidTierError(
            f"tier score {weight} does not match the tier's position {index}. The weight of "
            "a tier is its ordinal position, so 'expected_score' stays an expected tier "
            "index. Omit 'score' to use the position, or reorder the tiers."
        )
    return weight


def _normalize_options(candidates: type[Enum] | list[str]) -> list[str]:
    if isinstance(candidates, type) and issubclass(candidates, Enum):
        return [e.value for e in candidates]
    return [str(o) for o in candidates]


def _effective_threshold(n_options: int, configured: float | str) -> float:
    """Auto threshold scales inversely with candidate count (1.25 / N).

    TODO: this was fitted against gemma logprobs and is wrong for OpenJev's properly
    calibrated output. Re-fit on real tiebreak cases (see Phase 4).
    """
    if configured == "auto":
        return 1.25 / max(n_options, 1)
    return float(configured)


class DecisionClient:
    """Calibrated typed decision client over Bionic's Open Responses endpoint."""

    def __init__(self, config: Config):
        if config.temperature <= 0:
            raise ValueError("Temperature must be positive.")
        self.config = config
        self.client = BionicClient(config)

    # -- public surface ----------------------------------------------------

    def decide_choice(
        self,
        state: dict[str, Any],
        candidates: type[Enum] | list[str],
        criteria: Criteria = "",
        allow_abstain: bool = True,
    ) -> ChoiceDecision:
        """Multi-class categorical decision with calibrated probabilities."""
        options = _normalize_options(candidates)
        if len(options) > len(LETTERS):
            raise TooManyOptionsError(
                f"{len(options)} candidates exceed the {len(LETTERS)} single-letter token limit; "
                "split the decision into a hierarchy."
            )

        probs, raw, _ = self._readout(state, _criteria_text(criteria), options)
        return self._to_choice(probs, options, allow_abstain, raw)

    def decide_noul(self, state: dict[str, Any], assertion: str) -> NoulDecision:
        """Binary truth judgment: is the assertion TRUE or FALSE for the state?"""
        decision = self.decide_choice(
            state=state,
            candidates=["TRUE", "FALSE"],
            criteria=f"Evaluate whether the following assertion is strictly TRUE or FALSE: {assertion}",
            allow_abstain=False,
        )
        p_true = decision.probabilities.get("TRUE", 0.5)
        value = p_true >= 0.5
        conf = p_true if value else (1.0 - p_true)
        return NoulDecision(
            value=value,
            probability_true=p_true,
            confidence=conf,
            abstained=decision.abstained,
        )

    def decide_score(
        self,
        state: dict[str, Any],
        tiers: Sequence[str | dict[str, Any]],
        criteria: Criteria = "",
        allow_abstain: bool = True,
    ) -> ScoreDecision:
        """Ordinal evaluation: probability mass across tiers + expected score.

        A tier is either a plain label string, or a dict carrying `label` (or
        `value`) plus an optional `score`. The weight of a tier is its ordinal
        position, so an explicit `score` must equal that position: a custom
        weight would make `expected_score` a weighted expectation over an
        arbitrary scale rather than the expected tier index, which is what
        `decide_score` promises and what its callers compare against.

        The resulting `tier_weights` echoes the scale, so `expected_score` never
        has to be read against an assumed convention.
        """
        labels: list[str] = []
        scores: dict[str, float] = {}
        for index, tier in enumerate(tiers):
            if isinstance(tier, dict):
                label = tier.get("label") or tier.get("value")
                if label is None:
                    raise InvalidTierError(
                        f"tier at position {index} is a dict without 'label' or 'value': "
                        f"{tier!r}. Accepted shapes: a label string, or a dict with "
                        "'label' (or 'value') and an optional 'score' equal to the "
                        "tier's position. Note that '{\"$text\": ...}' is not a tier."
                    )
                label = str(label)
                weight = _tier_weight(tier.get("score", index), index)
            else:
                label, weight = str(tier), float(index)
            if label in scores:
                raise InvalidTierError(
                    f"duplicate tier label {label!r} at position {index}. Tier labels must "
                    "be unique: they key the probability mass and the expected score."
                )
            labels.append(label)
            scores[label] = weight

        decision = self.decide_choice(
            state=state, candidates=labels, criteria=criteria, allow_abstain=allow_abstain
        )
        weight = dict(scores)
        if "UNKNOWN" in decision.probabilities and "UNKNOWN" not in weight:
            weight["UNKNOWN"] = 0.0
        expected = sum(p * weight.get(label, 0.0) for label, p in decision.probabilities.items())
        return ScoreDecision(
            expected_score=expected,
            level_probabilities=decision.probabilities,
            tier_weights=weight,
            confidence=score_confidence(list(decision.probabilities.values())),
            abstained=decision.abstained,
        )

    # -- readout -----------------------------------------------------------

    def _readout(
        self, state: dict[str, Any], instructions: str, options: list[str]
    ) -> tuple[list[float], dict[str, float], int]:
        """Temperature-scaled probabilities aligned with `options`, plus the raw letter logits.

        Averages over `config.perms` letterings (option orders) when perms > 1.
        """
        if self.config.perms <= 1:
            raw, tokens = self._readout_once(state, instructions, options)
            probs = softmax([v / self.config.temperature for v in raw])
            return probs, dict(zip(options, raw, strict=True)), tokens

        import random

        orders = []
        for j in range(self.config.perms):
            order = list(range(len(options)))
            random.Random(j).shuffle(order)
            orders.append(order)
        acc = [0.0] * len(options)
        tokens = 0
        for order in orders:
            shuffled = [options[i] for i in order]
            raw, t = self._readout_once(state, instructions, shuffled)
            tokens += t
            p = softmax([v / self.config.temperature for v in raw])
            for pos, i in enumerate(order):
                acc[i] += p[pos] / self.config.perms
        return acc, {}, tokens

    def _readout_once(
        self, state: dict[str, Any], instructions: str, options: list[str]
    ) -> tuple[list[float], int]:
        """One lettering: raw letter log-probabilities aligned with `options`, plus prompt tokens."""
        letters = [LETTERS[i] for i in range(len(options))]
        content = build_prompt(state, instructions, options)
        try:
            top, tokens = self.client.top_logprobs(content)
        except BionicError as exc:
            raise DecisionError(str(exc)) from exc
        raw = letter_scores(top, letters)
        vals: list[float] = []
        for letter, v in zip(letters, raw, strict=True):
            if v is None:
                raise OpenJevReadoutError(f"letter {letter} absent from the model readout")
            vals.append(v)
        return vals, tokens

    # -- decision assembly --------------------------------------------------

    def _to_choice(
        self, probs: list[float], options: list[str], allow_abstain: bool, raw: dict[str, float]
    ) -> ChoiceDecision:
        best = max(range(len(probs)), key=lambda i: probs[i])
        confidence = choice_confidence(probs)
        effective_threshold = _effective_threshold(len(options), self.config.threshold)
        abstained = allow_abstain and confidence < effective_threshold
        value = "UNKNOWN" if abstained else options[best]
        tentative = options[best] if abstained else None
        return ChoiceDecision(
            value=value,
            probabilities={o: p for o, p in zip(options, probs, strict=True)},
            confidence=round(confidence, 4),
            abstained=abstained,
            tentative_value=tentative,
            raw_logits={k: round(v, 6) for k, v in raw.items()},
        )


def _criteria_text(criteria: Criteria) -> str:
    if isinstance(criteria, dict):
        return "\n".join(f"- {k}: {v}" for k, v in criteria.items())
    return str(criteria).strip()


__all__ = [
    "DecisionClient",
    "DecisionError",
    "InvalidTierError",
    "OpenJevReadoutError",
    "TooManyOptionsError",
]
