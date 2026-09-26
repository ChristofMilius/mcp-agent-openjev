"""The OpenJev readout: prompt layout, letter scoring, calibration.

The prompt layout and the calibration math are the ones of OpenJev's released helper (helper/shim.py): one score per option
letter at the first output position, then a temperature-scaled softmax. With the same profile constants this produces the
helper's answers.

Calibration constants (profiles/openjev.json): ``temp 0.85, noul_t 1.829074, noul_bias 0.0, perms 1``.
"""

from __future__ import annotations

import json
import math
from typing import Any

LETTERS = [chr(65 + i) for i in range(26)] + [chr(97 + i) for i in range(26)]  # the 52 single-letter option labels


def build_prompt(state: Any, instructions: str, options: list[str]) -> str:
    """The exact prompt for one lettering: the option list, then the readout instruction."""
    state_text = json.dumps(state, ensure_ascii=False) if isinstance(state, dict) else str(state)
    lines = "\n".join(f"[{LETTERS[i]}] {opt}" for i, opt in enumerate(options))
    return (
        f"State:\n{state_text}\n\n"
        f"Question: {instructions}\n"
        f"Options:\n{lines}\n\n"
        "Answer with the letter of the best option only."
    )


def letter_scores(top: dict[str, float], letters: list[str]) -> list[float | None]:
    """Log-probabilities of `letters` at the answer position, matched by label text (bare form preferred over a
    whitespace-padded or lowercased look-alike). None marks a letter the model never scored."""
    out: list[float | None] = []
    for letter in letters:
        value = None
        for form in (letter, f" {letter}", letter.lower(), f" {letter.lower()}"):
            if form in top:
                value = top[form]
                break
        out.append(value)
    return out


def softmax(z: list[float]) -> list[float]:
    m = max(z)
    e = [math.exp(v - m) for v in z]
    s = sum(e)
    return [v / s for v in e]


def choice_confidence(p: list[float]) -> float:
    """TypeSafe's official formula (system-one-adapter-python confidence_metrics.py)."""
    if len(p) == 1:
        return 1.0
    u = 1.0 / len(p)
    return max(0.0, (max(p) - u) / (1.0 - u))


def score_confidence(p: list[float]) -> float:
    if len(p) == 1:
        return 1.0
    mode = max(range(len(p)), key=p.__getitem__)
    dist = sum(pi * abs(i - mode) for i, pi in enumerate(p))
    c = (len(p) - 1) / 2
    umad = sum(abs(i - c) for i in range(len(p))) / len(p)
    return max(0.0, 1.0 - dist / umad)


def noul_value(p_true: float, noul_t: float, noul_bias: float) -> float:
    """Map P(TRUE) through the noul sigmoid to a 0..1 truth score."""
    py = min(max(p_true, 1e-4), 1 - 1e-4)
    z = math.log(py / (1 - py)) / noul_t + noul_bias
    return 1 / (1 + math.exp(-z))


class OpenJevReadoutError(RuntimeError):
    """The model did not return a score for every candidate letter."""


__all__ = [
    "LETTERS",
    "OpenJevReadoutError",
    "build_prompt",
    "choice_confidence",
    "letter_scores",
    "noul_value",
    "score_confidence",
    "softmax",
]
