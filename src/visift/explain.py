"""
How a recommendation's 0-100 score is built, as an exact additive breakdown.

Every chart scorer combines its components the same way:

    suitability (Q) = 0.30 semantic fit + 0.20 readability
                      + 0.20 data quality + 0.30 sample support
    pattern factor  = 0.35 + 0.65 x pattern strength (S) / 100
    score           = Q x pattern factor

Read from 100 downwards, that is a stack of non-negative segments:

    100 = sum_i w_i (100 - c_i)        suitability deductions (sum to 100 - Q)
        + Q x 0.65 x (1 - S / 100)     pattern shortfall
        + score                        what remains

Components are stored rounded to two decimals, so the pattern shortfall is
taken as Q - score (it differs from the formula by at most ~0.01). That keeps
the segments summing to exactly 100 and ending exactly at the shown score.
"""

from collections.abc import Mapping
from typing import Any, Literal, TypedDict

SUITABILITY_WEIGHTS: tuple[tuple[str, str, float], ...] = (
    ("semantic_fit", "Semantic fit", 0.30),
    ("readability", "Readability", 0.20),
    ("data_quality", "Data quality", 0.20),
    ("sample_support", "Sample support", 0.30),
)

PATTERN_FLOOR = 0.35
PATTERN_WEIGHT = 0.65

SegmentKind = Literal["deduction", "pattern", "score"]


class Segment(TypedDict):
    key: str
    label: str
    value: float
    kind: SegmentKind
    detail: str


class ScoreBreakdown(TypedDict):
    score: float
    suitability: float
    pattern_strength: float
    pattern_factor: float
    segments: list[Segment]
    formula: str


def pattern_factor(pattern_strength: float) -> float:
    """0.35 for no pattern at all, 1.0 for the strongest pattern."""

    return PATTERN_FLOOR + PATTERN_WEIGHT * pattern_strength / 100


def score_breakdown(candidate: Mapping[str, Any]) -> ScoreBreakdown:
    components = candidate["components"]
    score = float(candidate["score"])
    signal = float(components["signal"])

    segments: list[Segment] = []
    suitability = 100.0

    for key, label, weight in SUITABILITY_WEIGHTS:
        value = float(components[key])
        deduction = max(0.0, weight * (100 - value))
        suitability -= deduction
        segments.append(
            {
                "key": key,
                "label": label,
                "value": deduction,
                "kind": "deduction",
                "detail": f"{label} {value:.1f}/100 (weight {weight:.2f}) "
                f"costs {deduction:.1f} points",
            }
        )

    shortfall = max(0.0, suitability - score)
    factor = pattern_factor(signal)

    segments.append(
        {
            "key": "pattern",
            "label": "Pattern shortfall",
            "value": shortfall,
            "kind": "pattern",
            "detail": f"Pattern strength {signal:.1f}/100 gives a pattern factor of "
            f"{factor:.2f}, which costs {shortfall:.1f} points",
        }
    )

    segments.append(
        {
            "key": "score",
            "label": "Score",
            "value": score,
            "kind": "score",
            "detail": f"Final score {score:.1f}",
        }
    )

    stored_suitability = float(components["visualization_quality"])

    return {
        "score": score,
        "suitability": stored_suitability,
        "pattern_strength": signal,
        "pattern_factor": factor,
        "segments": segments,
        "formula": (
            f"{score:.1f} = {stored_suitability:.1f} chart suitability × "
            f"{factor:.2f} pattern factor (0.35 + 0.65 × {signal:.1f}/100)"
        ),
    }
