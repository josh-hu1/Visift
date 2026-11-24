"""
Target-aware ranking: "What do you want to explain?"

When the user names an outcome column (the target), Visift answers two
questions instead of one:

1. Drivers of <target>: every two-variable recommendation that involves the
   target, ranked by its existing calibrated score.
2. Everything else, ranked exactly as without a target.

There is deliberately no boost constant. Scores stay comparable with the
no-target mode, and the target only changes which list a chart appears in.
"""

import re
from collections.abc import Sequence

from visift.types import ColumnProfile, DatasetProfile, Recommendation

# Column names that usually denote an outcome, strongest evidence first.
# Exact names are matched against the whole normalized column name; the
# word list is matched against its underscore-separated tokens.
EXACT_OUTCOME_NAMES = ("target", "label", "outcome", "y")

OUTCOME_WORDS = (
    "churn",
    "churned",
    "survived",
    "survival",
    "converted",
    "conversion",
    "subscribed",
    "purchased",
    "clicked",
    "responded",
    "default",
)

CATEGORICAL_TARGET_TYPES = {"categorical", "ordinal"}

MAX_CATEGORICAL_TARGET_LEVELS = 10


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _is_target_shaped(column: ColumnProfile) -> bool:
    semantic_type = column["semantic_type"]

    if semantic_type == "boolean":
        return True

    return (
        semantic_type in CATEGORICAL_TARGET_TYPES
        and 2 <= column["unique_count"] <= MAX_CATEGORICAL_TARGET_LEVELS
    )


def _outcome_name_rank(name: str) -> int | None:
    """Lower is stronger evidence; None means the name says nothing."""

    normalized = _normalize(name)

    if normalized in EXACT_OUTCOME_NAMES:
        return EXACT_OUTCOME_NAMES.index(normalized)

    tokens = set(normalized.split("_"))

    for index, word in enumerate(OUTCOME_WORDS):
        if word in tokens:
            return len(EXACT_OUTCOME_NAMES) + index

    return None


def suggest_target(profile: DatasetProfile) -> tuple[str | None, str]:
    """
    Suggest a likely outcome column, with a human-readable reason.

    Rules, in order:

    1. A boolean or low-cardinality categorical column whose name reads like
       an outcome (target, label, outcome, y, churn, survived, converted,
       subscribed, default, ...). Stronger names win; ties go to the later
       column, since outcomes are conventionally last.
    2. Otherwise, the last column if it is boolean.

    Returns (None, reason) when nothing qualifies. Numeric targets are
    never guessed: a numeric column is only a target if the user says so.
    """

    columns = profile["column_profiles"]

    named = []

    for position, column in enumerate(columns):
        rank = _outcome_name_rank(column["name"])

        if rank is not None and _is_target_shaped(column):
            named.append((rank, -position, column))

    if named:
        _, _, best = min(named, key=lambda item: (item[0], item[1]))
        kind = "binary" if best["semantic_type"] == "boolean" else "categorical"

        return (best["name"], f"{kind} column named like an outcome")

    if columns and columns[-1]["semantic_type"] == "boolean":
        return (columns[-1]["name"], "last column, and it is binary")

    return (None, "no binary or categorical column named like an outcome")


def involves_target(recommendation: Recommendation, target: str) -> bool:
    """A two-variable chart with the target on either axis."""

    y = recommendation.get("y")

    return y is not None and target in (recommendation["x"], y)


def split_by_target(
    recommendations: Sequence[Recommendation], target: str
) -> tuple[list[Recommendation], list[Recommendation]]:
    """
    Split ranked recommendations into (drivers of target, everything else).

    Both lists keep the input order, i.e. the existing score ranking.
    """

    drivers = [r for r in recommendations if involves_target(r, target)]
    rest = [r for r in recommendations if not involves_target(r, target)]

    return (drivers, rest)


def rank_with_target(
    recommendations: Sequence[Recommendation], target: str
) -> list[Recommendation]:
    """Drivers of the target first, then everything else."""

    drivers, rest = split_by_target(recommendations, target)

    return drivers + rest
