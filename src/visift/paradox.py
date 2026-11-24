"""
Simpson's paradox detection: relationships that reverse (or only appear)
once you look within groups.

Method and thresholds (pre-registered and committed before running on any
real-world dataset; designed with the well-known Palmer Penguins example
in mind: bill length vs bill depth is negative overall but positive within
every species):

For each numeric pair (x, y) that is not flagged redundant, and each
eligible grouping column g:

- Eligible groupings: categorical, ordinal or boolean columns with 2-8
  levels, using only levels with at least MIN_ROWS_PER_GROUP rows (and at
  least 2 such levels). Groupings that are near-duplicates of, or near-
  deterministically linked to, x or y are skipped.
- Cap: per pair, at most MAX_GROUPINGS_PER_PAIR groupings, chosen by
  min(eta^2(g -> x), eta^2(g -> y)). A reversal needs the groups to shift
  both x and y, so groupings that shift neither cannot produce one.
- overall: Spearman rho of x and y on the rows used.
- r_w (within-group association): ranks of x and y are computed within
  each group and centered by the group mean rank; r_w is the Pearson
  correlation of those centered ranks (a pooled within-group Spearman).
- per-group: Spearman rho within each group.
- Significance: overall uses the usual t test (df = n - 2); r_w uses
  df = n - k - 1 for k groups. Bonferroni over all (pair, grouping) checks
  run on the dataset, alpha = 0.05.

REVERSAL when all hold:
  sign(overall) != sign(r_w); |overall| >= 0.15 and |r_w| >= 0.15; both
  significant after correction; groups covering >= 2/3 of the rows have a
  per-group rho with the same sign as r_w.

MASKED (kept separate) when: |overall| < 0.10, |r_w| >= 0.30, r_w
significant after correction, and the same 2/3 consistency rule.

Findings are not put on the calibrated 0-100 scale and do not change any
existing score or rank.
"""

import itertools
import math
from typing import Literal, TypedDict

import numpy as np
import pandas as pd
from scipy import stats

from visift.redundancy import Redundancy, flagged_pairs
from visift.types import DatasetProfile

MIN_ROWS_PER_GROUP = 20
MIN_GROUP_LEVELS = 2
MAX_GROUP_LEVELS = 8
MAX_GROUPINGS_PER_PAIR = 3

REVERSAL_MIN_ABS = 0.15
MASKED_MAX_OVERALL = 0.10
MASKED_MIN_WITHIN = 0.30
CONSISTENT_ROW_SHARE = 2 / 3
ALPHA = 0.05

NUMERIC_TYPES = {"numeric_continuous", "numeric_discrete"}
GROUPING_TYPES = {"categorical", "ordinal", "boolean"}

ParadoxKind = Literal["reversal", "masked"]


class GroupStat(TypedDict):
    group: str
    n: int
    rho: float


class Paradox(TypedDict):
    kind: ParadoxKind
    x: str
    y: str
    group: str
    n: int
    overall: float
    within: float
    overall_p: float
    within_p: float
    groups: list[GroupStat]
    gap: float
    sentence: str


def _p_value(r: float, df: int) -> float:
    if df <= 0 or not np.isfinite(r):
        return 1.0
    if abs(r) >= 1:
        return 0.0
    t = r * math.sqrt(df / (1 - r * r))
    return float(2 * stats.t.sf(abs(t), df))


def _eta_squared(groups: pd.Series, values: pd.Series) -> float:
    total = float(((values - values.mean()) ** 2).sum())
    if total == 0:
        return 0.0
    means = values.groupby(groups).transform("mean")
    return float(((means - values.mean()) ** 2).sum()) / total


def _within_rank_correlation(frame: pd.DataFrame, x: str, y: str, g: str) -> float:
    grouped = frame.groupby(g, observed=True)
    rx = grouped[x].rank()
    ry = grouped[y].rank()
    rx = rx - rx.groupby(frame[g], observed=True).transform("mean")
    ry = ry - ry.groupby(frame[g], observed=True).transform("mean")
    denominator = math.sqrt(float((rx * rx).sum()) * float((ry * ry).sum()))
    return float((rx * ry).sum()) / denominator if denominator > 0 else 0.0


def _eligible_groupings(
    df: pd.DataFrame, profile: DatasetProfile
) -> dict[str, list[object]]:
    """Grouping column -> the levels with enough rows to use."""

    types = {c["name"]: c["semantic_type"] for c in profile["column_profiles"]}
    eligible: dict[str, list[object]] = {}

    for column in df.columns:
        name = str(column)
        if types.get(name) not in GROUPING_TYPES:
            continue
        counts = df[name].value_counts(dropna=True)
        if not MIN_GROUP_LEVELS <= len(counts) <= MAX_GROUP_LEVELS:
            continue
        levels = [level for level, n in counts.items() if n >= MIN_ROWS_PER_GROUP]
        if len(levels) >= MIN_GROUP_LEVELS:
            eligible[name] = levels

    return eligible


def _sentence(p: "Paradox") -> str:
    def direction(r: float) -> str:
        return "go together" if r > 0 else "go in opposite directions"

    within = f"{p['within']:+.2f}"
    overall = f"{p['overall']:+.2f}"

    if p["kind"] == "reversal":
        return (
            f"Across all rows, {p['x']} and {p['y']} {direction(p['overall'])} "
            f"(ρ = {overall}), but within each {p['group']} they "
            f"{direction(p['within'])} (within-{p['group']} ρ = {within})."
        )

    return (
        f"Across all rows, {p['x']} and {p['y']} look unrelated (ρ = {overall}), "
        f"but within each {p['group']} they {direction(p['within'])} "
        f"(within-{p['group']} ρ = {within})."
    )


def find_paradoxes(
    df: pd.DataFrame,
    profile: DatasetProfile,
    redundancies: list[Redundancy] | None = None,
) -> list[Paradox]:
    """Reversals and masked relationships, largest gap first."""

    types = {c["name"]: c["semantic_type"] for c in profile["column_profiles"]}
    numeric = [str(c) for c in df.columns if types.get(str(c)) in NUMERIC_TYPES]
    groupings = _eligible_groupings(df, profile)

    if len(numeric) < 2 or not groupings:
        return []

    redundant = flagged_pairs(redundancies or [])

    # Choose groupings per pair, then count the tests for Bonferroni.
    plan: list[tuple[str, str, str]] = []

    for x, y in itertools.combinations(numeric, 2):
        if frozenset((x, y)) in redundant:
            continue

        ranked = []
        for g, levels in groupings.items():
            if frozenset((g, x)) in redundant or frozenset((g, y)) in redundant:
                continue
            frame = df[[x, y, g]].dropna()
            frame = frame[frame[g].isin(levels)]
            if len(frame) < MIN_ROWS_PER_GROUP * MIN_GROUP_LEVELS:
                continue
            strength = min(
                _eta_squared(frame[g], frame[x].astype(float)),
                _eta_squared(frame[g], frame[y].astype(float)),
            )
            ranked.append((strength, g))

        ranked.sort(key=lambda item: -item[0])
        plan.extend((x, y, g) for _, g in ranked[:MAX_GROUPINGS_PER_PAIR])

    if not plan:
        return []

    threshold = ALPHA / len(plan)
    found: list[Paradox] = []

    for x, y, g in plan:
        frame = df[[x, y, g]].dropna()
        frame = frame[frame[g].isin(groupings[g])]
        n = len(frame)
        k = frame[g].nunique()

        if frame[x].nunique() < 3 or frame[y].nunique() < 3:
            continue

        overall = float(stats.spearmanr(frame[x], frame[y])[0])
        within = _within_rank_correlation(frame, x, y, g)

        if not (np.isfinite(overall) and np.isfinite(within)):
            continue

        overall_p = _p_value(overall, n - 2)
        within_p = _p_value(within, n - k - 1)

        group_stats: list[GroupStat] = []
        for level, part in frame.groupby(g, observed=True):
            rho = (
                float(stats.spearmanr(part[x], part[y])[0])
                if part[x].nunique() > 1 and part[y].nunique() > 1
                else float("nan")
            )
            group_stats.append({"group": str(level), "n": len(part), "rho": rho})

        agreeing = sum(
            s["n"]
            for s in group_stats
            if np.isfinite(s["rho"]) and np.sign(s["rho"]) == np.sign(within)
        )
        consistent = agreeing >= CONSISTENT_ROW_SHARE * n

        kind: ParadoxKind | None = None

        if (
            np.sign(overall) != np.sign(within)
            and abs(overall) >= REVERSAL_MIN_ABS
            and abs(within) >= REVERSAL_MIN_ABS
            and overall_p < threshold
            and within_p < threshold
            and consistent
        ):
            kind = "reversal"

        elif (
            abs(overall) < MASKED_MAX_OVERALL
            and abs(within) >= MASKED_MIN_WITHIN
            and within_p < threshold
            and consistent
        ):
            kind = "masked"

        if kind is None:
            continue

        paradox: Paradox = {
            "kind": kind,
            "x": x,
            "y": y,
            "group": g,
            "n": n,
            "overall": overall,
            "within": within,
            "overall_p": overall_p,
            "within_p": within_p,
            "groups": group_stats,
            "gap": abs(within - overall),
            "sentence": "",
        }
        paradox["sentence"] = _sentence(paradox)
        found.append(paradox)

    found.sort(key=lambda p: -p["gap"])
    return found
