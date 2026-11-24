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


def _eta_squared(codes: np.ndarray, values: np.ndarray, k: int) -> float:
    """Share of the variance in values explained by the group means."""

    centered = values - values.mean()
    total = float(centered @ centered)
    if total == 0:
        return 0.0
    counts = np.bincount(codes, minlength=k)
    sums = np.bincount(codes, weights=centered, minlength=k)
    present = counts > 0
    return float((sums[present] ** 2 / counts[present]).sum()) / total


def _centered_group_ranks(codes: np.ndarray, values: np.ndarray, k: int) -> np.ndarray:
    """Average ranks within each group, minus the group's mean rank."""

    ranks = np.empty(len(values), dtype=float)
    for group in range(k):
        members = codes == group
        if members.any():
            group_ranks = stats.rankdata(values[members])
            ranks[members] = group_ranks - group_ranks.mean()
    return ranks


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denominator = math.sqrt(float(a @ a) * float(b @ b))
    return float(a @ b) / denominator if denominator > 0 else float("nan")


class _Grouping:
    """A grouping column prepared once: integer codes for its usable levels."""

    def __init__(self, series: pd.Series, levels: list[object]) -> None:
        # groupby order (sorted levels), as reported in per-group statistics
        self.levels = list(pd.Index(levels).sort_values())
        lookup = {level: code for code, level in enumerate(self.levels)}
        self.codes = np.fromiter(
            (lookup.get(value, -1) for value in series.to_numpy()),
            dtype=np.int64,
            count=len(series),
        )
        self.k = len(self.levels)


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

    values = {c: df[c].to_numpy(dtype=float) for c in numeric}
    present = {c: ~np.isnan(values[c]) for c in numeric}
    prepared = {g: _Grouping(df[g], levels) for g, levels in groupings.items()}
    usable = {g: grouping.codes >= 0 for g, grouping in prepared.items()}

    def rows(x: str, y: str, g: str) -> np.ndarray:
        return present[x] & present[y] & usable[g]

    # Choose groupings per pair, then count the tests for Bonferroni.
    plan: list[tuple[str, str, str]] = []

    for x, y in itertools.combinations(numeric, 2):
        if frozenset((x, y)) in redundant:
            continue

        ranked = []
        for g, grouping in prepared.items():
            if frozenset((g, x)) in redundant or frozenset((g, y)) in redundant:
                continue
            mask = rows(x, y, g)
            if int(mask.sum()) < MIN_ROWS_PER_GROUP * MIN_GROUP_LEVELS:
                continue
            codes = grouping.codes[mask]
            strength = min(
                _eta_squared(codes, values[x][mask], grouping.k),
                _eta_squared(codes, values[y][mask], grouping.k),
            )
            ranked.append((strength, g))

        ranked.sort(key=lambda item: -item[0])
        plan.extend((x, y, g) for _, g in ranked[:MAX_GROUPINGS_PER_PAIR])

    if not plan:
        return []

    threshold = ALPHA / len(plan)
    found: list[Paradox] = []

    # Rank caches, valid when a column has no missing values on the rows a
    # grouping uses (the common case); otherwise ranks are computed per pair.
    overall_ranks: dict[tuple[str, str], np.ndarray] = {}
    within_ranks: dict[tuple[str, str], np.ndarray] = {}

    def ranks_for(
        column: str, g: str, mask: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        grouping = prepared[g]
        complete = bool(np.array_equal(mask, usable[g] & present[column])) and bool(
            np.array_equal(mask, usable[g])
        )
        key = (column, g)
        if complete and key in overall_ranks:
            return overall_ranks[key], within_ranks[key]
        subset = values[column][mask]
        overall = stats.rankdata(subset)
        within = _centered_group_ranks(grouping.codes[mask], subset, grouping.k)
        if complete:
            overall_ranks[key], within_ranks[key] = overall, within
        return overall, within

    for x, y, g in plan:
        grouping = prepared[g]
        mask = rows(x, y, g)
        codes = grouping.codes[mask]
        n = int(mask.sum())
        k = int(np.unique(codes).size)

        if np.unique(values[x][mask]).size < 3 or np.unique(values[y][mask]).size < 3:
            continue

        overall_rx, within_rx = ranks_for(x, g, mask)
        overall_ry, within_ry = ranks_for(y, g, mask)

        overall = _pearson(overall_rx, overall_ry)
        within = _pearson(within_rx, within_ry)

        if not (np.isfinite(overall) and np.isfinite(within)):
            continue

        overall_p = _p_value(overall, n - 2)
        within_p = _p_value(within, n - k - 1)

        # Per-group Spearman = Pearson of the within-group ranks in that group.
        products = np.bincount(
            codes, weights=within_rx * within_ry, minlength=grouping.k
        )
        squares_x = np.bincount(codes, weights=within_rx**2, minlength=grouping.k)
        squares_y = np.bincount(codes, weights=within_ry**2, minlength=grouping.k)
        counts = np.bincount(codes, minlength=grouping.k)

        group_stats: list[GroupStat] = []
        for code, level in enumerate(grouping.levels):
            if counts[code] == 0:
                continue
            denominator = math.sqrt(float(squares_x[code]) * float(squares_y[code]))
            rho = (
                float(products[code]) / denominator if denominator > 0 else float("nan")
            )
            group_stats.append(
                {"group": str(level), "n": int(counts[code]), "rho": rho}
            )

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
