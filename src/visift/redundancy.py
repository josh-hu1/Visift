"""
Redundancy and triviality detection.

Some column pairs are related by construction rather than by anything worth
discovering: a "feels like" temperature that tracks the measured one, two
encodings of the same contact history, a flag that is a function of another
column. Charts *between* such pairs score very highly and would otherwise
crowd the top of the ranking.

find_redundancies() reports these pairs with a statistic and a plain-English
reason. The pipeline (visift.pipeline) then:

- treats each near-duplicate cluster as one variable when diversifying, so
  charts that differ only by swapping a column for its twin fold together;
- moves charts between flagged pairs to the dataset notes instead of the
  ranking. Scores are never changed and nothing is deleted.

Thresholds (pre-registered before inspecting their effect on the real-world
benchmark rankings; chosen on principle, not to catch particular columns):

- Near-duplicate numeric: |Spearman| >= 0.97. Rank correlation 0.97 means
  the two orderings almost coincide (about 6% of the variance of one rank
  is not explained by the other). Planted "strong" relationships in the
  synthetic suites (r ~ 0.9, Spearman ~ 0.89) stay well below it.
- Near-duplicate categorical: bias-corrected Cramer's V >= 0.95 (Bergsma
  2013 correction, so small tables with many levels are not inflated).
  Because V also reaches 1 for a coarsening (quarter from month), a pair is
  labeled a near-duplicate only if each column also determines the other
  under the conditional-entropy threshold below; a one-way coarsening is
  reported as near-deterministic instead.
- Near-deterministic categorical -> categorical: normalized conditional
  entropy H(Y|X) / H(Y) <= 0.05, i.e. knowing X leaves at most 5% of Y's
  uncertainty.
- Near-deterministic categorical -> numeric: eta squared >= 0.95, i.e. the
  grouping explains at least 95% of the numeric variance.

Support requirements: at least MIN_PAIR_SUPPORT pairwise-complete rows, and
a determining column needs at most MAX_DETERMINANT_LEVELS levels with at
least MIN_ROWS_PER_LEVEL rows per level on average (otherwise any
near-unique column would trivially "determine" everything).
"""

import itertools
from typing import Literal, TypedDict

import numpy as np
import pandas as pd

from visift.types import DatasetProfile

SPEARMAN_NEAR_DUPLICATE = 0.97
CRAMERS_V_NEAR_DUPLICATE = 0.95
CONDITIONAL_ENTROPY_MAX = 0.05
ETA_SQUARED_DETERMINED = 0.95

MIN_PAIR_SUPPORT = 30
MAX_DETERMINANT_LEVELS = 50
MIN_ROWS_PER_LEVEL = 5

NUMERIC_TYPES = {"numeric_continuous", "numeric_discrete"}
CATEGORY_TYPES = {"categorical", "ordinal", "boolean", "geographic"}
# Numeric-coded calendar fields (mnth, yr, weekday codes) behave like
# categories for determinism checks when they have few levels.
CODED_TIME_TYPES = {"temporal"}

RedundancyKind = Literal[
    "near_duplicate_numeric",
    "near_duplicate_categorical",
    "determined_categorical",
    "determined_numeric",
]

NEAR_DUPLICATE_KINDS = {"near_duplicate_numeric", "near_duplicate_categorical"}


class Redundancy(TypedDict):
    kind: RedundancyKind
    # Near-duplicates: the pair in column order.
    # Determined: (determining column, determined column).
    columns: tuple[str, str]
    statistic: str
    value: float
    reason: str


# --------------------------------------------------
# Statistics
# --------------------------------------------------


def bias_corrected_cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bergsma (2013) bias-corrected Cramer's V for two categorical series."""

    table = pd.crosstab(x, y).to_numpy(dtype=float)
    n = float(table.sum())
    rows, cols = table.shape

    if n < 2 or rows < 2 or cols < 2:
        return 0.0

    expected = table.sum(axis=1, keepdims=True) * table.sum(axis=0, keepdims=True) / n
    chi2 = float(((table - expected) ** 2 / expected).sum())

    phi2 = chi2 / n
    phi2_corrected = max(0.0, phi2 - (cols - 1) * (rows - 1) / (n - 1))
    rows_corrected = rows - (rows - 1) ** 2 / (n - 1)
    cols_corrected = cols - (cols - 1) ** 2 / (n - 1)
    denominator = min(cols_corrected - 1, rows_corrected - 1)

    if denominator <= 0:
        return 0.0

    return float(np.sqrt(phi2_corrected / denominator))


def _entropy(counts: np.ndarray) -> float:
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def normalized_conditional_entropy(x: pd.Series, y: pd.Series) -> float:
    """H(Y | X) / H(Y): 0 when X determines Y, 1 when X says nothing about Y."""

    table = pd.crosstab(x, y).to_numpy(dtype=float)
    total = float(table.sum())
    h_y = _entropy(table.sum(axis=0))

    if total == 0 or h_y == 0:
        return 1.0

    h_y_given_x = sum(
        (float(row.sum()) / total) * _entropy(row) for row in table if row.sum() > 0
    )

    return float(h_y_given_x / h_y)


def eta_squared(groups: pd.Series, values: pd.Series) -> float:
    """Share of the variance in values explained by the group means."""

    total = float(((values - values.mean()) ** 2).sum())

    if total == 0:
        return 0.0

    means = values.groupby(groups).transform("mean")
    between = float(((means - values.mean()) ** 2).sum())

    return between / total


# --------------------------------------------------
# Detection
# --------------------------------------------------


def _types(profile: DatasetProfile) -> dict[str, str]:
    return {c["name"]: c["semantic_type"] for c in profile["column_profiles"]}


def _can_determine(series: pd.Series) -> bool:
    levels = series.nunique()
    return 2 <= levels <= MAX_DETERMINANT_LEVELS and (
        len(series) / levels >= MIN_ROWS_PER_LEVEL
    )


def find_redundancies(df: pd.DataFrame, profile: DatasetProfile) -> list[Redundancy]:
    """Near-duplicate and near-deterministic column pairs, in column order."""

    types = _types(profile)
    columns = [str(c) for c in df.columns]

    numeric = [c for c in columns if types.get(c) in NUMERIC_TYPES]
    categorical = [
        c
        for c in columns
        if types.get(c) in CATEGORY_TYPES
        or (
            types.get(c) in CODED_TIME_TYPES
            and df[c].nunique() <= MAX_DETERMINANT_LEVELS
        )
    ]

    found: list[Redundancy] = []
    flagged_pairs: set[frozenset[str]] = set()

    # Near-duplicate numeric columns
    for a, b in itertools.combinations(numeric, 2):
        pair = df[[a, b]].dropna()

        if (
            len(pair) < MIN_PAIR_SUPPORT
            or pair[a].nunique() < 2
            or pair[b].nunique() < 2
        ):
            continue

        rho = float(pair[a].rank().corr(pair[b].rank()))

        if np.isfinite(rho) and abs(rho) >= SPEARMAN_NEAR_DUPLICATE:
            direction = "together" if rho > 0 else "in opposite directions"
            found.append(
                {
                    "kind": "near_duplicate_numeric",
                    "columns": (a, b),
                    "statistic": "Spearman",
                    "value": rho,
                    "reason": (
                        f"{a} and {b} move {direction} (Spearman {rho:.2f}), "
                        "so Visift treats them as one variable."
                    ),
                }
            )
            flagged_pairs.add(frozenset((a, b)))

    # Near-duplicate categorical columns
    #
    # Cramer's V reaches 1 whenever the coarser column is a function of the
    # finer one (quarter from month), so V alone cannot tell a twin from a
    # coarsening. A pair with V >= threshold is a near-duplicate only if each
    # column determines the other (same conditional-entropy threshold as
    # below); otherwise it is left to the one-way determinism check.
    for a, b in itertools.combinations(categorical, 2):
        pair = df[[a, b]].dropna()

        if len(pair) < MIN_PAIR_SUPPORT:
            continue

        v = bias_corrected_cramers_v(pair[a], pair[b])

        mutual = (
            normalized_conditional_entropy(pair[a], pair[b]) <= CONDITIONAL_ENTROPY_MAX
            and normalized_conditional_entropy(pair[b], pair[a])
            <= CONDITIONAL_ENTROPY_MAX
        )

        if v >= CRAMERS_V_NEAR_DUPLICATE and mutual:
            found.append(
                {
                    "kind": "near_duplicate_categorical",
                    "columns": (a, b),
                    "statistic": "Cramér's V",
                    "value": v,
                    "reason": (
                        f"{a} and {b} carry almost the same information "
                        f"(Cramér's V {v:.2f}), so Visift treats them as one variable."
                    ),
                }
            )
            flagged_pairs.add(frozenset((a, b)))

    # Near-deterministic: categorical -> categorical
    for x, y in itertools.permutations(categorical, 2):
        if frozenset((x, y)) in flagged_pairs:
            continue

        pair = df[[x, y]].dropna()

        if len(pair) < MIN_PAIR_SUPPORT or not _can_determine(pair[x]):
            continue

        if pair[y].nunique() < 2:
            continue

        remaining = normalized_conditional_entropy(pair[x], pair[y])

        if remaining <= CONDITIONAL_ENTROPY_MAX:
            found.append(
                {
                    "kind": "determined_categorical",
                    "columns": (x, y),
                    "statistic": "H(Y|X)/H(Y)",
                    "value": remaining,
                    "reason": (
                        f"{y} is almost fully determined by {x} (knowing {x} "
                        f"leaves {remaining:.0%} of the uncertainty in {y})."
                    ),
                }
            )
            flagged_pairs.add(frozenset((x, y)))

    # Near-deterministic: categorical -> numeric
    for x, y in itertools.product(categorical, numeric):
        if frozenset((x, y)) in flagged_pairs:
            continue

        pair = df[[x, y]].dropna()

        if len(pair) < MIN_PAIR_SUPPORT or not _can_determine(pair[x]):
            continue

        value = eta_squared(pair[x], pair[y].astype(float))

        if value >= ETA_SQUARED_DETERMINED:
            found.append(
                {
                    "kind": "determined_numeric",
                    "columns": (x, y),
                    "statistic": "η²",
                    "value": value,
                    "reason": (
                        f"{x} explains {value:.0%} of the variance in {y} "
                        f"(η² {value:.2f}), so {y} is almost a relabeling of {x}."
                    ),
                }
            )
            flagged_pairs.add(frozenset((x, y)))

    return found


def flagged_pairs(redundancies: list[Redundancy]) -> set[frozenset[str]]:
    """Unordered column pairs that should not get a chart of their own."""

    return {frozenset(r["columns"]) for r in redundancies}


def duplicate_aliases(
    df: pd.DataFrame, redundancies: list[Redundancy], target: str | None = None
) -> dict[str, str]:
    """
    Map every member of a near-duplicate cluster to its representative.

    Representative: the target if it is in the cluster; otherwise the
    column with the fewest missing values, then the earliest column.
    Columns outside any cluster are not in the mapping.
    """

    parent: dict[str, str] = {}

    def find(column: str) -> str:
        while parent.setdefault(column, column) != column:
            column = parent[column]
        return column

    for r in redundancies:
        if r["kind"] in NEAR_DUPLICATE_KINDS:
            a, b = r["columns"]
            parent[find(a)] = find(b)

    clusters: dict[str, list[str]] = {}
    for column in parent:
        clusters.setdefault(find(column), []).append(column)

    order = {str(c): i for i, c in enumerate(df.columns)}
    aliases: dict[str, str] = {}

    for members in clusters.values():
        if target in members:
            representative = target
        else:
            representative = min(
                members, key=lambda c: (int(df[c].isna().sum()), order.get(c, 0))
            )

        for member in members:
            aliases[member] = representative

    return aliases
