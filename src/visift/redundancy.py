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

Joint determination by two columns (added after the pairwise rules; rule and
thresholds fixed, and checked on pure-noise data, before looking at their
effect on the real-world benchmark). Some columns are definitional only
through a *pair*: Bike Sharing's workingday is "not a weekend and not a
holiday", so neither weekday nor holiday determines it alone, and the
weekday-vs-workingday chart ranked in the top 3. For a column Y and two
other columns X1, X2:

- Categorical Y from two categorical columns: H(Y | X1, X2) / H(Y) <= 0.05
  (the pairwise threshold), with the pair's joint levels held to the same
  support rules as a single determining column (at most
  MAX_DETERMINANT_LEVELS observed combinations, at least MIN_ROWS_PER_LEVEL
  rows per combination on average). Plug-in conditional entropy is biased
  low when cells are small, so the observed ratio must also be at most
  JOINT_NULL_FRACTION of the smallest ratio over JOINT_NULL_PERMUTATIONS
  fixed-seed shuffles of Y (a determined column sits near 0 while chance
  structure in small cells stays near its shuffled value).
- Numeric Y as a sum or product of two numeric columns: R^2 >= 0.999 for
  Y ~ X1 + X2 (sums, differences, weighted totals such as cnt = casual +
  registered), or for log Y ~ log X1 + log X2 when all three are positive
  (products and ratios such as revenue = units x price). 0.999 means the
  residual standard deviation is about 3% of Y's: real measurements almost
  never get that close by chance.
- Both columns must be needed: neither may determine Y alone (H(Y|Xi)/H(Y)
  > 0.05; single-column R^2 <= 0.99), and a pair already flagged is skipped.

The charts between Y and each of X1 and X2 move to the dataset notes, with
one note for the triple. The chart between X1 and X2 is untouched.
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

JOINT_R2_DETERMINED = 0.999
JOINT_SINGLE_R2_MAX = 0.99
JOINT_NULL_PERMUTATIONS = 19
JOINT_NULL_FRACTION = 0.25
JOINT_SEED = 0

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
    "jointly_determined_categorical",
    "jointly_determined_numeric",
]

NEAR_DUPLICATE_KINDS = {"near_duplicate_numeric", "near_duplicate_categorical"}


class _RedundancyFields(TypedDict):
    kind: RedundancyKind
    # Near-duplicates: the pair in column order.
    # Determined: (determining column, determined column).
    # Jointly determined: (first determining column, second, determined column).
    columns: tuple[str, ...]
    statistic: str
    value: float
    reason: str


class Redundancy(_RedundancyFields, total=False):
    # Jointly determined only: every column of the triple that the other two
    # determine (charts between each of these and the rest are not ranked).
    determined: tuple[str, ...]


# --------------------------------------------------
# Statistics
# --------------------------------------------------


def _codes(series: pd.Series) -> np.ndarray:
    """Integer level codes in order of appearance, -1 for missing."""

    return np.asarray(pd.factorize(series, use_na_sentinel=True)[0])


def _numbers(series: pd.Series) -> np.ndarray:
    """Float values with NaN for missing (or unparseable) entries."""

    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float, na_value=np.nan)


def _compact(codes: np.ndarray) -> tuple[np.ndarray, int]:
    """Renumber the observed codes as 0..k-1; returns (codes, k)."""

    uniques, inverse = np.unique(codes, return_inverse=True)
    return inverse.reshape(-1), len(uniques)


def _counts(x: np.ndarray, n_x: int, y: np.ndarray, n_y: int) -> np.ndarray:
    """Contingency table of two compact code arrays (rows = x levels)."""

    cells = np.bincount(x * n_y + y, minlength=n_x * n_y)
    return cells.reshape(n_x, n_y).astype(float)


def _table(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Contingency table of two code arrays over the rows where both are
    present, with observed levels only (what pd.crosstab gives).
    """

    keep = (x >= 0) & (y >= 0)
    x_codes, n_x = _compact(x[keep])
    y_codes, n_y = _compact(y[keep])
    return _counts(x_codes, n_x, y_codes, n_y)


def _cramers_v(table: np.ndarray) -> float:
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


def _remaining_entropy(table: np.ndarray) -> float:
    """H(Y | X) / H(Y) for a table with X as rows and Y as columns."""

    total = float(table.sum())
    h_y = _entropy(table.sum(axis=0))

    if total == 0 or h_y == 0:
        return 1.0

    row_totals = table.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = table / row_totals
        terms = np.where(table > 0, p * np.log2(np.where(table > 0, p, 1.0)), 0.0)

    return float(-terms.sum(axis=1) @ (row_totals[:, 0] / total)) / h_y


def _eta_squared(x: np.ndarray, values: np.ndarray) -> float:
    """eta squared from compact group codes and complete float values."""

    mean = values.mean()
    total = float(((values - mean) ** 2).sum())

    if total == 0:
        return 0.0

    counts = np.bincount(x)
    sums = np.bincount(x, weights=values)
    seen = counts > 0
    between = float((counts[seen] * (sums[seen] / counts[seen] - mean) ** 2).sum())

    return between / total


def bias_corrected_cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bergsma (2013) bias-corrected Cramer's V for two categorical series."""

    return _cramers_v(_table(_codes(x), _codes(y)))


def normalized_conditional_entropy(x: pd.Series, y: pd.Series) -> float:
    """H(Y | X) / H(Y): 0 when X determines Y, 1 when X says nothing about Y."""

    return _remaining_entropy(_table(_codes(x), _codes(y)))


def eta_squared(groups: pd.Series, values: pd.Series) -> float:
    """Share of the variance in values explained by the group means."""

    codes = _codes(groups)
    numbers = values.to_numpy(dtype=float)
    keep = (codes >= 0) & ~np.isnan(numbers)
    return _eta_squared(_compact(codes[keep])[0], numbers[keep])


# --------------------------------------------------
# Detection
# --------------------------------------------------


def _types(profile: DatasetProfile) -> dict[str, str]:
    return {c["name"]: c["semantic_type"] for c in profile["column_profiles"]}


def _can_determine(levels: int, rows: float) -> bool:
    """Support rule for a determining column (or pair of columns)."""

    return 2 <= levels <= MAX_DETERMINANT_LEVELS and rows / levels >= MIN_ROWS_PER_LEVEL


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

    # Categorical checks share integer codes and cached contingency tables,
    # which keeps wide tables fast (pd.crosstab per pair was the bottleneck).
    codes = {c: _codes(df[c]) for c in categorical}
    tables: dict[tuple[str, str], np.ndarray] = {}

    def table(x: str, y: str) -> np.ndarray:
        """Rows = levels of x, columns = levels of y, both present."""

        if (y, x) in tables:
            return tables[(y, x)].T
        if (x, y) not in tables:
            tables[(x, y)] = _table(codes[x], codes[y])
        return tables[(x, y)]

    # Near-duplicate categorical columns
    #
    # Cramer's V reaches 1 whenever the coarser column is a function of the
    # finer one (quarter from month), so V alone cannot tell a twin from a
    # coarsening. A pair with V >= threshold is a near-duplicate only if each
    # column determines the other (same conditional-entropy threshold as
    # below); otherwise it is left to the one-way determinism check.
    for a, b in itertools.combinations(categorical, 2):
        counts = table(a, b)

        if counts.sum() < MIN_PAIR_SUPPORT:
            continue

        v = _cramers_v(counts)

        if v < CRAMERS_V_NEAR_DUPLICATE:
            continue

        if (
            _remaining_entropy(counts) <= CONDITIONAL_ENTROPY_MAX
            and _remaining_entropy(counts.T) <= CONDITIONAL_ENTROPY_MAX
        ):
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

        counts = table(x, y)
        rows = float(counts.sum())
        levels, y_levels = counts.shape

        if rows < MIN_PAIR_SUPPORT or not _can_determine(levels, rows):
            continue

        if y_levels < 2:
            continue

        remaining = _remaining_entropy(counts)

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
    numbers = {y: _numbers(df[y]) for y in numeric}

    for x, y in itertools.product(categorical, numeric):
        if frozenset((x, y)) in flagged_pairs:
            continue

        keep = (codes[x] >= 0) & ~np.isnan(numbers[y])
        rows = float(keep.sum())

        if rows < MIN_PAIR_SUPPORT:
            continue

        groups, levels = _compact(codes[x][keep])

        if not _can_determine(levels, rows):
            continue

        value = _eta_squared(groups, numbers[y][keep])

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

    found.extend(_joint_categorical(codes, flagged_pairs))
    found.extend(_joint_numeric(df, numeric, numbers, flagged_pairs))

    return found


def _redundant_pairs(redundancy: Redundancy) -> list[frozenset[str]]:
    columns = redundancy["columns"]

    if len(columns) == 2:
        return [frozenset(columns)]

    determined = redundancy.get("determined", (columns[-1],))

    return [
        frozenset((d, other)) for d in determined for other in columns if other != d
    ]


def flagged_pairs(redundancies: list[Redundancy]) -> set[frozenset[str]]:
    """Unordered column pairs that should not get a chart of their own."""

    return {pair for r in redundancies for pair in _redundant_pairs(r)}


# --------------------------------------------------
# Joint determination by two columns
# --------------------------------------------------


def _joint_categorical(
    codes: dict[str, np.ndarray], flagged: set[frozenset[str]]
) -> list[Redundancy]:
    """
    Unordered triples where one column is determined by the other two. Every
    column of the triple that passes is reported as determined (Bike
    Sharing: workingday from weekday + holiday, and holiday from weekday +
    workingday); the note names the one with the most entropy first.

    Loops over input pairs, not triples, so each pair's joint codes are
    built once; when no value is missing the codes are reused as they are.
    """

    columns = list(codes)

    if len(columns) < 3:
        return []

    n_rows = len(codes[columns[0]])
    order = {c: k for k, c in enumerate(columns)}
    width = {c: int(np.max(codes[c], initial=-1)) + 1 for c in columns}
    complete = {c: bool((codes[c] >= 0).all()) for c in columns}
    inputs = [c for c in columns if 2 <= width[c] <= MAX_DETERMINANT_LEVELS]
    passes: dict[tuple[str, ...], list[tuple[float, float, str, str, str]]] = {}

    for x1, x2 in itertools.combinations(inputs, 2):
        if frozenset((x1, x2)) in flagged:
            continue

        joint = codes[x1] * width[x2] + codes[x2]
        both = (codes[x1] >= 0) & (codes[x2] >= 0)
        whole = _compact(joint) if complete[x1] and complete[x2] else None

        for y in columns:
            if y in (x1, x2) or {frozenset((x1, y)), frozenset((x2, y))} & flagged:
                continue

            rows: np.ndarray | None = None

            if whole is not None and complete[y]:
                (joint_codes, n_joint), y_codes, n_y = whole, codes[y], width[y]
                n = n_rows
            else:
                rows = both & (codes[y] >= 0)
                n = int(rows.sum())
                if n < MIN_PAIR_SUPPORT:
                    continue
                joint_codes, n_joint = _compact(joint[rows])
                y_codes, n_y = _compact(codes[y][rows])

            if n < MIN_PAIR_SUPPORT or n_y < 2 or not _can_determine(n_joint, n):
                continue

            ratio = _remaining_entropy(_counts(joint_codes, n_joint, y_codes, n_y))

            if ratio > CONDITIONAL_ENTROPY_MAX:
                continue

            # Rare from here on. Each column alone must leave Y undetermined.
            alone = [
                _remaining_entropy(
                    _counts(
                        *_compact(codes[x] if rows is None else codes[x][rows]),
                        y_codes,
                        n_y,
                    )
                )
                for x in (x1, x2)
            ]

            if min(alone) <= CONDITIONAL_ENTROPY_MAX:
                continue

            # Small cells make plug-in entropy look low by chance.
            rng = np.random.default_rng(JOINT_SEED)
            null = min(
                _remaining_entropy(
                    _counts(joint_codes, n_joint, rng.permutation(y_codes), n_y)
                )
                for _ in range(JOINT_NULL_PERMUTATIONS)
            )

            if ratio > JOINT_NULL_FRACTION * null:
                continue

            entropy = _entropy(np.bincount(y_codes).astype(float))
            key = tuple(sorted((x1, x2, y), key=order.__getitem__))
            passes.setdefault(key, []).append((entropy, ratio, x1, x2, y))

    found: list[Redundancy] = []

    # Report in column order; a triple sharing a pair with one already
    # reported adds nothing new.
    for key in sorted(passes, key=lambda t: [order[c] for c in t]):
        if any(frozenset(p) in flagged for p in itertools.combinations(key, 2)):
            continue

        passed = sorted(passes[key], key=lambda item: (-item[0], order[item[4]]))
        _, ratio, x1, x2, y = passed[0]

        reason = (
            f"{y} is almost fully determined by {x1} and {x2} together (knowing "
            f"both leaves {ratio:.0%} of the uncertainty in {y})"
        )
        for _, _, other_x1, other_x2, other in passed[1:]:
            reason += f"; {other} is likewise determined by {other_x1} and {other_x2}"
        reason += ". Charts between these columns are not ranked."

        note: Redundancy = {
            "kind": "jointly_determined_categorical",
            "columns": (x1, x2, y),
            "statistic": "H(Y|X1,X2)/H(Y)",
            "value": ratio,
            "reason": reason,
            "determined": tuple(item[4] for item in passed),
        }
        found.append(note)
        flagged.update(_redundant_pairs(note))

    return found


def _formula(
    y: str, x1: str, x2: str, coef: np.ndarray, intercept: float, log: bool
) -> str:
    def number(value: float) -> str:
        return f"{value:.3g}"

    if log:
        scale = (
            "" if abs(intercept) < 1e-3 else f"{number(float(np.exp(intercept)))} × "
        )
        return f"{y} ≈ {scale}{x1}^{number(coef[0])} × {x2}^{number(coef[1])}"

    terms = f"{number(coef[0])}·{x1} {'+' if coef[1] >= 0 else '−'} "
    terms += f"{number(abs(coef[1]))}·{x2}"
    if abs(intercept) >= 1e-6:
        terms += f" {'+' if intercept >= 0 else '−'} {number(abs(intercept))}"
    return f"{y} ≈ {terms}"


def _fit(values: pd.DataFrame, x1: str, x2: str, y: str) -> tuple[float, np.ndarray]:
    """R^2 and (coef x1, coef x2, intercept) of y ~ x1 + x2 on complete rows."""

    rows = values[[x1, x2, y]].dropna()
    if len(rows) < MIN_PAIR_SUPPORT:
        return 0.0, np.zeros(3)

    design = np.column_stack(
        [rows[x1].to_numpy(), rows[x2].to_numpy(), np.ones(len(rows))]
    )
    target = rows[y].to_numpy()
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    total = float(((target - target.mean()) ** 2).sum())
    if total == 0:
        return 0.0, solution

    residual = target - design @ solution
    return 1 - float((residual**2).sum()) / total, solution


def _joint_numeric(
    df: pd.DataFrame,
    numeric: list[str],
    numbers: dict[str, np.ndarray],
    flagged: set[frozenset[str]],
) -> list[Redundancy]:
    """
    Unordered triples where one column is (almost exactly) a linear or
    log-linear combination of the other two. Such an identity holds in every
    orientation (cnt = casual + registered, casual = cnt - registered), so
    only one is reported: the one whose two coefficients are positive (a
    total or a product), else the one with the most variance. Only its two
    charts are set aside; the chart between the two inputs is a real
    relationship and stays ranked.
    """

    found: list[Redundancy] = []

    if len(numeric) < 3:
        return found

    raw = pd.DataFrame(numbers, index=df.index, columns=numeric)
    spaces: list[tuple[bool, pd.DataFrame]] = [(False, raw)]
    positive = [c for c in numeric if bool((raw[c].dropna() > 0).all())]
    if len(positive) >= 3:
        spaces.append((True, pd.DataFrame(np.log(raw[positive]))))

    seen: set[frozenset[str]] = set()

    for log, values in spaces:
        columns = list(values.columns)
        corr = values.corr(min_periods=MIN_PAIR_SUPPORT).to_numpy()
        index = {c: k for k, c in enumerate(columns)}

        for triple in itertools.combinations(columns, 3):
            if frozenset(triple) in seen:
                continue
            if any(frozenset(p) in flagged for p in itertools.combinations(triple, 2)):
                continue

            orientations = []

            for y in triple:
                x1, x2 = (c for c in triple if c != y)
                r1 = corr[index[x1], index[y]]
                r2 = corr[index[x2], index[y]]
                r12 = corr[index[x1], index[x2]]

                if not np.isfinite([r1, r2, r12]).all() or abs(r12) >= 0.999:
                    continue
                if max(r1 * r1, r2 * r2) > JOINT_SINGLE_R2_MAX:
                    continue
                if (r1 * r1 + r2 * r2 - 2 * r1 * r2 * r12) / (
                    1 - r12 * r12
                ) < JOINT_R2_DETERMINED:
                    continue

                r_squared, solution = _fit(values, x1, x2, y)
                if r_squared < JOINT_R2_DETERMINED:
                    continue

                both_positive = bool(solution[0] > 0 and solution[1] > 0)
                spread = float(values[y].var())
                orientations.append(
                    (both_positive, spread, r_squared, solution, x1, x2, y)
                )

            if not orientations:
                continue

            orientations.sort(key=lambda item: (not item[0], -item[1]))
            _, _, r_squared, solution, x1, x2, y = orientations[0]
            formula = _formula(y, x1, x2, solution[:2], float(solution[2]), log)

            note: Redundancy = {
                "kind": "jointly_determined_numeric",
                "columns": (x1, x2, y),
                "statistic": "R² (log scale)" if log else "R²",
                "value": r_squared,
                "reason": (
                    f"{y} is almost exactly computed from {x1} and {x2} "
                    f"({formula}, R² {r_squared:.3f}), so charts of {y} against "
                    "either are not ranked."
                ),
                "determined": (y,),
            }
            found.append(note)
            seen.add(frozenset(triple))
            flagged.update(_redundant_pairs(note))

    return found


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
