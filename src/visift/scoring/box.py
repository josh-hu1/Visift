import math
from collections.abc import Sequence
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import kruskal, levene

from visift.types import Candidate, DatasetProfile, ScoreResult

CATEGORY_TYPES = {"categorical", "ordinal", "boolean", "geographic"}

NUMERIC_TYPES = {"numeric_continuous", "numeric_discrete"}

# ==================================================
# SIGNIFICANCE GATES (Phase 2.6, pre-registered)
# ==================================================
#
# Each of the three grouped-distribution signals (location, dispersion,
# outlier pattern) is multiplied by a 0-1 reliability derived from a
# significance test of the same kind of difference:
#
# - location: Kruskal-Wallis across groups
# - dispersion: Brown-Forsythe (Levene with median centering) across the
#   groups the IQR ratio uses (>= 8 rows)
# - outlier pattern: chi-square statistic of per-group outlier rates with a
#   permutation p-value (see amendment 1 below)
#
# Reliability ramp: the one the line scorer already uses for its Levene-gated
# volatility signal (full weight at p <= 0.001, zero at p >= 0.10, linear in
# log10(p) in between). A smooth ramp avoids a cliff at an arbitrary alpha
# and keeps the scorers consistent. The underlying 0-100 scales are
# unchanged; only how much of each signal is trusted changes.
#
# Why: on pure noise the ungated signals score >= 50 in 4.8% of the test
# suite's null box cases and in 20-81% of cases with ~20 rows per group,
# driven by chance IQR ratios (dispersion), chance outlier-rate gaps
# (outlier) and, rarely, location.
#
# Amendments, made after the pre-registered version (commit 9ce488f) failed
# its acceptance criteria on the validation data (null >= 50 up to 5.0% per
# cell, and some >= 65):
#
# 1. The pre-registered outlier test permuted the 0/1 outlier indicators.
#    That is invalid: each point's outlier status depends on its own
#    group's estimated fences, so indicators are not exchangeable across
#    groups, and under pure noise the test gave p < 0.01 in 3-6% of cases.
#    The test now permutes the raw values across groups and recomputes every
#    group's fences and the chi-square statistic each time (a valid
#    permutation test of identical distributions). For speed it stops early
#    once 20 permutations reach the observed statistic (Besag & Clifford
#    1991 sequential p-value), so the full 999 run only when p is small.
# 2. The signal is the maximum of three gated components, so their p-values
#    are Bonferroni-corrected (x3) before the reliability ramp.
#
# Thresholds, the ramp and the score scales are unchanged.

GATE_STRONG_P = 0.001
GATE_WEAK_P = 0.10
OUTLIER_TEST_PERMUTATIONS = 999
OUTLIER_TEST_STOP_AFTER = 20
OUTLIER_TEST_SEED = 0
MIN_ROWS_FOR_SPREAD = 8
GATED_COMPONENTS = 3


def clamp(value: float, minimum: float = 0, maximum: float = 100) -> float:
    """
    Keep a score between 0 and 100.
    """
    return max(minimum, min(value, maximum))


def piecewise_score(value: float, points: Sequence[tuple[float, float]]) -> float:
    """
    Map a statistic to a calibrated score using
    linear interpolation between control points.
    """

    if value is None:
        return 0

    try:
        value = float(value)

    except (TypeError, ValueError):
        return 0

    if not math.isfinite(value):
        return 0

    if value <= points[0][0]:
        return points[0][1]

    if value >= points[-1][0]:
        return points[-1][1]

    for (x0, y0), (x1, y1) in zip(points, points[1:], strict=False):
        if x0 <= value <= x1:
            if x1 == x0:
                return y1

            fraction = (value - x0) / (x1 - x0)

            return y0 + fraction * (y1 - y0)

    return 0


class _Pair:
    """
    The rows where x and y are both present, and y's values split by group,
    computed once per chart and shared by the component scores. Groups come
    in groupby's sorted key order and rows keep their original order, so
    each group's values are the same array pandas would hand out. Means are
    NumPy's (pandas computes the same sum / count) and quartiles are
    np.percentile's linear method (which pandas calls), so on NumPy-backed
    columns every number is unchanged.
    """

    def __init__(self, df: pd.DataFrame, x: str, y: str) -> None:
        self.x = x
        self.y = y
        self.clean = df[[x, y]].dropna()

    @cached_property
    def grouped(self) -> Any:
        return self.clean.groupby(self.x, observed=True)

    @cached_property
    def values(self) -> np.ndarray:
        """y as a NumPy array (its own dtype when NumPy-backed, else float)."""

        column = self.clean[self.y]
        if isinstance(column.dtype, np.dtype) and column.dtype.kind in "iuf":
            return column.to_numpy()
        return column.to_numpy(dtype=float)

    @cached_property
    def sizes(self) -> np.ndarray:
        return np.bincount(self._codes)

    @cached_property
    def _codes(self) -> np.ndarray:
        """Group number of each row, numbered in groupby's order."""

        column = self.clean[self.x]
        if isinstance(column.dtype, pd.CategoricalDtype):
            codes = self.grouped.ngroup().to_numpy()
        else:
            # What groupby itself uses to number non-categorical keys.
            codes, _ = pd.factorize(column, sort=True)
        return np.asarray(codes, dtype=np.int64)

    @cached_property
    def group_values(self) -> list[np.ndarray]:
        """y's values in each group, in groupby's order."""

        if self.clean.empty:
            return []
        labels = self._codes.astype(_label_dtype(len(self.sizes)))
        order = np.argsort(labels, kind="stable")
        return np.split(self.values[order], np.cumsum(self.sizes)[:-1])

    @cached_property
    def quartiles(self) -> list[tuple[np.ndarray, float, float]]:
        """(values, Q1, Q3) of y for each group with at least 8 rows."""

        found = []
        for values in self.group_values:
            if len(values) < 8:
                continue
            q1, q3 = _quartiles(values)
            found.append((values, q1, q3))
        return found


def _quartiles(values: np.ndarray) -> tuple[float, float]:
    """Q1 and Q3 exactly as pandas' Series.quantile computes them."""

    q1, q3 = np.percentile(values, [25.0, 75.0])
    return float(q1), float(q3)


def _pair(df: pd.DataFrame, x: str, y: str, pair: _Pair | None) -> _Pair:
    return pair if pair is not None else _Pair(df, x, y)


def get_semantic_type(profile: DatasetProfile, column_name: str) -> str | None:
    for column in profile["column_profiles"]:
        if column["name"] == column_name:
            return column["semantic_type"]
    return None


def semantic_fit_score(x_type: str | None, y_type: str | None) -> float:
    category_scores: dict[str | None, float] = {
        "categorical": 100,
        "ordinal": 95,
        "boolean": 85,
        "geographic": 60,
    }
    numeric_scores: dict[str | None, float] = {
        "numeric_continuous": 100,
        "numeric_discrete": 85,
    }
    x_score = category_scores.get(x_type, 0)
    y_score = numeric_scores.get(y_type, 0)
    if x_score == 0 or y_score == 0:
        return 0
    return 0.60 * x_score + 0.40 * y_score


def category_readability_score(unique_count: int) -> float:
    if unique_count <= 1:
        return 0
    if unique_count <= 6:
        return 100
    if unique_count <= 10:
        return 90
    if unique_count <= 15:
        return 70
    if unique_count <= 25:
        return 45
    return 20


def data_quality_score(
    df: pd.DataFrame, x: str, y: str, pair: _Pair | None = None
) -> float:
    if len(df) == 0:
        return 0
    complete = _pair(df, x, y, pair).clean
    return (len(complete) / len(df)) * 100


def sample_support_score(counts: pd.Series) -> float:
    if counts.empty:
        return 0
    average_group_size = counts.mean()
    minimum_group_size = counts.min()
    average_score = clamp(average_group_size / 20 * 100)
    minimum_score = clamp(minimum_group_size / 10 * 100)
    return 0.50 * average_score + 0.50 * minimum_score


def significance_reliability(p_value: float) -> float:
    """0 at p >= GATE_WEAK_P, 1 at p <= GATE_STRONG_P, log-linear between."""

    if p_value is None or not np.isfinite(p_value) or p_value >= GATE_WEAK_P:
        return 0.0

    if p_value <= GATE_STRONG_P:
        return 1.0

    position = (math.log10(GATE_WEAK_P) - math.log10(p_value)) / (
        math.log10(GATE_WEAK_P) - math.log10(GATE_STRONG_P)
    )

    return float(clamp(position, 0, 1))


def _group_values(
    df: pd.DataFrame, x: str, y: str, minimum_rows: int, pair: _Pair | None = None
) -> list[np.ndarray]:
    return [
        values.astype(float)
        for values in _pair(df, x, y, pair).group_values
        if len(values) >= minimum_rows
    ]


def location_p_value(
    df: pd.DataFrame, x: str, y: str, pair: _Pair | None = None
) -> float:
    """Kruskal-Wallis p-value for a difference in location across groups."""

    groups = _group_values(df, x, y, minimum_rows=2, pair=pair)

    if len(groups) < 2:
        return 1.0

    try:
        p_value = float(kruskal(*groups).pvalue)
    except ValueError:  # all values identical
        return 1.0

    return p_value if np.isfinite(p_value) else 1.0


def dispersion_p_value(
    df: pd.DataFrame, x: str, y: str, pair: _Pair | None = None
) -> float:
    """Brown-Forsythe p-value for a difference in spread across groups."""

    groups = _group_values(df, x, y, minimum_rows=MIN_ROWS_FOR_SPREAD, pair=pair)

    if len(groups) < 2:
        return 1.0

    p_value = float(levene(*groups, center="median").pvalue)

    return p_value if np.isfinite(p_value) else 1.0


def _label_dtype(k: int) -> type[np.unsignedinteger[Any]]:
    if k <= np.iinfo(np.uint8).max + 1:
        return np.uint8
    if k <= np.iinfo(np.uint16).max + 1:
        return np.uint16
    return np.uint64


class _OutlierStatistic:
    """
    Chi-square statistic of per-group outlier rates, where each group's
    outliers fall outside its own 1.5 x IQR fences (quartiles as in
    np.quantile's default linear method). Groups with IQR 0 have no
    outliers.

    Shuffling values across fixed group labels is the same as shuffling the
    labels across fixed values, so the values are sorted once and each
    permutation only needs a stable integer sort of the labels; within each
    group the values then stay in ascending order.
    """

    def __init__(self, values: np.ndarray, codes: np.ndarray, k: int) -> None:
        # Small unsigned labels let NumPy's stable sort use radix sort, which
        # is several times faster here and gives the same (unique) order.
        self.codes = codes.astype(_label_dtype(k))
        self.k = k
        self.n = len(values)
        self.sort_index = np.argsort(values, kind="stable")
        self.sorted_values = values[self.sort_index]
        self.counts = np.bincount(codes, minlength=k).astype(float)
        starts = np.concatenate([[0], np.cumsum(self.counts)[:-1]]).astype(int)
        sizes = self.counts.astype(int)

        positions = [(sizes - 1) * q for q in (0.25, 0.75)]
        self.lower = [starts + np.floor(pos).astype(int) for pos in positions]
        self.upper = [starts + np.ceil(pos).astype(int) for pos in positions]
        self.weight = [pos - np.floor(pos) for pos in positions]

    def __call__(self, permutation: np.ndarray | None = None) -> float:
        """Statistic for values[permutation] against the fixed group codes."""

        if permutation is None:
            labels = self.codes
        else:
            # values[permutation][i] = values[permutation[i]] sits in group
            # codes[i], so value permutation[i] gets label codes[i].
            labels = np.empty_like(self.codes)
            labels[permutation] = self.codes

        sorted_labels = labels[self.sort_index]
        ordered = self.sorted_values[np.argsort(sorted_labels, kind="stable")]

        q1, q3 = (
            ordered[lo] + (ordered[hi] - ordered[lo]) * w
            for lo, hi, w in zip(self.lower, self.upper, self.weight, strict=True)
        )
        iqr = q3 - q1
        valid = np.isfinite(iqr) & (iqr > 0)
        low_fence = np.where(valid, q1 - 1.5 * iqr, -np.inf)
        high_fence = np.where(valid, q3 + 1.5 * iqr, np.inf)

        outside = (self.sorted_values < low_fence[sorted_labels]) | (
            self.sorted_values > high_fence[sorted_labels]
        )
        hits = np.bincount(sorted_labels, weights=outside, minlength=self.k)

        rate = float(hits.sum()) / float(self.counts.sum())

        if rate <= 0 or rate >= 1:
            return 0.0

        expected_hits = self.counts * rate
        expected_misses = self.counts * (1 - rate)
        return float(
            ((hits - expected_hits) ** 2 / expected_hits).sum()
            + ((self.counts - hits - expected_misses) ** 2 / expected_misses).sum()
        )


def outlier_p_value(
    df: pd.DataFrame, x: str, y: str, pair: _Pair | None = None
) -> float:
    """
    Permutation p-value that per-group outlier rates are equal: values are
    shuffled across groups and every group's fences recomputed, with
    Besag-Clifford sequential stopping after OUTLIER_TEST_STOP_AFTER
    permutations reach the observed statistic.
    """

    groups = _group_values(df, x, y, minimum_rows=MIN_ROWS_FOR_SPREAD, pair=pair)

    if len(groups) < 2:
        return 1.0

    values = np.concatenate(groups)
    codes = np.concatenate([np.full(len(g), i) for i, g in enumerate(groups)])
    statistic = _OutlierStatistic(values, codes, len(groups))

    observed = statistic()

    if observed <= 0:
        return 1.0

    rng = np.random.default_rng(OUTLIER_TEST_SEED)
    exceed = 0

    for done in range(1, OUTLIER_TEST_PERMUTATIONS + 1):
        if statistic(rng.permutation(len(values))) >= observed - 1e-12:
            exceed += 1
            if exceed >= OUTLIER_TEST_STOP_AFTER:
                return exceed / done

    return (1 + exceed) / (1 + OUTLIER_TEST_PERMUTATIONS)


def eta_squared_score(
    df: pd.DataFrame, x: str, y: str, pair: _Pair | None = None
) -> float:
    pair = _pair(df, x, y, pair)
    clean = pair.clean
    if clean.empty or len(pair.sizes) <= 1:
        return 0
    overall_mean = clean[y].mean()
    total_variation = ((clean[y] - overall_mean) ** 2).sum()
    if total_variation == 0:
        return 0
    between_group_variation = 0
    for values in pair.group_values:
        group_mean = values.mean()
        between_group_variation += len(values) * (group_mean - overall_mean) ** 2
    eta_squared = between_group_variation / total_variation
    return clamp(eta_squared * 100)


def median_separation_score(
    df: pd.DataFrame, x: str, y: str, pair: _Pair | None = None
) -> float:
    pair = _pair(df, x, y, pair)
    if pair.clean.empty:
        return 0
    medians = pair.grouped[y].median()
    if len(medians) <= 1:
        return 0
    median_range = medians.max() - medians.min()
    q1, q3 = _quartiles(pair.values)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    relative_separation = median_range / iqr
    return clamp(relative_separation / 2 * 100)


def grouped_outlier_statistics(
    df: pd.DataFrame, x: str, y: str, pair: _Pair | None = None
) -> dict[str, Any]:
    pair = _pair(df, x, y, pair)
    clean = pair.clean
    default = {
        "overall_rate": 0.0,
        "rate_disparity": 0.0,
        "score": 0.0,
        "minimum_rate": 0.0,
        "maximum_rate": 0.0,
    }
    if clean.empty:
        return default
    rates = []
    weighted_outliers = 0
    weighted_observations = 0
    for values, q1, q3 in pair.quartiles:
        iqr = q3 - q1
        if pd.isna(iqr) or iqr <= 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(np.count_nonzero((values < lower) | (values > upper)))
        rate = outlier_count / len(values)
        rates.append(rate)
        weighted_outliers += outlier_count
        weighted_observations += len(values)
    if len(rates) < 2 or weighted_observations == 0:
        return default
    rates_series = pd.Series(rates)
    median_rate = float(rates_series.median())
    max_rate = float(rates_series.max())
    min_rate = float(rates_series.min())
    disparity = max(0.0, max_rate - median_rate)
    score = piecewise_score(
        disparity,
        [
            (0.000, 0),
            (0.015, 0),
            (0.030, 10),
            (0.060, 25),
            (0.100, 45),
            (0.200, 65),
            (0.300, 78),
            (0.400, 88),
        ],
    )
    return {
        "overall_rate": weighted_outliers / weighted_observations,
        "rate_disparity": disparity,
        "score": clamp(score),
        "minimum_rate": min_rate,
        "maximum_rate": max_rate,
    }


def dispersion_difference_score(
    df: pd.DataFrame, x: str, y: str, pair: _Pair | None = None
) -> dict[str, Any]:
    pair = _pair(df, x, y, pair)
    clean = pair.clean
    default = {
        "score": 0.0,
        "dispersion_ratio": 1.0,
        "minimum_iqr": 0.0,
        "maximum_iqr": 0.0,
    }
    if clean.empty:
        return default
    iqrs = []
    for _values, q1, q3 in pair.quartiles:
        iqr = float(q3 - q1)
        if math.isfinite(iqr) and iqr > 0:
            iqrs.append(iqr)
    if len(iqrs) < 2:
        return default
    minimum_iqr = min(iqrs)
    maximum_iqr = max(iqrs)
    if minimum_iqr <= 0:
        return default
    ratio = maximum_iqr / minimum_iqr
    score = piecewise_score(
        ratio,
        [
            (1.00, 0),
            (1.15, 0),
            (1.30, 10),
            (1.50, 20),
            (2.00, 40),
            (3.00, 65),
            (4.00, 80),
            (6.00, 90),
            (8.00, 95),
        ],
    )
    return {
        "score": clamp(score),
        "dispersion_ratio": ratio,
        "minimum_iqr": minimum_iqr,
        "maximum_iqr": maximum_iqr,
    }


def distribution_signal_score(
    df: pd.DataFrame, x: str, y: str, pair: _Pair | None = None
) -> dict[str, Any]:
    pair = _pair(df, x, y, pair)
    eta_score = eta_squared_score(df, x, y, pair)
    median_score = median_separation_score(df, x, y, pair)
    location_signal = 0.55 * eta_score + 0.45 * median_score

    outlier_stats = grouped_outlier_statistics(df, x, y, pair)
    dispersion_stats = dispersion_difference_score(df, x, y, pair)

    group_count = len(pair.sizes)

    if group_count <= 4:
        comparison_reliability = 1.0
    else:
        comparison_reliability = max(0.40, math.sqrt(4 / group_count))

    # Bonferroni across the three gated components (amendment 2).
    location_p = min(1.0, GATED_COMPONENTS * location_p_value(df, x, y, pair))
    dispersion_p = min(1.0, GATED_COMPONENTS * dispersion_p_value(df, x, y, pair))
    ungated_location = location_signal
    location_signal = location_signal * significance_reliability(location_p)

    dispersion_difference_signal = (
        dispersion_stats["score"]
        * comparison_reliability
        * significance_reliability(dispersion_p)
    )

    # The outlier permutation test is the expensive one. Its gated value can
    # never exceed the ungated one, so when that is 0, or no larger than the
    # stronger of the other two gated signals, the test cannot change the
    # final signal (a max of the three) and is skipped: outlier_p is NaN and
    # the outlier component is reported as 0. Exact: no score changes.
    ungated_outlier = outlier_stats["score"] * comparison_reliability
    outlier_test_needed = ungated_outlier > 0 and ungated_outlier > max(
        location_signal, dispersion_difference_signal
    )

    if outlier_test_needed:
        outlier_p = min(1.0, GATED_COMPONENTS * outlier_p_value(df, x, y, pair))
        outlier_difference_signal = ungated_outlier * significance_reliability(
            outlier_p
        )
    else:
        outlier_p = float("nan")
        outlier_difference_signal = 0.0

    signal = max(
        location_signal, outlier_difference_signal, dispersion_difference_signal
    )

    return {
        "signal": signal,
        "location_score": location_signal,
        "eta_squared_score": eta_score,
        "median_separation_score": median_score,
        "outlier_score": outlier_difference_signal,
        "outlier_rate": outlier_stats["overall_rate"],
        "outlier_rate_disparity": outlier_stats["rate_disparity"],
        "minimum_group_outlier_rate": outlier_stats["minimum_rate"],
        "maximum_group_outlier_rate": outlier_stats["maximum_rate"],
        "dispersion_score": dispersion_difference_signal,
        "dispersion_ratio": dispersion_stats["dispersion_ratio"],
        "ungated_location_score": ungated_location,
        "location_p": location_p,
        "dispersion_p": dispersion_p,
        "outlier_p": outlier_p,
    }


def adjust_signal_for_support(signal: float, support: float) -> float:
    reliability = (support / 100) ** 0.5
    return signal * reliability


def combine_quality_and_signal(
    semantic_fit: float,
    readability: float,
    quality: float,
    support: float,
    signal: float,
) -> tuple[float, float]:
    visualization_quality = (
        0.30 * semantic_fit + 0.20 * readability + 0.20 * quality + 0.30 * support
    )
    signal_multiplier = 0.35 + 0.65 * (signal / 100)
    final_score = visualization_quality * signal_multiplier
    return final_score, visualization_quality


def score_box_chart(
    df: pd.DataFrame, candidate: Candidate, profile: DatasetProfile
) -> ScoreResult:
    x = candidate["x"]
    y = candidate["y"]
    x_type = get_semantic_type(profile, x)
    y_type = get_semantic_type(profile, y)
    semantic_fit = semantic_fit_score(x_type, y_type)
    pair = _Pair(df, x, y)
    clean = pair.clean
    counts = pd.Series(pair.sizes)
    unique_count = len(pair.sizes)
    readability = category_readability_score(unique_count)
    support = sample_support_score(counts)
    quality = data_quality_score(df, x, y, pair)
    distribution = distribution_signal_score(df, x, y, pair)
    raw_signal = distribution["signal"]
    signal = adjust_signal_for_support(raw_signal, support)
    final_score, visualization_quality = combine_quality_and_signal(
        semantic_fit, readability, quality, support, signal
    )
    reasons = [
        f"X semantic type: {x_type}.",
        f"Y semantic type: {y_type}.",
        f"Semantic fit for box plot: {semantic_fit:.1f}/100.",
        f"{unique_count} groups detected.",
        f"{len(clean)} complete observations.",
        f"Average group size: {counts.mean():.1f}.",
        f"Smallest group size: {counts.min()}.",
        f"Data completeness: {quality:.1f}%.",
        f"Location difference signal: {distribution['location_score']:.1f}/100.",
        f"Dispersion difference signal: {distribution['dispersion_score']:.1f}/100.",
        f"Outlier-pattern difference signal: {distribution['outlier_score']:.1f}/100.",
        f"Raw distribution signal: {raw_signal:.1f}/100.",
        f"Sample-adjusted signal: {signal:.1f}/100.",
    ]
    return {
        "score": round(final_score, 2),
        "components": {
            "semantic_fit": round(semantic_fit, 2),
            "readability": round(readability, 2),
            "sample_support": round(support, 2),
            "data_quality": round(quality, 2),
            "signal": round(signal, 2),
            "visualization_quality": round(visualization_quality, 2),
        },
        "statistics": {
            "groups": int(unique_count),
            "observations": int(len(clean)),
            "average_group_size": round(float(counts.mean()), 2),
            "minimum_group_size": int(counts.min()),
            "location_score": round(float(distribution["location_score"]), 2),
            "eta_squared_score": round(float(distribution["eta_squared_score"]), 2),
            "median_separation_score": round(
                float(distribution["median_separation_score"]), 2
            ),
            "dispersion_score": round(float(distribution["dispersion_score"]), 2),
            "dispersion_ratio": round(float(distribution["dispersion_ratio"]), 4),
            "outlier_score": round(float(distribution["outlier_score"]), 2),
            "outlier_rate": round(float(distribution["outlier_rate"]), 4),
            "outlier_rate_disparity": round(
                float(distribution["outlier_rate_disparity"]), 4
            ),
            "maximum_group_outlier_rate": round(
                float(distribution["maximum_group_outlier_rate"]), 4
            ),
            "raw_signal": round(float(raw_signal), 2),
        },
        "reasons": reasons,
    }
