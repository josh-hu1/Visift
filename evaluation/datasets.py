from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ==================================================
# DATA STRUCTURES
# ==================================================

@dataclass(frozen=True)
class ExpectedVisualization:
    """
    Describes one visualization that should count
    as a correct recommendation.
    """

    chart: str
    x: str
    y: Optional[str] = None
    aggregation: Optional[str] = None
    unordered: bool = False


@dataclass
class BenchmarkScenario:
    """
    One synthetic benchmark dataset with known
    ground-truth behavior.

    null_chart_types:
        For null scenarios, optionally restrict
        false-positive evaluation to specific chart
        types.

        Example:
            Two independent skewed numeric variables
            have interesting histograms, but their
            scatterplot should not be considered a
            meaningful relationship.
    """

    name: str
    relationship_type: str
    description: str
    dataframe: pd.DataFrame
    expected: list[ExpectedVisualization]
    is_null: bool = False
    null_chart_types: Optional[tuple[str, ...]] = None


# ==================================================
# COMMON HELPERS
# ==================================================

def add_common_distractors(
    df,
    rng
):
    """
    Add unrelated variables so recovering the planted
    relationship is not completely trivial.
    """

    n = len(df)

    df = df.copy()

    df["noise_numeric_1"] = rng.normal(
        0,
        1,
        n
    )

    df["noise_numeric_2"] = rng.normal(
        0,
        1,
        n
    )

    df["noise_category"] = rng.choice(
        [
            "North",
            "South",
            "East",
            "West"
        ],
        size=n
    )

    df["record_id"] = np.arange(
        1,
        n + 1
    )

    return df


def apply_missingness(
    df,
    column,
    proportion,
    rng
):
    """
    Randomly replace a proportion of one column
    with missing values.
    """

    df = df.copy()

    missing_count = int(
        len(df) * proportion
    )

    if missing_count <= 0:
        return df

    indices = rng.choice(
        df.index,
        size=missing_count,
        replace=False
    )

    df.loc[
        indices,
        column
    ] = np.nan

    return df


# ==================================================
# LINEAR RELATIONSHIPS
# ==================================================

def linear_scenario(
    seed,
    n,
    noise_std,
    relationship_type,
    name_prefix
):
    """
    General strong linear scenario used for
    sample-size robustness testing.
    """

    rng = np.random.default_rng(
        seed
    )

    x = rng.normal(
        0,
        1,
        n
    )

    y = (
        3.0 * x
        + rng.normal(
            0,
            noise_std,
            n
        )
    )

    df = pd.DataFrame({
        "feature_x": x,
        "target_y": y
    })

    df = add_common_distractors(
        df,
        rng
    )

    return BenchmarkScenario(
        name=f"{name_prefix}_seed_{seed}",
        relationship_type=relationship_type,
        description=(
            f"Linear relationship with "
            f"{n} observations."
        ),
        dataframe=df,
        expected=[
            ExpectedVisualization(
                chart="scatter",
                x="feature_x",
                y="target_y",
                unordered=True
            )
        ]
    )


def strong_linear_scenario(
    seed,
    n=500
):
    return linear_scenario(
        seed=seed,
        n=n,
        noise_std=0.8,
        relationship_type="strong_linear_n500",
        name_prefix="strong_linear_n500"
    )


def strong_linear_n30_scenario(
    seed
):
    return linear_scenario(
        seed=seed,
        n=30,
        noise_std=0.8,
        relationship_type="strong_linear_n30",
        name_prefix="strong_linear_n30"
    )


def strong_linear_n75_scenario(
    seed
):
    return linear_scenario(
        seed=seed,
        n=75,
        noise_std=0.8,
        relationship_type="strong_linear_n75",
        name_prefix="strong_linear_n75"
    )


def strong_linear_n150_scenario(
    seed
):
    return linear_scenario(
        seed=seed,
        n=150,
        noise_std=0.8,
        relationship_type="strong_linear_n150",
        name_prefix="strong_linear_n150"
    )


def weak_linear_scenario(
    seed,
    n=500
):
    """
    Weak-to-moderate numeric relationship.
    """

    rng = np.random.default_rng(
        seed
    )

    x = rng.normal(
        0,
        1,
        n
    )

    y = (
        0.45 * x
        + rng.normal(
            0,
            1,
            n
        )
    )

    df = pd.DataFrame({
        "feature_x": x,
        "target_y": y
    })

    df = add_common_distractors(
        df,
        rng
    )

    return BenchmarkScenario(
        name=f"weak_linear_seed_{seed}",
        relationship_type="weak_linear",
        description=(
            "Weak-to-moderate positive linear "
            "relationship."
        ),
        dataframe=df,
        expected=[
            ExpectedVisualization(
                chart="scatter",
                x="feature_x",
                y="target_y",
                unordered=True
            )
        ]
    )


def missing_linear_scenario(
    seed,
    n=500
):
    """
    Strong linear relationship where 40% of the
    target observations are missing.
    """

    rng = np.random.default_rng(
        seed
    )

    x = rng.normal(
        0,
        1,
        n
    )

    y = (
        3.0 * x
        + rng.normal(
            0,
            0.8,
            n
        )
    )

    df = pd.DataFrame({
        "feature_x": x,
        "target_y": y
    })

    df = apply_missingness(
        df,
        "target_y",
        0.40,
        rng
    )

    df = add_common_distractors(
        df,
        rng
    )

    return BenchmarkScenario(
        name=f"linear_40pct_missing_seed_{seed}",
        relationship_type="linear_40pct_missing",
        description=(
            "Strong linear relationship with "
            "40% missing target values."
        ),
        dataframe=df,
        expected=[
            ExpectedVisualization(
                chart="scatter",
                x="feature_x",
                y="target_y",
                unordered=True
            )
        ]
    )


# ==================================================
# NONLINEAR RELATIONSHIPS
# ==================================================

def quadratic_scenario(
    seed,
    n=500
):
    """
    Strong U-shaped relationship.
    """

    rng = np.random.default_rng(
        seed
    )

    x = rng.uniform(
        -3,
        3,
        n
    )

    y = (
        x ** 2
        + rng.normal(
            0,
            0.6,
            n
        )
    )

    df = pd.DataFrame({
        "feature_x": x,
        "target_y": y
    })

    df = add_common_distractors(
        df,
        rng
    )

    return BenchmarkScenario(
        name=f"quadratic_strong_seed_{seed}",
        relationship_type="nonlinear_quadratic_strong",
        description=(
            "Strong quadratic relationship."
        ),
        dataframe=df,
        expected=[
            ExpectedVisualization(
                chart="scatter",
                x="feature_x",
                y="target_y",
                unordered=True
            )
        ]
    )


def noisy_quadratic_scenario(
    seed,
    n=500
):
    """
    Harder quadratic relationship with much
    more observational noise.
    """

    rng = np.random.default_rng(
        seed
    )

    x = rng.uniform(
        -3,
        3,
        n
    )

    y = (
        x ** 2
        + rng.normal(
            0,
            2.2,
            n
        )
    )

    df = pd.DataFrame({
        "feature_x": x,
        "target_y": y
    })

    df = add_common_distractors(
        df,
        rng
    )

    return BenchmarkScenario(
        name=f"quadratic_noisy_seed_{seed}",
        relationship_type="nonlinear_quadratic_noisy",
        description=(
            "Moderately noisy quadratic relationship."
        ),
        dataframe=df,
        expected=[
            ExpectedVisualization(
                chart="scatter",
                x="feature_x",
                y="target_y",
                unordered=True
            )
        ]
    )


def sinusoidal_scenario(
    seed,
    n=500
):
    """
    Strong oscillating nonlinear relationship.

    Across multiple cycles Pearson and Spearman
    correlation can both be relatively small.
    """

    rng = np.random.default_rng(
        seed
    )

    x = rng.uniform(
        0,
        4 * np.pi,
        n
    )

    y = (
        np.sin(x)
        + rng.normal(
            0,
            0.20,
            n
        )
    )

    df = pd.DataFrame({
        "feature_x": x,
        "target_y": y
    })

    df = add_common_distractors(
        df,
        rng
    )

    return BenchmarkScenario(
        name=f"sinusoidal_seed_{seed}",
        relationship_type="nonlinear_sinusoidal",
        description=(
            "Strong sinusoidal nonlinear relationship."
        ),
        dataframe=df,
        expected=[
            ExpectedVisualization(
                chart="scatter",
                x="feature_x",
                y="target_y",
                unordered=True
            )
        ]
    )


def threshold_scenario(
    seed,
    n=500
):
    """
    Symmetric threshold relationship.

    The target increases when |x| becomes large,
    creating dependence with little overall monotonic
    association.
    """

    rng = np.random.default_rng(
        seed
    )

    x = rng.uniform(
        -3,
        3,
        n
    )

    y = np.where(
        np.abs(x) >= 1.5,
        4.0,
        0.0
    )

    y = (
        y
        + rng.normal(
            0,
            0.7,
            n
        )
    )

    df = pd.DataFrame({
        "feature_x": x,
        "target_y": y
    })

    df = add_common_distractors(
        df,
        rng
    )

    return BenchmarkScenario(
        name=f"symmetric_threshold_seed_{seed}",
        relationship_type="nonlinear_threshold",
        description=(
            "Symmetric threshold dependence where "
            "large absolute x values produce a level shift."
        ),
        dataframe=df,
        expected=[
            ExpectedVisualization(
                chart="scatter",
                x="feature_x",
                y="target_y",
                unordered=True
            )
        ]
    )


# ==================================================
# CATEGORICAL RELATIONSHIP
# ==================================================

def categorical_effect_scenario(
    seed,
    n=500
):
    """
    Strong categorical group effect.
    """

    rng = np.random.default_rng(
        seed
    )

    groups = rng.choice(
        [
            "A",
            "B",
            "C",
            "D"
        ],
        size=n
    )

    group_means = {
        "A": 10,
        "B": 15,
        "C": 20,
        "D": 25
    }

    values = np.array([
        group_means[group]
        for group in groups
    ])

    values = (
        values
        + rng.normal(
            0,
            2.5,
            n
        )
    )

    df = pd.DataFrame({
        "group": groups,
        "outcome": values
    })

    df = add_common_distractors(
        df,
        rng
    )

    return BenchmarkScenario(
        name=f"categorical_effect_seed_{seed}",
        relationship_type="categorical_effect",
        description=(
            "Strong difference in numeric outcome "
            "across four categories."
        ),
        dataframe=df,
        expected=[
            ExpectedVisualization(
                chart="box",
                x="group",
                y="outcome"
            ),
            ExpectedVisualization(
                chart="bar",
                x="group",
                y="outcome",
                aggregation="mean"
            )
        ]
    )


def imbalanced_category_scenario(
    seed,
    n=500
):
    """
    Strongly imbalanced categorical frequency
    distribution.
    """

    rng = np.random.default_rng(
        seed
    )

    segments = rng.choice(
        [
            "Segment A",
            "Segment B",
            "Segment C",
            "Segment D"
        ],
        size=n,
        p=[
            0.70,
            0.20,
            0.08,
            0.02
        ]
    )

    df = pd.DataFrame({
        "segment": segments
    })

    df = add_common_distractors(
        df,
        rng
    )

    return BenchmarkScenario(
        name=f"imbalanced_category_seed_{seed}",
        relationship_type="category_imbalance",
        description=(
            "Strongly imbalanced categorical "
            "frequency distribution."
        ),
        dataframe=df,
        expected=[
            ExpectedVisualization(
                chart="bar",
                x="segment",
                aggregation="count"
            )
        ]
    )


# ==================================================
# TEMPORAL RELATIONSHIPS
# ==================================================

def temporal_trend_scenario(
    seed,
    n=500
):
    """
    Numeric measure with a clear upward trend.
    """

    rng = np.random.default_rng(
        seed
    )

    dates = pd.date_range(
        "2024-01-01",
        periods=n,
        freq="D"
    )

    trend = np.linspace(
        0,
        30,
        n
    )

    values = (
        100
        + trend
        + rng.normal(
            0,
            5,
            n
        )
    )

    df = pd.DataFrame({
        "date": dates,
        "metric": values
    })

    df = add_common_distractors(
        df,
        rng
    )

    return BenchmarkScenario(
        name=f"temporal_trend_seed_{seed}",
        relationship_type="temporal_trend",
        description=(
            "Numeric variable with a clear upward "
            "trend over time."
        ),
        dataframe=df,
        expected=[
            ExpectedVisualization(
                chart="line",
                x="date",
                y="metric"
            )
        ]
    )


def seasonal_scenario(
    seed,
    n=730
):
    """
    Strong periodic behavior with almost no long-run
    monotonic trend.

    This is designed to test whether the line scorer
    can recognize seasonality rather than only trend.
    """

    rng = np.random.default_rng(
        seed
    )

    dates = pd.date_range(
        "2024-01-01",
        periods=n,
        freq="D"
    )

    t = np.arange(
        n
    )

    values = (
        100
        + 18 * np.sin(
            2
            * np.pi
            * t
            / 30
        )
        + rng.normal(
            0,
            3,
            n
        )
    )

    df = pd.DataFrame({
        "date": dates,
        "metric": values
    })

    df = add_common_distractors(
        df,
        rng
    )

    return BenchmarkScenario(
        name=f"temporal_seasonality_seed_{seed}",
        relationship_type="temporal_seasonality",
        description=(
            "Strong repeating 30-day seasonal pattern "
            "without a long-term trend."
        ),
        dataframe=df,
        expected=[
            ExpectedVisualization(
                chart="line",
                x="date",
                y="metric"
            )
        ]
    )


def change_point_scenario(
    seed,
    n=500
):
    """
    Abrupt persistent level shift halfway through
    the time series.
    """

    rng = np.random.default_rng(
        seed
    )

    dates = pd.date_range(
        "2024-01-01",
        periods=n,
        freq="D"
    )

    values = np.where(
        np.arange(n) < n // 2,
        100.0,
        130.0
    )

    values = (
        values
        + rng.normal(
            0,
            4,
            n
        )
    )

    df = pd.DataFrame({
        "date": dates,
        "metric": values
    })

    df = add_common_distractors(
        df,
        rng
    )

    return BenchmarkScenario(
        name=f"temporal_change_point_seed_{seed}",
        relationship_type="temporal_change_point",
        description=(
            "Abrupt persistent level shift halfway "
            "through the time series."
        ),
        dataframe=df,
        expected=[
            ExpectedVisualization(
                chart="line",
                x="date",
                y="metric"
            )
        ]
    )


def trend_and_seasonality_scenario(
    seed,
    n=730
):
    """
    Time series containing both a long-term upward
    trend and periodic seasonality.
    """

    rng = np.random.default_rng(
        seed
    )

    dates = pd.date_range(
        "2024-01-01",
        periods=n,
        freq="D"
    )

    t = np.arange(
        n
    )

    trend = (
        0.05 * t
    )

    seasonal = (
        12
        * np.sin(
            2
            * np.pi
            * t
            / 30
        )
    )

    values = (
        100
        + trend
        + seasonal
        + rng.normal(
            0,
            4,
            n
        )
    )

    df = pd.DataFrame({
        "date": dates,
        "metric": values
    })

    df = add_common_distractors(
        df,
        rng
    )

    return BenchmarkScenario(
        name=f"trend_and_seasonality_seed_{seed}",
        relationship_type="temporal_trend_seasonality",
        description=(
            "Long-term trend combined with a repeating "
            "seasonal pattern."
        ),
        dataframe=df,
        expected=[
            ExpectedVisualization(
                chart="line",
                x="date",
                y="metric"
            )
        ]
    )


def volatility_shift_scenario(
    seed,
    n=500
):
    """
    Time series whose mean remains stable while
    variance increases substantially halfway through.

    This is intentionally difficult for a scorer based
    primarily on correlation with time.
    """

    rng = np.random.default_rng(
        seed
    )

    dates = pd.date_range(
        "2024-01-01",
        periods=n,
        freq="D"
    )

    first_half = rng.normal(
        100,
        2,
        n // 2
    )

    second_half = rng.normal(
        100,
        14,
        n - n // 2
    )

    values = np.concatenate([
        first_half,
        second_half
    ])

    df = pd.DataFrame({
        "date": dates,
        "metric": values
    })

    df = add_common_distractors(
        df,
        rng
    )

    return BenchmarkScenario(
        name=f"temporal_volatility_shift_seed_{seed}",
        relationship_type="temporal_volatility_shift",
        description=(
            "Stable mean with a large increase in "
            "variance halfway through time."
        ),
        dataframe=df,
        expected=[
            ExpectedVisualization(
                chart="line",
                x="date",
                y="metric"
            )
        ]
    )


# ==================================================
# DISTRIBUTION SCENARIOS
# ==================================================

def skewed_distribution_scenario(
    seed,
    n=500
):
    """
    Strong right-skewed numeric distribution.
    """

    rng = np.random.default_rng(
        seed
    )

    values = rng.lognormal(
        mean=2.0,
        sigma=1.1,
        size=n
    )

    df = pd.DataFrame({
        "skewed_measure": values
    })

    df = add_common_distractors(
        df,
        rng
    )

    return BenchmarkScenario(
        name=f"skewed_distribution_seed_{seed}",
        relationship_type="skewed_distribution",
        description=(
            "Strongly right-skewed numeric "
            "distribution."
        ),
        dataframe=df,
        expected=[
            ExpectedVisualization(
                chart="histogram",
                x="skewed_measure"
            )
        ]
    )


def outlier_distribution_scenario(
    seed,
    n=500
):
    """
    Mostly normal distribution with planted
    extreme outliers.
    """

    rng = np.random.default_rng(
        seed
    )

    values = rng.normal(
        50,
        8,
        n
    )

    outlier_count = max(
        1,
        int(
            n * 0.06
        )
    )

    indices = rng.choice(
        n,
        size=outlier_count,
        replace=False
    )

    values[
        indices
    ] += (
        rng.choice(
            [-1, 1],
            size=outlier_count
        )
        * rng.uniform(
            40,
            70,
            size=outlier_count
        )
    )

    df = pd.DataFrame({
        "outlier_measure": values
    })

    df = add_common_distractors(
        df,
        rng
    )

    return BenchmarkScenario(
        name=f"outlier_distribution_seed_{seed}",
        relationship_type="outlier_distribution",
        description=(
            "Numeric distribution containing "
            "approximately 6% extreme outliers."
        ),
        dataframe=df,
        expected=[
            ExpectedVisualization(
                chart="histogram",
                x="outlier_measure"
            )
        ]
    )


# ==================================================
# NULL SCENARIOS
# ==================================================

def null_noise_scenario(
    seed,
    n=500
):
    """
    General pure-noise dataset.

    All generated visualization types are evaluated
    for false-positive behavior.
    """

    rng = np.random.default_rng(
        seed
    )

    dates = pd.date_range(
        "2024-01-01",
        periods=n,
        freq="D"
    )

    df = pd.DataFrame({
        "noise_a": rng.normal(
            0,
            1,
            n
        ),
        "noise_b": rng.normal(
            0,
            1,
            n
        ),
        "noise_c": rng.normal(
            0,
            1,
            n
        ),
        "balanced_category": rng.choice(
            [
                "A",
                "B",
                "C",
                "D"
            ],
            size=n
        ),
        "date": dates,
        "random_over_time": rng.normal(
            0,
            1,
            n
        )
    })

    return BenchmarkScenario(
        name=f"null_all_noise_seed_{seed}",
        relationship_type="null_all_noise",
        description=(
            "Independent approximately Gaussian noise "
            "with no planted relationships."
        ),
        dataframe=df,
        expected=[],
        is_null=True
    )


def null_skewed_scatter_scenario(
    seed,
    n=500
):
    """
    Two independent but strongly skewed variables.

    Their histograms are legitimately interesting,
    so only scatterplots are evaluated as null
    relationship candidates.
    """

    rng = np.random.default_rng(
        seed
    )

    df = pd.DataFrame({
        "skewed_x": rng.lognormal(
            1.5,
            1.0,
            n
        ),
        "skewed_y": rng.lognormal(
            2.0,
            1.2,
            n
        )
    })

    return BenchmarkScenario(
        name=f"null_skewed_scatter_seed_{seed}",
        relationship_type="null_scatter_skewed",
        description=(
            "Independent skewed numeric variables."
        ),
        dataframe=df,
        expected=[],
        is_null=True,
        null_chart_types=(
            "scatter",
        )
    )


def null_outlier_scatter_scenario(
    seed,
    n=500
):
    """
    Independent variables containing extreme
    observations.

    Tests whether outliers create spurious dependence.
    """

    rng = np.random.default_rng(
        seed
    )

    x = rng.normal(
        0,
        1,
        n
    )

    y = rng.normal(
        0,
        1,
        n
    )

    outlier_count = int(
        n * 0.06
    )

    x_indices = rng.choice(
        n,
        size=outlier_count,
        replace=False
    )

    y_indices = rng.choice(
        n,
        size=outlier_count,
        replace=False
    )

    x[
        x_indices
    ] += rng.normal(
        0,
        12,
        outlier_count
    )

    y[
        y_indices
    ] += rng.normal(
        0,
        12,
        outlier_count
    )

    df = pd.DataFrame({
        "outlier_x": x,
        "outlier_y": y
    })

    return BenchmarkScenario(
        name=f"null_outlier_scatter_seed_{seed}",
        relationship_type="null_scatter_outliers",
        description=(
            "Independent numeric variables with "
            "extreme outliers."
        ),
        dataframe=df,
        expected=[],
        is_null=True,
        null_chart_types=(
            "scatter",
        )
    )


def null_small_sample_scatter_scenario(
    seed,
    n=30
):
    """
    Independent variables with a small sample.

    Tests chance correlations and MI instability.
    """

    rng = np.random.default_rng(
        seed
    )

    df = pd.DataFrame({
        "small_x": rng.normal(
            0,
            1,
            n
        ),
        "small_y": rng.normal(
            0,
            1,
            n
        ),
        "small_z": rng.normal(
            0,
            1,
            n
        )
    })

    return BenchmarkScenario(
        name=f"null_small_sample_seed_{seed}",
        relationship_type="null_scatter_small_sample",
        description=(
            "Independent numeric variables with only "
            "30 observations."
        ),
        dataframe=df,
        expected=[],
        is_null=True,
        null_chart_types=(
            "scatter",
        )
    )


def null_temporal_noise_scenario(
    seed,
    n=500
):
    """
    Pure white noise indexed by time.

    Only line charts are evaluated as null candidates.
    """

    rng = np.random.default_rng(
        seed
    )

    dates = pd.date_range(
        "2024-01-01",
        periods=n,
        freq="D"
    )

    df = pd.DataFrame({
        "date": dates,
        "random_metric": rng.normal(
            100,
            10,
            n
        )
    })

    return BenchmarkScenario(
        name=f"null_temporal_noise_seed_{seed}",
        relationship_type="null_line_white_noise",
        description=(
            "Independent white noise indexed by time."
        ),
        dataframe=df,
        expected=[],
        is_null=True,
        null_chart_types=(
            "line",
        )
    )


def null_categorical_effect_scenario(
    seed,
    n=500
):
    """
    Random groups with an independent numeric outcome.

    Only grouped bar and box plots are evaluated.
    """

    rng = np.random.default_rng(
        seed
    )

    groups = rng.choice(
        [
            "A",
            "B",
            "C",
            "D"
        ],
        size=n
    )

    outcome = rng.normal(
        50,
        10,
        n
    )

    df = pd.DataFrame({
        "group": groups,
        "outcome": outcome
    })

    return BenchmarkScenario(
        name=f"null_group_effect_seed_{seed}",
        relationship_type="null_group_random",
        description=(
            "Random categorical groups with an "
            "independent numeric outcome."
        ),
        dataframe=df,
        expected=[],
        is_null=True,
        null_chart_types=(
            "bar",
            "box"
        )
    )


# ==================================================
# BENCHMARK SUITE
# ==================================================

def build_benchmark_suite(
    seeds=None
):
    """
    Build the hardened benchmark.

    Per seed:
        18 positive scenarios
        6 null scenarios

    Five seeds:
        120 total datasets
    """

    if seeds is None:
        seeds = range(
            5
        )

    scenarios = []

    for seed in seeds:

        # -----------------------------------
        # Positive relationships
        # -----------------------------------

        scenarios.extend([
            strong_linear_scenario(
                seed
            ),
            strong_linear_n30_scenario(
                seed
            ),
            strong_linear_n75_scenario(
                seed
            ),
            strong_linear_n150_scenario(
                seed
            ),
            weak_linear_scenario(
                seed
            ),
            missing_linear_scenario(
                seed
            ),

            quadratic_scenario(
                seed
            ),
            noisy_quadratic_scenario(
                seed
            ),
            sinusoidal_scenario(
                seed
            ),
            threshold_scenario(
                seed
            ),

            categorical_effect_scenario(
                seed
            ),
            imbalanced_category_scenario(
                seed
            ),

            temporal_trend_scenario(
                seed
            ),
            seasonal_scenario(
                seed
            ),
            change_point_scenario(
                seed
            ),
            trend_and_seasonality_scenario(
                seed
            ),
            volatility_shift_scenario(
                seed
            ),

            skewed_distribution_scenario(
                seed
            ),
            outlier_distribution_scenario(
                seed
            )
        ])

        # -----------------------------------
        # Null scenarios
        # -----------------------------------

        scenarios.extend([
            null_noise_scenario(
                seed
            ),
            null_skewed_scatter_scenario(
                seed
            ),
            null_outlier_scatter_scenario(
                seed
            ),
            null_small_sample_scatter_scenario(
                seed
            ),
            null_temporal_noise_scenario(
                seed
            ),
            null_categorical_effect_scenario(
                seed
            )
        ])

    return scenarios