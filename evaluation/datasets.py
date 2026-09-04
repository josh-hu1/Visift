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
    Describes one visualization that should be
    considered a correct recommendation.
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
    """

    name: str
    relationship_type: str
    description: str
    dataframe: pd.DataFrame
    expected: list[ExpectedVisualization]
    is_null: bool = False


# ==================================================
# COMMON HELPERS
# ==================================================

def add_common_distractors(
    df,
    rng,
):
    """
    Add unrelated variables so that recovering the
    planted relationship is not completely trivial.
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
        ["North", "South", "East", "West"],
        size=n
    )

    df["record_id"] = np.arange(
        1,
        n + 1
    )

    return df


# ==================================================
# LINEAR RELATIONSHIPS
# ==================================================

def strong_linear_scenario(
    seed,
    n=500,
):
    """
    Strong numeric-to-numeric linear relationship.
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

    df = add_common_distractors(
        df,
        rng
    )

    return BenchmarkScenario(
        name=f"strong_linear_seed_{seed}",
        relationship_type="strong_linear",
        description=(
            "Strong positive linear relationship "
            "between feature_x and target_y."
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


def weak_linear_scenario(
    seed,
    n=500,
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


# ==================================================
# NONLINEAR RELATIONSHIP
# ==================================================

def quadratic_scenario(
    seed,
    n=500,
):
    """
    Strong U-shaped relationship.

    Pearson and Spearman correlation can both be
    close to zero even though the relationship is
    highly structured.

    This scenario intentionally exposes a current
    weakness of correlation-based scatter scoring.
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
        name=f"quadratic_seed_{seed}",
        relationship_type="nonlinear_quadratic",
        description=(
            "Strong quadratic relationship that "
            "Pearson/Spearman correlation may miss."
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
    n=500,
):
    """
    Strong categorical group effect.
    """

    rng = np.random.default_rng(
        seed
    )

    groups = rng.choice(
        ["A", "B", "C", "D"],
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


# ==================================================
# TEMPORAL RELATIONSHIP
# ==================================================

def temporal_trend_scenario(
    seed,
    n=500,
):
    """
    Numeric measure with a clear upward trend over time.
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


# ==================================================
# DISTRIBUTION SCENARIOS
# ==================================================

def skewed_distribution_scenario(
    seed,
    n=500,
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
    n=500,
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
        int(n * 0.06)
    )

    indices = rng.choice(
        n,
        size=outlier_count,
        replace=False
    )

    values[indices] += rng.choice(
        [-1, 1],
        size=outlier_count
    ) * rng.uniform(
        40,
        70,
        size=outlier_count
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
# CATEGORICAL COUNT DISTRIBUTION
# ==================================================

def imbalanced_category_scenario(
    seed,
    n=500,
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
# NULL / PURE NOISE
# ==================================================

def null_noise_scenario(
    seed,
    n=500,
):
    """
    Dataset intentionally containing no planted
    relationships.

    Useful for testing false-positive recommendation
    behavior.
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
            ["A", "B", "C", "D"],
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
        name=f"null_noise_seed_{seed}",
        relationship_type="null_noise",
        description=(
            "Independent noise variables with no "
            "intentionally planted relationship."
        ),
        dataframe=df,
        expected=[],
        is_null=True
    )


# ==================================================
# BENCHMARK SUITE
# ==================================================

def build_benchmark_suite(
    seeds=None,
    n=500,
):
    """
    Generate the full benchmark suite.

    Five seeds × nine scenario types = 45 datasets.
    """

    if seeds is None:
        seeds = range(5)

    scenarios = []

    for seed in seeds:

        scenarios.extend([
            strong_linear_scenario(
                seed,
                n
            ),
            weak_linear_scenario(
                seed,
                n
            ),
            quadratic_scenario(
                seed,
                n
            ),
            categorical_effect_scenario(
                seed,
                n
            ),
            temporal_trend_scenario(
                seed,
                n
            ),
            skewed_distribution_scenario(
                seed,
                n
            ),
            outlier_distribution_scenario(
                seed,
                n
            ),
            imbalanced_category_scenario(
                seed,
                n
            ),
            null_noise_scenario(
                seed,
                n
            )
        ])

    return scenarios