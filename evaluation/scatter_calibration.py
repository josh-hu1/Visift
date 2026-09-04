from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scoring.scatter import (
    relationship_signal_score,
    score_scatter_chart
)


# ==================================================
# CONFIGURATION
# ==================================================

RESULTS_DIR = Path(
    "evaluation/results/scatter_calibration_v1"
)

DEFAULT_SEEDS = range(
    25
)

DEFAULT_SAMPLE_SIZE = 1000

STRENGTH_LABELS = {
    0: "ordinary",
    1: "weak",
    2: "moderate",
    3: "strong",
    4: "extreme"
}

# Latent signal strengths used by the controlled
# relationship families. These are not intended to
# be interpreted as exact Pearson correlations for
# nonlinear families; they control signal-to-noise
# strength in a consistent way.
STRENGTH_VALUES = {
    0: 0.00,
    1: 0.15,
    2: 0.30,
    3: 0.55,
    4: 0.85
}

SAMPLE_SIZES = [
    20,
    30,
    75,
    150,
    500
]

MISSINGNESS_LEVELS = [
    0.00,
    0.20,
    0.40,
    0.60,
    0.80
]


# ==================================================
# DATA STRUCTURES
# ==================================================

@dataclass(frozen=True)
class ScatterCalibrationScenario:
    """
    One controlled scatter calibration scenario.

    The hardened benchmark tests retrieval and
    false-positive behavior across the complete
    Visift pipeline.

    This file instead isolates scatter scoring so
    we can inspect whether recommendation scores
    behave sensibly as relationship strength,
    sample size, and missingness change.
    """

    name: str
    family: str
    level: str
    strength_index: int
    seed: int
    description: str
    dataframe: pd.DataFrame
    x: str
    y: str


@dataclass(frozen=True)
class ScatterRobustnessScenario:
    """
    One scatter robustness scenario used for
    sample-size, missingness, or null testing.
    """

    name: str
    family: str
    setting: str
    setting_value: float
    seed: int
    description: str
    dataframe: pd.DataFrame
    x: str
    y: str


# ==================================================
# COMMON HELPERS
# ==================================================

def manual_scatter_profile(
    x,
    y
):
    """
    Build the minimal profile required by the
    scatter scorer.

    Semantic inference is intentionally fixed here
    because this calibration benchmark isolates
    score behavior. The hardened benchmark already
    tests the full profiling and candidate pipeline.
    """

    return {
        "column_profiles": [
            {
                "name": x,
                "semantic_type":
                    "numeric_continuous"
            },
            {
                "name": y,
                "semantic_type":
                    "numeric_continuous"
            }
        ]
    }


def standardized(
    values
):
    """
    Standardize an array to mean 0 and standard
    deviation 1.
    """

    values = np.asarray(
        values,
        dtype=float
    )

    std = values.std()

    if std <= 0:
        return np.zeros_like(
            values
        )

    return (
        values
        - values.mean()
    ) / std


def blend_signal_and_noise(
    signal_component,
    strength,
    rng
):
    """
    Blend a standardized planted relationship with
    independent Gaussian noise.

    For strength in [0, 1]:

        y = strength * signal
            + sqrt(1-strength^2) * noise

    For the linear family this makes the population
    Pearson correlation approximately equal to the
    requested strength.

    For nonlinear families it provides a consistent
    signal-to-noise control while keeping the direct
    x-y correlation low where the planted function
    is symmetric.
    """

    signal_component = standardized(
        signal_component
    )

    noise = rng.normal(
        0,
        1,
        len(
            signal_component
        )
    )

    noise_scale = np.sqrt(
        max(
            0.0,
            1.0
            - strength ** 2
        )
    )

    return (
        strength
        * signal_component
        + noise_scale
        * noise
    )


def make_dataframe(
    x_values,
    y_values,
    x="feature_x",
    y="target_y"
):
    return pd.DataFrame({
        x: x_values,
        y: y_values
    })


def strength_label(
    strength_index
):
    return STRENGTH_LABELS[
        strength_index
    ]


# ==================================================
# LINEAR FAMILY
# ==================================================

def linear_values(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE
):
    """
    Generate a controlled linear relationship.

    Because x and noise are standard normal and
    independent, the requested strength is
    approximately the population Pearson
    correlation.
    """

    rng = np.random.default_rng(
        seed
    )

    strength = STRENGTH_VALUES[
        strength_index
    ]

    x = rng.normal(
        0,
        1,
        n
    )

    noise = rng.normal(
        0,
        1,
        n
    )

    y = (
        strength
        * x
        + np.sqrt(
            max(
                0.0,
                1.0
                - strength ** 2
            )
        )
        * noise
    )

    return (
        x,
        y
    )


def linear_scenario(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE
):
    level = strength_label(
        strength_index
    )

    x, y = linear_values(
        seed=seed,
        strength_index=
            strength_index,
        n=n
    )

    return ScatterCalibrationScenario(
        name=(
            f"scatter_linear_{level}_"
            f"seed_{seed}"
        ),
        family="linear",
        level=level,
        strength_index=strength_index,
        seed=seed,
        description=(
            "Controlled linear relationship "
            f"at {level} strength."
        ),
        dataframe=make_dataframe(
            x,
            y
        ),
        x="feature_x",
        y="target_y"
    )


# ==================================================
# QUADRATIC FAMILY
# ==================================================

def quadratic_values(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE
):
    """
    Generate a symmetric U-shaped relationship.

    Pearson and Spearman association with x should
    remain relatively small, so this family mainly
    calibrates the mutual-information path.
    """

    rng = np.random.default_rng(
        seed
    )

    strength = STRENGTH_VALUES[
        strength_index
    ]

    x = rng.uniform(
        -3,
        3,
        n
    )

    latent = (
        x ** 2
    )

    y = blend_signal_and_noise(
        latent,
        strength,
        rng
    )

    return (
        x,
        y
    )


def quadratic_scenario(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE
):
    level = strength_label(
        strength_index
    )

    x, y = quadratic_values(
        seed=seed,
        strength_index=
            strength_index,
        n=n
    )

    return ScatterCalibrationScenario(
        name=(
            f"scatter_quadratic_{level}_"
            f"seed_{seed}"
        ),
        family="quadratic",
        level=level,
        strength_index=strength_index,
        seed=seed,
        description=(
            "Controlled symmetric quadratic "
            f"relationship at {level} strength."
        ),
        dataframe=make_dataframe(
            x,
            y
        ),
        x="feature_x",
        y="target_y"
    )


# ==================================================
# PERIODIC FAMILY
# ==================================================

def periodic_values(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE
):
    """
    Generate a symmetric periodic nonlinear
    relationship.

    cos(2x) over a symmetric x range keeps direct
    linear association low while preserving a clear
    nonlinear pattern as signal strength increases.
    """

    rng = np.random.default_rng(
        seed
    )

    strength = STRENGTH_VALUES[
        strength_index
    ]

    x = rng.uniform(
        -np.pi,
        np.pi,
        n
    )

    latent = np.cos(
        2 * x
    )

    y = blend_signal_and_noise(
        latent,
        strength,
        rng
    )

    return (
        x,
        y
    )


def periodic_scenario(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE
):
    level = strength_label(
        strength_index
    )

    x, y = periodic_values(
        seed=seed,
        strength_index=
            strength_index,
        n=n
    )

    return ScatterCalibrationScenario(
        name=(
            f"scatter_periodic_{level}_"
            f"seed_{seed}"
        ),
        family="periodic",
        level=level,
        strength_index=strength_index,
        seed=seed,
        description=(
            "Controlled periodic nonlinear "
            f"relationship at {level} strength."
        ),
        dataframe=make_dataframe(
            x,
            y
        ),
        x="feature_x",
        y="target_y"
    )


# ==================================================
# THRESHOLD FAMILY
# ==================================================

def threshold_values(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE
):
    """
    Generate a symmetric threshold relationship.

    Large absolute x values belong to a different
    response regime. Symmetry keeps ordinary
    monotonic correlation relatively small.
    """

    rng = np.random.default_rng(
        seed
    )

    strength = STRENGTH_VALUES[
        strength_index
    ]

    x = rng.uniform(
        -3,
        3,
        n
    )

    latent = np.where(
        np.abs(
            x
        )
        >= 1.5,
        1.0,
        -1.0
    )

    y = blend_signal_and_noise(
        latent,
        strength,
        rng
    )

    return (
        x,
        y
    )


def threshold_scenario(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE
):
    level = strength_label(
        strength_index
    )

    x, y = threshold_values(
        seed=seed,
        strength_index=
            strength_index,
        n=n
    )

    return ScatterCalibrationScenario(
        name=(
            f"scatter_threshold_{level}_"
            f"seed_{seed}"
        ),
        family="threshold",
        level=level,
        strength_index=strength_index,
        seed=seed,
        description=(
            "Controlled symmetric threshold "
            f"relationship at {level} strength."
        ),
        dataframe=make_dataframe(
            x,
            y
        ),
        x="feature_x",
        y="target_y"
    )


# ==================================================
# STRENGTH CALIBRATION SUITE
# ==================================================

def build_strength_suite(
    seeds=None,
    n=DEFAULT_SAMPLE_SIZE
):
    """
    Build the main graduated scatter calibration
    suite.

    Default configuration:

        4 relationship families
        5 strength levels
        25 seeds
        500 total scenarios
    """

    if seeds is None:
        seeds = DEFAULT_SEEDS

    scenarios = []

    for seed in seeds:

        for strength_index in range(
            5
        ):

            scenarios.extend([
                linear_scenario(
                    seed=seed,
                    strength_index=
                        strength_index,
                    n=n
                ),
                quadratic_scenario(
                    seed=seed,
                    strength_index=
                        strength_index,
                    n=n
                ),
                periodic_scenario(
                    seed=seed,
                    strength_index=
                        strength_index,
                    n=n
                ),
                threshold_scenario(
                    seed=seed,
                    strength_index=
                        strength_index,
                    n=n
                )
            ])

    return scenarios


# ==================================================
# SAMPLE-SIZE RELIABILITY
# ==================================================

def sample_size_scenario(
    seed,
    n
):
    """
    Hold the planted linear population effect fixed
    while changing sample size.

    The scorer should generally become more
    confident as statistical support improves,
    without making strong small-sample relationships
    disappear entirely.
    """

    rng = np.random.default_rng(
        seed
    )

    target_strength = 0.50

    x = rng.normal(
        0,
        1,
        n
    )

    noise = rng.normal(
        0,
        1,
        n
    )

    y = (
        target_strength
        * x
        + np.sqrt(
            1
            - target_strength ** 2
        )
        * noise
    )

    return ScatterRobustnessScenario(
        name=(
            f"scatter_sample_size_n{n}_"
            f"seed_{seed}"
        ),
        family="sample_size",
        setting=f"n={n}",
        setting_value=float(
            n
        ),
        seed=seed,
        description=(
            "Fixed population linear effect "
            f"with n={n} observations."
        ),
        dataframe=make_dataframe(
            x,
            y
        ),
        x="feature_x",
        y="target_y"
    )


def build_sample_size_suite(
    seeds=None
):
    if seeds is None:
        seeds = DEFAULT_SEEDS

    scenarios = []

    for seed in seeds:

        for n in SAMPLE_SIZES:

            scenarios.append(
                sample_size_scenario(
                    seed=seed,
                    n=n
                )
            )

    return scenarios


# ==================================================
# MISSINGNESS ROBUSTNESS
# ==================================================

def missingness_scenario(
    seed,
    missingness,
    n=DEFAULT_SAMPLE_SIZE
):
    """
    Hold a strong linear relationship fixed while
    progressively removing paired target values.

    This tests whether score degradation from
    completeness and sample support is gradual and
    understandable.
    """

    rng = np.random.default_rng(
        seed
    )

    target_strength = 0.70

    x = rng.normal(
        0,
        1,
        n
    )

    noise = rng.normal(
        0,
        1,
        n
    )

    y = (
        target_strength
        * x
        + np.sqrt(
            1
            - target_strength ** 2
        )
        * noise
    )

    df = make_dataframe(
        x,
        y
    )

    missing_count = int(
        n * missingness
    )

    if missing_count > 0:

        indices = rng.choice(
            df.index,
            size=missing_count,
            replace=False
        )

        df.loc[
            indices,
            "target_y"
        ] = np.nan

    return ScatterRobustnessScenario(
        name=(
            "scatter_missingness_"
            f"{int(missingness * 100)}pct_"
            f"seed_{seed}"
        ),
        family="missingness",
        setting=(
            f"{int(missingness * 100)}%"
        ),
        setting_value=float(
            missingness
        ),
        seed=seed,
        description=(
            "Fixed strong linear relationship "
            f"with {missingness * 100:.0f}% "
            "missing target values."
        ),
        dataframe=df,
        x="feature_x",
        y="target_y"
    )


def build_missingness_suite(
    seeds=None,
    n=DEFAULT_SAMPLE_SIZE
):
    if seeds is None:
        seeds = DEFAULT_SEEDS

    scenarios = []

    for seed in seeds:

        for missingness in (
            MISSINGNESS_LEVELS
        ):

            scenarios.append(
                missingness_scenario(
                    seed=seed,
                    missingness=
                        missingness,
                    n=n
                )
            )

    return scenarios


# ==================================================
# NULL ROBUSTNESS
# ==================================================

def null_normal_scenario(
    seed,
    n=DEFAULT_SAMPLE_SIZE
):
    rng = np.random.default_rng(
        seed
    )

    df = make_dataframe(
        rng.normal(
            0,
            1,
            n
        ),
        rng.normal(
            0,
            1,
            n
        )
    )

    return ScatterRobustnessScenario(
        name=(
            f"scatter_null_normal_seed_{seed}"
        ),
        family="null_normal",
        setting="independent",
        setting_value=0.0,
        seed=seed,
        description=(
            "Independent Gaussian variables."
        ),
        dataframe=df,
        x="feature_x",
        y="target_y"
    )


def null_skewed_scenario(
    seed,
    n=DEFAULT_SAMPLE_SIZE
):
    rng = np.random.default_rng(
        seed
    )

    df = make_dataframe(
        rng.lognormal(
            1.5,
            1.0,
            n
        ),
        rng.lognormal(
            2.0,
            1.2,
            n
        )
    )

    return ScatterRobustnessScenario(
        name=(
            f"scatter_null_skewed_seed_{seed}"
        ),
        family="null_skewed",
        setting="independent",
        setting_value=0.0,
        seed=seed,
        description=(
            "Independent strongly skewed variables."
        ),
        dataframe=df,
        x="feature_x",
        y="target_y"
    )


def null_outlier_scenario(
    seed,
    n=DEFAULT_SAMPLE_SIZE
):
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

    outlier_count = max(
        1,
        int(
            n * 0.06
        )
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

    df = make_dataframe(
        x,
        y
    )

    return ScatterRobustnessScenario(
        name=(
            f"scatter_null_outliers_seed_{seed}"
        ),
        family="null_outliers",
        setting="independent",
        setting_value=0.0,
        seed=seed,
        description=(
            "Independent variables with separate "
            "extreme outliers."
        ),
        dataframe=df,
        x="feature_x",
        y="target_y"
    )


def null_small_sample_scenario(
    seed,
    n=30
):
    rng = np.random.default_rng(
        seed
    )

    df = make_dataframe(
        rng.normal(
            0,
            1,
            n
        ),
        rng.normal(
            0,
            1,
            n
        )
    )

    return ScatterRobustnessScenario(
        name=(
            f"scatter_null_small_sample_seed_{seed}"
        ),
        family="null_small_sample",
        setting=f"n={n}",
        setting_value=float(
            n
        ),
        seed=seed,
        description=(
            "Independent Gaussian variables with "
            "only 30 observations."
        ),
        dataframe=df,
        x="feature_x",
        y="target_y"
    )


def build_null_suite(
    seeds=None
):
    if seeds is None:
        seeds = DEFAULT_SEEDS

    scenarios = []

    for seed in seeds:

        scenarios.extend([
            null_normal_scenario(
                seed
            ),
            null_skewed_scenario(
                seed
            ),
            null_outlier_scenario(
                seed
            ),
            null_small_sample_scenario(
                seed
            )
        ])

    return scenarios


# ==================================================
# SCORING
# ==================================================

def score_dataframe(
    df,
    x,
    y
):
    """
    Run one dataframe through the current scatter
    scorer and expose the key calibration statistics.
    """

    candidate = {
        "chart": "scatter",
        "x": x,
        "y": y
    }

    profile = (
        manual_scatter_profile(
            x,
            y
        )
    )

    result = score_scatter_chart(
        df=df,
        candidate=candidate,
        profile=profile
    )

    relationship = (
        relationship_signal_score(
            df,
            x,
            y
        )
    )

    complete_pairs = (
        df[
            [
                x,
                y
            ]
        ]
        .dropna()
    )

    return {
        "observations":
            len(
                complete_pairs
            ),

        "score":
            result[
                "score"
            ],

        "signal":
            result[
                "components"
            ][
                "signal"
            ],

        "visualization_quality":
            result[
                "components"
            ][
                "visualization_quality"
            ],

        "pearson":
            relationship[
                "pearson"
            ],

        "spearman":
            relationship[
                "spearman"
            ],

        "raw_linear_signal":
            relationship[
                "raw_linear_signal"
            ],

        "linear_reliability":
            relationship[
                "linear_reliability"
            ],

        "linear_signal":
            relationship[
                "linear_signal"
            ],

        "mutual_information":
            relationship[
                "mutual_information"
            ],

        "mi_threshold":
            relationship[
                "mi_threshold"
            ],

        "mi_excess":
            relationship[
                "mi_excess"
            ],

        "nonlinear_signal":
            relationship[
                "nonlinear_signal"
            ]
    }


def run_strength_scenario(
    scenario
):
    metrics = score_dataframe(
        df=scenario.dataframe,
        x=scenario.x,
        y=scenario.y
    )

    return {
        "name": scenario.name,
        "family": scenario.family,
        "level": scenario.level,
        "strength_index":
            scenario.strength_index,
        "seed": scenario.seed,
        **metrics
    }


def run_robustness_scenario(
    scenario
):
    metrics = score_dataframe(
        df=scenario.dataframe,
        x=scenario.x,
        y=scenario.y
    )

    return {
        "name": scenario.name,
        "family": scenario.family,
        "setting": scenario.setting,
        "setting_value":
            scenario.setting_value,
        "seed": scenario.seed,
        **metrics
    }


# ==================================================
# SUMMARY HELPERS
# ==================================================

def percentile_10(
    series
):
    return series.quantile(
        0.10
    )


def percentile_90(
    series
):
    return series.quantile(
        0.90
    )


def strength_level_summary(
    results
):
    return (
        results
        .groupby(
            [
                "family",
                "strength_index",
                "level"
            ],
            as_index=False
        )
        .agg(
            scenarios=(
                "score",
                "size"
            ),
            median_score=(
                "score",
                "median"
            ),
            mean_score=(
                "score",
                "mean"
            ),
            p10_score=(
                "score",
                percentile_10
            ),
            p90_score=(
                "score",
                percentile_90
            ),
            median_signal=(
                "signal",
                "median"
            ),
            median_pearson=(
                "pearson",
                "median"
            ),
            median_spearman=(
                "spearman",
                "median"
            ),
            median_linear_signal=(
                "linear_signal",
                "median"
            ),
            median_nonlinear_signal=(
                "nonlinear_signal",
                "median"
            ),
            median_mi=(
                "mutual_information",
                "median"
            )
        )
        .sort_values(
            [
                "family",
                "strength_index"
            ]
        )
        .reset_index(
            drop=True
        )
    )


def adjacent_strength_monotonicity(
    results
):
    """
    Compare adjacent planted strength levels using
    matched seeds.
    """

    rows = []

    for family in sorted(
        results[
            "family"
        ].unique()
    ):

        family_results = (
            results[
                results[
                    "family"
                ]
                == family
            ]
        )

        for lower_strength in range(
            4
        ):

            higher_strength = (
                lower_strength
                + 1
            )

            lower = (
                family_results[
                    family_results[
                        "strength_index"
                    ]
                    == lower_strength
                ][
                    [
                        "seed",
                        "score"
                    ]
                ]
                .rename(
                    columns={
                        "score":
                            "lower_score"
                    }
                )
            )

            higher = (
                family_results[
                    family_results[
                        "strength_index"
                    ]
                    == higher_strength
                ][
                    [
                        "seed",
                        "score"
                    ]
                ]
                .rename(
                    columns={
                        "score":
                            "higher_score"
                    }
                )
            )

            paired = lower.merge(
                higher,
                on="seed",
                how="inner"
            )

            if paired.empty:
                continue

            paired[
                "delta"
            ] = (
                paired[
                    "higher_score"
                ]
                - paired[
                    "lower_score"
                ]
            )

            lower_label = (
                strength_label(
                    lower_strength
                )
            )

            higher_label = (
                strength_label(
                    higher_strength
                )
            )

            rows.append({
                "family": family,
                "comparison": (
                    f"{lower_label} -> "
                    f"{higher_label}"
                ),
                "paired_seeds":
                    len(
                        paired
                    ),
                "median_score_delta":
                    paired[
                        "delta"
                    ].median(),
                "mean_score_delta":
                    paired[
                        "delta"
                    ].mean(),
                "stronger_higher_pct":
                    (
                        (
                            paired[
                                "delta"
                            ]
                            > 0
                        )
                        .mean()
                        * 100
                    )
            })

    return pd.DataFrame(
        rows
    )


def full_strength_monotonicity(
    results
):
    rows = []

    for family in sorted(
        results[
            "family"
        ].unique()
    ):

        family_results = (
            results[
                results[
                    "family"
                ]
                == family
            ]
        )

        pivot = (
            family_results
            .pivot_table(
                index="seed",
                columns="strength_index",
                values="score",
                aggfunc="first"
            )
        )

        required = [
            0,
            1,
            2,
            3,
            4
        ]

        if not all(
            column in pivot.columns
            for column in required
        ):

            monotonic_pct = (
                float(
                    "nan"
                )
            )

        else:

            ordered = (
                pivot[
                    required
                ]
                .dropna()
            )

            monotonic = (
                (
                    ordered[0]
                    < ordered[1]
                )
                & (
                    ordered[1]
                    < ordered[2]
                )
                & (
                    ordered[2]
                    < ordered[3]
                )
                & (
                    ordered[3]
                    < ordered[4]
                )
            )

            monotonic_pct = (
                monotonic.mean()
                * 100
            )

        rows.append({
            "family": family,
            "fully_monotonic_seed_pct":
                monotonic_pct
        })

    return pd.DataFrame(
        rows
    )


def robustness_summary(
    results
):
    return (
        results
        .groupby(
            [
                "family",
                "setting",
                "setting_value"
            ],
            as_index=False
        )
        .agg(
            scenarios=(
                "score",
                "size"
            ),
            median_score=(
                "score",
                "median"
            ),
            mean_score=(
                "score",
                "mean"
            ),
            p10_score=(
                "score",
                percentile_10
            ),
            p90_score=(
                "score",
                percentile_90
            ),
            median_signal=(
                "signal",
                "median"
            ),
            median_raw_linear_signal=(
                "raw_linear_signal",
                "median"
            ),
            median_linear_reliability=(
                "linear_reliability",
                "median"
            ),
            median_linear_signal=(
                "linear_signal",
                "median"
            ),
            median_nonlinear_signal=(
                "nonlinear_signal",
                "median"
            )
        )
        .sort_values(
            [
                "family",
                "setting_value"
            ]
        )
        .reset_index(
            drop=True
        )
    )


def null_summary(
    results
):
    return (
        results
        .groupby(
            "family",
            as_index=False
        )
        .agg(
            scenarios=(
                "score",
                "size"
            ),
            mean_score=(
                "score",
                "mean"
            ),
            median_score=(
                "score",
                "median"
            ),
            p90_score=(
                "score",
                percentile_90
            ),
            highest_score=(
                "score",
                "max"
            ),
            pct_above_50=(
                "score",
                lambda values:
                (
                    values
                    .ge(
                        50
                    )
                    .mean()
                    * 100
                )
            ),
            pct_above_65=(
                "score",
                lambda values:
                (
                    values
                    .ge(
                        65
                    )
                    .mean()
                    * 100
                )
            )
        )
    )


# ==================================================
# PRINT HELPERS
# ==================================================

def format_numeric_columns(
    dataframe,
    columns,
    decimals=2
):
    output = (
        dataframe.copy()
    )

    for column in columns:

        if column in output.columns:

            output[
                column
            ] = output[
                column
            ].map(
                lambda value:
                (
                    f"{value:.{decimals}f}"
                    if pd.notna(
                        value
                    )
                    else "N/A"
                )
            )

    return output


def format_percentage_columns(
    dataframe,
    columns
):
    output = (
        dataframe.copy()
    )

    for column in columns:

        if column in output.columns:

            output[
                column
            ] = output[
                column
            ].map(
                lambda value:
                (
                    f"{value:.1f}%"
                    if pd.notna(
                        value
                    )
                    else "N/A"
                )
            )

    return output


def print_report(
    strength_results,
    level_results,
    adjacent_results,
    full_monotonicity,
    sample_size_results,
    missingness_results,
    null_results
):
    print()
    print(
        "=" * 108
    )
    print(
        "VISIFT SCATTER SCORE CALIBRATION"
    )
    print(
        "=" * 108
    )
    print()

    print(
        f"Strength scenarios tested:        "
        f"{len(strength_results)}"
    )

    print(
        f"Relationship families:            "
        f"{strength_results['family'].nunique()}"
    )

    print(
        f"Random seeds:                     "
        f"{strength_results['seed'].nunique()}"
    )

    print(
        f"Default observations/scenario:    "
        f"{DEFAULT_SAMPLE_SIZE}"
    )

    print()

    print(
        "PURPOSE"
    )
    print(
        "-" * 108
    )

    print(
        "This benchmark isolates the scatter scorer. "
        "It measures score calibration across "
        "graduated relationship strengths and tests "
        "sample-size reliability, missingness, and "
        "null robustness."
    )

    print()

    print(
        "STRENGTH LEVEL SUMMARY"
    )
    print(
        "-" * 108
    )

    printable_levels = (
        format_numeric_columns(
            level_results,
            [
                "median_score",
                "mean_score",
                "p10_score",
                "p90_score",
                "median_signal",
                "median_pearson",
                "median_spearman",
                "median_linear_signal",
                "median_nonlinear_signal",
                "median_mi"
            ]
        )
    )

    print(
        printable_levels.to_string(
            index=False
        )
    )

    print()

    print(
        "ADJACENT STRENGTH COMPARISONS"
    )
    print(
        "-" * 108
    )

    printable_adjacent = (
        format_numeric_columns(
            adjacent_results,
            [
                "median_score_delta",
                "mean_score_delta"
            ]
        )
    )

    printable_adjacent = (
        format_percentage_columns(
            printable_adjacent,
            [
                "stronger_higher_pct"
            ]
        )
    )

    print(
        printable_adjacent.to_string(
            index=False
        )
    )

    print()

    print(
        "FULL FIVE-LEVEL MONOTONICITY"
    )
    print(
        "-" * 108
    )

    printable_full = (
        format_percentage_columns(
            full_monotonicity,
            [
                "fully_monotonic_seed_pct"
            ]
        )
    )

    print(
        printable_full.to_string(
            index=False
        )
    )

    print()

    print(
        "SAMPLE-SIZE RELIABILITY"
    )
    print(
        "-" * 108
    )

    printable_sample = (
        format_numeric_columns(
            sample_size_results,
            [
                "median_score",
                "mean_score",
                "p10_score",
                "p90_score",
                "median_signal",
                "median_raw_linear_signal",
                "median_linear_reliability",
                "median_linear_signal",
                "median_nonlinear_signal"
            ]
        )
    )

    print(
        printable_sample.to_string(
            index=False
        )
    )

    print()

    print(
        "MISSINGNESS ROBUSTNESS"
    )
    print(
        "-" * 108
    )

    printable_missing = (
        format_numeric_columns(
            missingness_results,
            [
                "median_score",
                "mean_score",
                "p10_score",
                "p90_score",
                "median_signal",
                "median_raw_linear_signal",
                "median_linear_reliability",
                "median_linear_signal",
                "median_nonlinear_signal"
            ]
        )
    )

    print(
        printable_missing.to_string(
            index=False
        )
    )

    print()

    print(
        "NULL ROBUSTNESS"
    )
    print(
        "-" * 108
    )

    printable_null = (
        format_numeric_columns(
            null_results,
            [
                "mean_score",
                "median_score",
                "p90_score",
                "highest_score"
            ]
        )
    )

    printable_null = (
        format_percentage_columns(
            printable_null,
            [
                "pct_above_50",
                "pct_above_65"
            ]
        )
    )

    print(
        printable_null.to_string(
            index=False
        )
    )

    print()

    print(
        "INTERPRETATION GUIDE"
    )
    print(
        "-" * 108
    )

    print(
        "Look for:"
    )
    print(
        "  1. Ordinary/null relationships staying near "
        "the bottom of the recommendation scale."
    )
    print(
        "  2. Weak < moderate < strong < extreme within "
        "each relationship family."
    )
    print(
        "  3. Linear calibration roughly tracking the "
        "planted correlation strength."
    )
    print(
        "  4. Nonlinear families increasing through the "
        "permutation-calibrated MI path."
    )
    print(
        "  5. Small-sample reliability reducing chance "
        "correlations without crushing real effects."
    )
    print(
        "  6. Missingness lowering chart suitability "
        "gradually rather than causing erratic scores."
    )
    print(
        "  7. Null skew/outlier/small-sample scenarios "
        "remaining below recommendation thresholds."
    )

    print()

    print(
        "=" * 108
    )
    print(
        f"Raw results saved to {RESULTS_DIR}/"
    )
    print(
        "=" * 108
    )
    print()


# ==================================================
# MAIN BENCHMARK
# ==================================================

def run_scatter_calibration(
    seeds=None,
    n=DEFAULT_SAMPLE_SIZE
):
    if seeds is None:
        seeds = DEFAULT_SEEDS

    # -----------------------------------
    # Graduated relationship strength
    # -----------------------------------

    strength_scenarios = (
        build_strength_suite(
            seeds=seeds,
            n=n
        )
    )

    strength_rows = []

    for index, scenario in enumerate(
        strength_scenarios,
        start=1
    ):

        print(
            "[strength "
            f"{index:03d}/"
            f"{len(strength_scenarios):03d}] "
            f"{scenario.name}"
        )

        strength_rows.append(
            run_strength_scenario(
                scenario
            )
        )

    strength_results = pd.DataFrame(
        strength_rows
    )

    # -----------------------------------
    # Sample-size reliability
    # -----------------------------------

    sample_scenarios = (
        build_sample_size_suite(
            seeds=seeds
        )
    )

    sample_rows = []

    for index, scenario in enumerate(
        sample_scenarios,
        start=1
    ):

        print(
            "[sample "
            f"{index:03d}/"
            f"{len(sample_scenarios):03d}] "
            f"{scenario.name}"
        )

        sample_rows.append(
            run_robustness_scenario(
                scenario
            )
        )

    sample_results = pd.DataFrame(
        sample_rows
    )

    # -----------------------------------
    # Missingness robustness
    # -----------------------------------

    missing_scenarios = (
        build_missingness_suite(
            seeds=seeds,
            n=n
        )
    )

    missing_rows = []

    for index, scenario in enumerate(
        missing_scenarios,
        start=1
    ):

        print(
            "[missing "
            f"{index:03d}/"
            f"{len(missing_scenarios):03d}] "
            f"{scenario.name}"
        )

        missing_rows.append(
            run_robustness_scenario(
                scenario
            )
        )

    missing_results = pd.DataFrame(
        missing_rows
    )

    # -----------------------------------
    # Null robustness
    # -----------------------------------

    null_scenarios = (
        build_null_suite(
            seeds=seeds
        )
    )

    null_rows = []

    for index, scenario in enumerate(
        null_scenarios,
        start=1
    ):

        print(
            "[null "
            f"{index:03d}/"
            f"{len(null_scenarios):03d}] "
            f"{scenario.name}"
        )

        null_rows.append(
            run_robustness_scenario(
                scenario
            )
        )

    null_results_raw = pd.DataFrame(
        null_rows
    )

    # -----------------------------------
    # Summaries
    # -----------------------------------

    level_results = (
        strength_level_summary(
            strength_results
        )
    )

    adjacent_results = (
        adjacent_strength_monotonicity(
            strength_results
        )
    )

    full_monotonicity = (
        full_strength_monotonicity(
            strength_results
        )
    )

    sample_size_results = (
        robustness_summary(
            sample_results
        )
    )

    missingness_results = (
        robustness_summary(
            missing_results
        )
    )

    null_results = (
        null_summary(
            null_results_raw
        )
    )

    # -----------------------------------
    # Save
    # -----------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    strength_results.to_csv(
        RESULTS_DIR
        / "strength_raw.csv",
        index=False
    )

    level_results.to_csv(
        RESULTS_DIR
        / "strength_level_summary.csv",
        index=False
    )

    adjacent_results.to_csv(
        RESULTS_DIR
        / "adjacent_monotonicity.csv",
        index=False
    )

    full_monotonicity.to_csv(
        RESULTS_DIR
        / "full_monotonicity.csv",
        index=False
    )

    sample_results.to_csv(
        RESULTS_DIR
        / "sample_size_raw.csv",
        index=False
    )

    sample_size_results.to_csv(
        RESULTS_DIR
        / "sample_size_summary.csv",
        index=False
    )

    missing_results.to_csv(
        RESULTS_DIR
        / "missingness_raw.csv",
        index=False
    )

    missingness_results.to_csv(
        RESULTS_DIR
        / "missingness_summary.csv",
        index=False
    )

    null_results_raw.to_csv(
        RESULTS_DIR
        / "null_raw.csv",
        index=False
    )

    null_results.to_csv(
        RESULTS_DIR
        / "null_summary.csv",
        index=False
    )

    # -----------------------------------
    # Report
    # -----------------------------------

    print_report(
        strength_results=
            strength_results,
        level_results=
            level_results,
        adjacent_results=
            adjacent_results,
        full_monotonicity=
            full_monotonicity,
        sample_size_results=
            sample_size_results,
        missingness_results=
            missingness_results,
        null_results=
            null_results
    )

    return {
        "strength_raw":
            strength_results,
        "strength_level_summary":
            level_results,
        "adjacent_monotonicity":
            adjacent_results,
        "full_monotonicity":
            full_monotonicity,
        "sample_size_raw":
            sample_results,
        "sample_size_summary":
            sample_size_results,
        "missingness_raw":
            missing_results,
        "missingness_summary":
            missingness_results,
        "null_raw":
            null_results_raw,
        "null_summary":
            null_results
    }


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":

    run_scatter_calibration()
