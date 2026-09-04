from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scoring.bar import (
    count_signal_score,
    group_separation_score,
    score_bar_chart
)

from scoring.box import (
    distribution_signal_score,
    score_box_chart
)


# ==================================================
# CONFIGURATION
# ==================================================

RESULTS_DIR = Path(
    "evaluation/results/categorical_calibration_v1"
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


# ==================================================
# DATA STRUCTURES
# ==================================================

@dataclass(frozen=True)
class GroupEffectScenario:
    """
    Controlled category + numeric relationship.

    Both the mean bar-chart scorer and the box-plot
    scorer are evaluated on the same dataset so their
    score scales can be compared directly.
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
class CountScenario:
    """
    Controlled categorical-frequency scenario for
    count bar charts.
    """

    name: str
    family: str
    level: str
    strength_index: int
    seed: int
    description: str
    dataframe: pd.DataFrame
    x: str


@dataclass(frozen=True)
class RobustnessScenario:
    """
    Category + numeric robustness scenario.
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

def strength_label(
    strength_index
):
    return STRENGTH_LABELS[
        strength_index
    ]


def manual_profile(
    x,
    y=None
):
    """
    Build the minimal semantic profile required by
    the bar and box scorers.

    The hardened Visift benchmark already evaluates
    semantic inference. This calibration benchmark
    intentionally isolates score behavior.
    """

    profiles = [
        {
            "name": x,
            "semantic_type": "categorical"
        }
    ]

    if y is not None:

        profiles.append(
            {
                "name": y,
                "semantic_type":
                    "numeric_continuous"
            }
        )

    return {
        "column_profiles": profiles
    }


def balanced_groups(
    rng,
    n,
    labels
):
    """
    Create nearly balanced groups and shuffle their
    row order.
    """

    repeated = np.resize(
        np.asarray(
            labels,
            dtype=object
        ),
        n
    )

    rng.shuffle(
        repeated
    )

    return repeated


def standardized_group_positions(
    count
):
    """
    Return equally spaced, centered group positions
    with standard deviation approximately one.

    This lets category-count experiments keep the
    overall planted effect scale comparable even when
    the number of categories changes.
    """

    positions = np.linspace(
        -1,
        1,
        count
    )

    positions = (
        positions
        - positions.mean()
    )

    std = positions.std()

    if std <= 0:
        return positions

    return (
        positions
        / std
    )


# ==================================================
# GROUP-SEPARATION FAMILY
# ==================================================

def group_effect_values(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE
):
    """
    Generate four balanced groups with increasingly
    separated numeric means.

    Noise standard deviation is fixed at 1.

    The effect-scale sequence intentionally covers
    small through very large group effects without
    directly hard-coding expected Visift scores.
    """

    rng = np.random.default_rng(
        seed
    )

    labels = np.array([
        "A",
        "B",
        "C",
        "D"
    ])

    groups = balanced_groups(
        rng,
        n,
        labels
    )

    effect_scale = {
        0: 0.00,
        1: 0.15,
        2: 0.35,
        3: 0.70,
        4: 1.40
    }[
        strength_index
    ]

    positions = {
        "A": -1.5,
        "B": -0.5,
        "C": 0.5,
        "D": 1.5
    }

    means = np.array([
        effect_scale
        * positions[
            group
        ]
        for group in groups
    ])

    values = (
        means
        + rng.normal(
            0,
            1,
            n
        )
    )

    return (
        groups,
        values
    )


def group_effect_scenario(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE
):
    level = strength_label(
        strength_index
    )

    groups, values = (
        group_effect_values(
            seed=seed,
            strength_index=
                strength_index,
            n=n
        )
    )

    df = pd.DataFrame({
        "group": groups,
        "outcome": values
    })

    return GroupEffectScenario(
        name=(
            f"categorical_group_effect_"
            f"{level}_seed_{seed}"
        ),
        family="group_separation",
        level=level,
        strength_index=
            strength_index,
        seed=seed,
        description=(
            "Four balanced categories with "
            f"{level} location separation."
        ),
        dataframe=df,
        x="group",
        y="outcome"
    )


# ==================================================
# CATEGORY-IMBALANCE FAMILY
# ==================================================

def category_imbalance_values(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE
):
    """
    Generate four-category frequency distributions
    ranging from balanced to highly concentrated.
    """

    rng = np.random.default_rng(
        seed
    )

    probabilities = {
        0: [
            0.25,
            0.25,
            0.25,
            0.25
        ],
        1: [
            0.35,
            0.25,
            0.22,
            0.18
        ],
        2: [
            0.50,
            0.25,
            0.15,
            0.10
        ],
        3: [
            0.65,
            0.20,
            0.10,
            0.05
        ],
        4: [
            0.80,
            0.12,
            0.06,
            0.02
        ]
    }[
        strength_index
    ]

    groups = rng.choice(
        [
            "A",
            "B",
            "C",
            "D"
        ],
        size=n,
        p=probabilities
    )

    return groups


def category_imbalance_scenario(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE
):
    level = strength_label(
        strength_index
    )

    groups = (
        category_imbalance_values(
            seed=seed,
            strength_index=
                strength_index,
            n=n
        )
    )

    df = pd.DataFrame({
        "segment": groups
    })

    return CountScenario(
        name=(
            f"categorical_imbalance_"
            f"{level}_seed_{seed}"
        ),
        family="category_imbalance",
        level=level,
        strength_index=
            strength_index,
        seed=seed,
        description=(
            "Four-category frequency "
            f"distribution at {level} "
            "imbalance."
        ),
        dataframe=df,
        x="segment"
    )


# ==================================================
# DIFFERENTIAL OUTLIER FAMILY
# ==================================================

def differential_outlier_values(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE
):
    """
    Generate groups with equal central location but
    increasingly heavy two-sided outliers in group D.

    This family is intentionally box-plot specific:

        - the box score should recognize distribution
          differences as outliers become substantial

        - the mean bar score should ideally remain
          relatively low because the group means are
          designed to stay similar
    """

    rng = np.random.default_rng(
        seed
    )

    labels = np.array([
        "A",
        "B",
        "C",
        "D"
    ])

    groups = balanced_groups(
        rng,
        n,
        labels
    )

    values = rng.normal(
        0,
        1,
        n
    )

    target_rate = {
        0: 0.00,
        1: 0.04,
        2: 0.12,
        3: 0.24,
        4: 0.40
    }[
        strength_index
    ]

    d_indices = np.flatnonzero(
        groups == "D"
    )

    outlier_count = int(
        len(
            d_indices
        )
        * target_rate
    )

    if outlier_count > 0:

        selected = rng.choice(
            d_indices,
            size=outlier_count,
            replace=False
        )

        # Pair positive and negative extreme values as
        # closely as possible so the group mean remains
        # near the other groups while dispersion changes.
        signs = np.ones(
            outlier_count
        )

        signs[
            :outlier_count // 2
        ] = -1

        rng.shuffle(
            signs
        )

        magnitudes = rng.uniform(
            5,
            8,
            outlier_count
        )

        values[
            selected
        ] += (
            signs
            * magnitudes
        )

    return (
        groups,
        values
    )


def differential_outlier_scenario(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE
):
    level = strength_label(
        strength_index
    )

    groups, values = (
        differential_outlier_values(
            seed=seed,
            strength_index=
                strength_index,
            n=n
        )
    )

    df = pd.DataFrame({
        "group": groups,
        "outcome": values
    })

    return GroupEffectScenario(
        name=(
            "categorical_differential_"
            f"outliers_{level}_seed_{seed}"
        ),
        family="differential_outliers",
        level=level,
        strength_index=
            strength_index,
        seed=seed,
        description=(
            "Equal-location groups with "
            f"{level} differential outlier "
            "structure."
        ),
        dataframe=df,
        x="group",
        y="outcome"
    )



# ==================================================
# DISPERSION-SHIFT FAMILY
# ==================================================

def dispersion_shift_values(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE
):
    """
    Generate groups with equal central location but
    increasingly different within-group spread.

    Groups A-C retain standard deviation 1 while
    group D becomes progressively more variable.

    This is a core box-plot use case:
        - mean bars should remain near the floor
        - box plots should eventually recognize the
          difference in distribution spread
    """

    rng = np.random.default_rng(
        seed
    )

    labels = np.array([
        "A",
        "B",
        "C",
        "D"
    ])

    groups = balanced_groups(
        rng,
        n,
        labels
    )

    d_scale = {
        0: 1.00,
        1: 1.25,
        2: 1.75,
        3: 2.50,
        4: 4.00
    }[
        strength_index
    ]

    scale_by_group = {
        "A": 1.0,
        "B": 1.0,
        "C": 1.0,
        "D": d_scale
    }

    values = np.array([
        rng.normal(
            0,
            scale_by_group[
                group
            ]
        )
        for group in groups
    ])

    return (
        groups,
        values
    )


def dispersion_shift_scenario(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE
):
    level = strength_label(
        strength_index
    )

    groups, values = (
        dispersion_shift_values(
            seed=seed,
            strength_index=
                strength_index,
            n=n
        )
    )

    df = pd.DataFrame({
        "group": groups,
        "outcome": values
    })

    return GroupEffectScenario(
        name=(
            "categorical_dispersion_shift_"
            f"{level}_seed_{seed}"
        ),
        family="dispersion_shift",
        level=level,
        strength_index=
            strength_index,
        seed=seed,
        description=(
            "Equal-center groups with "
            f"{level} differences in "
            "within-group spread."
        ),
        dataframe=df,
        x="group",
        y="outcome"
    )


# ==================================================
# STRENGTH SUITES
# ==================================================

def build_group_strength_suite(
    seeds=None,
    n=DEFAULT_SAMPLE_SIZE
):
    """
    Build category + numeric strength scenarios.

    Per seed:
        5 group-separation levels
        5 differential-outlier levels
        5 dispersion-shift levels
    """

    if seeds is None:
        seeds = DEFAULT_SEEDS

    scenarios = []

    for seed in seeds:

        for strength_index in range(
            5
        ):

            scenarios.extend([
                group_effect_scenario(
                    seed=seed,
                    strength_index=
                        strength_index,
                    n=n
                ),
                differential_outlier_scenario(
                    seed=seed,
                    strength_index=
                        strength_index,
                    n=n
                ),
                dispersion_shift_scenario(
                    seed=seed,
                    strength_index=
                        strength_index,
                    n=n
                )
            ])

    return scenarios


def build_count_strength_suite(
    seeds=None,
    n=DEFAULT_SAMPLE_SIZE
):
    """
    Build graduated count-bar imbalance scenarios.
    """

    if seeds is None:
        seeds = DEFAULT_SEEDS

    scenarios = []

    for seed in seeds:

        for strength_index in range(
            5
        ):

            scenarios.append(
                category_imbalance_scenario(
                    seed=seed,
                    strength_index=
                        strength_index,
                    n=n
                )
            )

    return scenarios


# ==================================================
# GROUP-SUPPORT ROBUSTNESS
# ==================================================

def group_support_scenario(
    seed,
    strength_index,
    n=200
):
    """
    Hold a meaningful group effect approximately
    fixed while making category sizes increasingly
    unequal.

    This tests whether rare groups reduce confidence
    through sample-support scoring.
    """

    rng = np.random.default_rng(
        seed
    )

    probabilities = {
        0: [
            0.25,
            0.25,
            0.25,
            0.25
        ],
        1: [
            0.40,
            0.30,
            0.20,
            0.10
        ],
        2: [
            0.55,
            0.25,
            0.15,
            0.05
        ],
        3: [
            0.70,
            0.20,
            0.08,
            0.02
        ],
        4: [
            0.85,
            0.10,
            0.04,
            0.01
        ]
    }[
        strength_index
    ]

    groups = rng.choice(
        [
            "A",
            "B",
            "C",
            "D"
        ],
        size=n,
        p=probabilities
    )

    means = {
        "A": -1.05,
        "B": -0.35,
        "C": 0.35,
        "D": 1.05
    }

    values = np.array([
        means[
            group
        ]
        for group in groups
    ])

    values = (
        values
        + rng.normal(
            0,
            1,
            n
        )
    )

    df = pd.DataFrame({
        "group": groups,
        "outcome": values
    })

    level = strength_label(
        strength_index
    )

    return RobustnessScenario(
        name=(
            "categorical_group_support_"
            f"{level}_seed_{seed}"
        ),
        family="group_support",
        setting=level,
        setting_value=float(
            strength_index
        ),
        seed=seed,
        description=(
            "Fixed group effect with "
            f"{level} group-size imbalance."
        ),
        dataframe=df,
        x="group",
        y="outcome"
    )


def build_group_support_suite(
    seeds=None
):
    if seeds is None:
        seeds = DEFAULT_SEEDS

    scenarios = []

    for seed in seeds:

        for strength_index in range(
            5
        ):

            scenarios.append(
                group_support_scenario(
                    seed=seed,
                    strength_index=
                        strength_index
                )
            )

    return scenarios


# ==================================================
# CATEGORY-COUNT / READABILITY ROBUSTNESS
# ==================================================

def category_count_scenario(
    seed,
    category_count,
    n=DEFAULT_SAMPLE_SIZE
):
    """
    Hold total location structure approximately
    constant while varying the number of categories.

    This isolates the readability penalty applied
    by bar and box plots.
    """

    rng = np.random.default_rng(
        seed
    )

    labels = [
        f"G{i + 1}"
        for i in range(
            category_count
        )
    ]

    groups = balanced_groups(
        rng,
        n,
        labels
    )

    positions = (
        standardized_group_positions(
            category_count
        )
    )

    means = {
        label: (
            0.75
            * positions[
                index
            ]
        )
        for index, label
        in enumerate(
            labels
        )
    }

    values = np.array([
        means[
            group
        ]
        for group in groups
    ])

    values = (
        values
        + rng.normal(
            0,
            1,
            n
        )
    )

    df = pd.DataFrame({
        "group": groups,
        "outcome": values
    })

    return RobustnessScenario(
        name=(
            "categorical_category_count_"
            f"{category_count}_seed_{seed}"
        ),
        family="category_count",
        setting=(
            f"{category_count} groups"
        ),
        setting_value=float(
            category_count
        ),
        seed=seed,
        description=(
            "Comparable group effect with "
            f"{category_count} categories."
        ),
        dataframe=df,
        x="group",
        y="outcome"
    )


def build_category_count_suite(
    seeds=None
):
    if seeds is None:
        seeds = DEFAULT_SEEDS

    category_counts = [
        2,
        4,
        8,
        12,
        20
    ]

    scenarios = []

    for seed in seeds:

        for category_count in (
            category_counts
        ):

            scenarios.append(
                category_count_scenario(
                    seed=seed,
                    category_count=
                        category_count
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
    Hold a strong category + numeric relationship
    fixed while progressively removing outcome
    values at random.
    """

    groups, values = (
        group_effect_values(
            seed=seed,
            strength_index=3,
            n=n
        )
    )

    df = pd.DataFrame({
        "group": groups,
        "outcome": values
    })

    rng = np.random.default_rng(
        seed
        + 100_000
    )

    missing_count = int(
        n
        * missingness
    )

    if missing_count > 0:

        indices = rng.choice(
            df.index,
            size=missing_count,
            replace=False
        )

        df.loc[
            indices,
            "outcome"
        ] = np.nan

    return RobustnessScenario(
        name=(
            "categorical_missingness_"
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
            "Strong category + numeric "
            f"relationship with "
            f"{missingness * 100:.0f}% "
            "missing outcomes."
        ),
        dataframe=df,
        x="group",
        y="outcome"
    )


def build_missingness_suite(
    seeds=None,
    n=DEFAULT_SAMPLE_SIZE
):
    if seeds is None:
        seeds = DEFAULT_SEEDS

    missingness_levels = [
        0.00,
        0.20,
        0.40,
        0.60,
        0.80
    ]

    scenarios = []

    for seed in seeds:

        for missingness in (
            missingness_levels
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

def null_group_scenario(
    seed,
    n=DEFAULT_SAMPLE_SIZE
):
    """
    Random categorical groups with independent
    numeric outcomes.
    """

    rng = np.random.default_rng(
        seed
    )

    df = pd.DataFrame({
        "group": rng.choice(
            [
                "A",
                "B",
                "C",
                "D"
            ],
            size=n
        ),
        "outcome": rng.normal(
            0,
            1,
            n
        )
    })

    return RobustnessScenario(
        name=(
            "categorical_null_group_"
            f"seed_{seed}"
        ),
        family="null_group_random",
        setting="independent",
        setting_value=0.0,
        seed=seed,
        description=(
            "Random groups with an independent "
            "numeric outcome."
        ),
        dataframe=df,
        x="group",
        y="outcome"
    )


def build_null_suite(
    seeds=None,
    n=DEFAULT_SAMPLE_SIZE
):
    if seeds is None:
        seeds = DEFAULT_SEEDS

    return [
        null_group_scenario(
            seed=seed,
            n=n
        )
        for seed in seeds
    ]


# ==================================================
# SCORING
# ==================================================

def score_group_dataframe(
    df,
    x,
    y
):
    """
    Score the same category + numeric dataset as
    both a mean bar chart and a box plot.
    """

    profile = manual_profile(
        x,
        y
    )

    bar_candidate = {
        "chart": "bar",
        "x": x,
        "y": y,
        "aggregation": "mean"
    }

    box_candidate = {
        "chart": "box",
        "x": x,
        "y": y
    }

    bar_result = score_bar_chart(
        df=df,
        candidate=bar_candidate,
        profile=profile
    )

    box_result = score_box_chart(
        df=df,
        candidate=box_candidate,
        profile=profile
    )

    clean = (
        df[
            [
                x,
                y
            ]
        ]
        .dropna()
    )

    counts = (
        clean[x]
        .value_counts()
    )

    raw_bar_signal = (
        group_separation_score(
            df,
            x,
            y
        )
    )

    box_distribution = (
        distribution_signal_score(
            df,
            x,
            y
        )
    )

    return {
        "observations": len(
            clean
        ),
        "groups": int(
            clean[x]
            .nunique()
        ),
        "minimum_group_size": (
            int(
                counts.min()
            )
            if not counts.empty
            else 0
        ),

        "bar_score":
            bar_result[
                "score"
            ],
        "bar_signal":
            bar_result[
                "components"
            ][
                "signal"
            ],
        "bar_quality":
            bar_result[
                "components"
            ][
                "visualization_quality"
            ],
        "bar_support":
            bar_result[
                "components"
            ][
                "sample_support"
            ],
        "bar_raw_eta_signal":
            raw_bar_signal,

        "box_score":
            box_result[
                "score"
            ],
        "box_signal":
            box_result[
                "components"
            ][
                "signal"
            ],
        "box_quality":
            box_result[
                "components"
            ][
                "visualization_quality"
            ],
        "box_support":
            box_result[
                "components"
            ][
                "sample_support"
            ],
        "box_raw_signal":
            box_distribution[
                "signal"
            ],
        "box_eta_score":
            box_distribution[
                "eta_squared_score"
            ],
        "box_median_score":
            box_distribution[
                "median_separation_score"
            ],
        "box_outlier_score":
            box_distribution[
                "outlier_score"
            ],
        "box_outlier_rate":
            box_distribution[
                "outlier_rate"
            ],
        "box_outlier_rate_disparity":
            box_distribution.get(
                "outlier_rate_disparity",
                0
            ),
        "box_dispersion_score":
            box_distribution.get(
                "dispersion_score",
                0
            ),
        "box_dispersion_ratio":
            box_distribution.get(
                "dispersion_ratio",
                1
            )
    }


def score_count_dataframe(
    df,
    x
):
    """
    Score one count bar chart.
    """

    profile = manual_profile(
        x
    )

    candidate = {
        "chart": "bar",
        "x": x,
        "aggregation": "count"
    }

    result = score_bar_chart(
        df=df,
        candidate=candidate,
        profile=profile
    )

    counts = (
        df[x]
        .dropna()
        .value_counts()
    )

    raw_signal = (
        count_signal_score(
            counts
        )
    )

    return {
        "observations":
            int(
                counts.sum()
            ),
        "groups":
            int(
                len(
                    counts
                )
            ),
        "minimum_group_size": (
            int(
                counts.min()
            )
            if not counts.empty
            else 0
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
        "quality":
            result[
                "components"
            ][
                "visualization_quality"
            ],
        "support":
            result[
                "components"
            ][
                "sample_support"
            ],
        "raw_imbalance_signal":
            raw_signal
    }


# ==================================================
# RUN INDIVIDUAL SCENARIOS
# ==================================================

def run_group_strength_scenario(
    scenario
):
    metrics = score_group_dataframe(
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


def run_count_strength_scenario(
    scenario
):
    metrics = score_count_dataframe(
        df=scenario.dataframe,
        x=scenario.x
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
    metrics = score_group_dataframe(
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


def group_strength_summary(
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
                "bar_score",
                "size"
            ),

            median_bar_score=(
                "bar_score",
                "median"
            ),
            p10_bar_score=(
                "bar_score",
                percentile_10
            ),
            p90_bar_score=(
                "bar_score",
                percentile_90
            ),
            median_bar_signal=(
                "bar_signal",
                "median"
            ),
            median_eta_signal=(
                "bar_raw_eta_signal",
                "median"
            ),

            median_box_score=(
                "box_score",
                "median"
            ),
            p10_box_score=(
                "box_score",
                percentile_10
            ),
            p90_box_score=(
                "box_score",
                percentile_90
            ),
            median_box_signal=(
                "box_signal",
                "median"
            ),
            median_box_eta=(
                "box_eta_score",
                "median"
            ),
            median_box_median=(
                "box_median_score",
                "median"
            ),
            median_box_outlier=(
                "box_outlier_score",
                "median"
            ),
            median_box_outlier_rate=(
                "box_outlier_rate",
                "median"
            ),
            median_box_outlier_disparity=(
                "box_outlier_rate_disparity",
                "median"
            ),
            median_box_dispersion=(
                "box_dispersion_score",
                "median"
            ),
            median_box_dispersion_ratio=(
                "box_dispersion_ratio",
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


def count_strength_summary(
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
            median_minimum_group_size=(
                "minimum_group_size",
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


def adjacent_monotonicity(
    results,
    score_column,
    family_column="family"
):
    """
    Compare adjacent strength levels using matched
    random seeds.
    """

    rows = []

    for family in sorted(
        results[
            family_column
        ].unique()
    ):

        family_results = (
            results[
                results[
                    family_column
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
                        score_column
                    ]
                ]
                .rename(
                    columns={
                        score_column:
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
                        score_column
                    ]
                ]
                .rename(
                    columns={
                        score_column:
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

            rows.append({
                "family": family,
                "comparison": (
                    f"{strength_label(lower_strength)} "
                    "-> "
                    f"{strength_label(higher_strength)}"
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


def full_monotonicity(
    results,
    score_column
):
    """
    Measure the percentage of seeds for which all
    five strength levels increase strictly.
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

        pivot = (
            family_results
            .pivot_table(
                index="seed",
                columns="strength_index",
                values=score_column,
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

        ordered = (
            pivot[
                required
            ]
            .dropna()
        )

        if ordered.empty:

            pct = float(
                "nan"
            )

        else:

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

            pct = (
                monotonic.mean()
                * 100
            )

        rows.append({
            "family": family,
            "fully_monotonic_seed_pct":
                pct
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
                "bar_score",
                "size"
            ),
            median_min_group=(
                "minimum_group_size",
                "median"
            ),
            median_bar_score=(
                "bar_score",
                "median"
            ),
            p10_bar_score=(
                "bar_score",
                percentile_10
            ),
            p90_bar_score=(
                "bar_score",
                percentile_90
            ),
            median_bar_signal=(
                "bar_signal",
                "median"
            ),
            median_bar_support=(
                "bar_support",
                "median"
            ),
            median_box_score=(
                "box_score",
                "median"
            ),
            p10_box_score=(
                "box_score",
                percentile_10
            ),
            p90_box_score=(
                "box_score",
                percentile_90
            ),
            median_box_signal=(
                "box_signal",
                "median"
            ),
            median_box_support=(
                "box_support",
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
    """
    Summarize null group scores for both chart forms.
    """

    return pd.DataFrame([
        {
            "chart": "mean_bar",
            "scenarios": len(
                results
            ),
            "mean_score":
                results[
                    "bar_score"
                ].mean(),
            "median_score":
                results[
                    "bar_score"
                ].median(),
            "p90_score":
                results[
                    "bar_score"
                ].quantile(
                    0.90
                ),
            "highest_score":
                results[
                    "bar_score"
                ].max(),
            "pct_above_50":
                results[
                    "bar_score"
                ].ge(
                    50
                ).mean()
                * 100,
            "pct_above_65":
                results[
                    "bar_score"
                ].ge(
                    65
                ).mean()
                * 100
        },
        {
            "chart": "box",
            "scenarios": len(
                results
            ),
            "mean_score":
                results[
                    "box_score"
                ].mean(),
            "median_score":
                results[
                    "box_score"
                ].median(),
            "p90_score":
                results[
                    "box_score"
                ].quantile(
                    0.90
                ),
            "highest_score":
                results[
                    "box_score"
                ].max(),
            "pct_above_50":
                results[
                    "box_score"
                ].ge(
                    50
                ).mean()
                * 100,
            "pct_above_65":
                results[
                    "box_score"
                ].ge(
                    65
                ).mean()
                * 100
        }
    ])


# ==================================================
# PRINT HELPERS
# ==================================================

def format_numeric_columns(
    dataframe,
    columns,
    decimals=2
):
    output = dataframe.copy()

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
    output = dataframe.copy()

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


# ==================================================
# REPORT
# ==================================================

def print_report(
    group_strength_results,
    group_summary,
    bar_monotonicity,
    box_monotonicity,
    bar_full_monotonicity,
    box_full_monotonicity,
    count_summary,
    count_monotonicity,
    count_full_monotonicity,
    support_summary,
    category_count_summary_results,
    missingness_summary_results,
    null_summary_results
):
    print()
    print(
        "=" * 112
    )
    print(
        "VISIFT CATEGORICAL SCORE CALIBRATION"
    )
    print(
        "=" * 112
    )
    print()

    print(
        f"Group-strength rows scored:       "
        f"{len(group_strength_results)}"
    )

    print(
        f"Random seeds:                     "
        f"{group_strength_results['seed'].nunique()}"
    )

    print()

    print(
        "PURPOSE"
    )
    print(
        "-" * 112
    )

    print(
        "This benchmark isolates Visift's count-bar, "
        "mean-bar, and box-plot scorers. It tests "
        "graduated group effects, category-frequency "
        "imbalance, differential outliers, dispersion "
        "shifts, group support, category count, "
        "missingness, and null behavior."
    )

    print()

    # -----------------------------------
    # Category + numeric strength
    # -----------------------------------

    print(
        "CATEGORY + NUMERIC STRENGTH"
    )
    print(
        "-" * 112
    )

    printable_group = (
        format_numeric_columns(
            group_summary,
            [
                "median_bar_score",
                "p10_bar_score",
                "p90_bar_score",
                "median_bar_signal",
                "median_eta_signal",
                "median_box_score",
                "p10_box_score",
                "p90_box_score",
                "median_box_signal",
                "median_box_eta",
                "median_box_median",
                "median_box_outlier",
                "median_box_outlier_rate",
                "median_box_outlier_disparity",
                "median_box_dispersion",
                "median_box_dispersion_ratio"
            ]
        )
    )

    print(
        printable_group.to_string(
            index=False
        )
    )

    print()

    print(
        "MEAN-BAR ADJACENT STRENGTH COMPARISONS"
    )
    print(
        "-" * 112
    )

    printable = (
        format_numeric_columns(
            bar_monotonicity,
            [
                "median_score_delta",
                "mean_score_delta"
            ]
        )
    )

    printable = (
        format_percentage_columns(
            printable,
            [
                "stronger_higher_pct"
            ]
        )
    )

    print(
        printable.to_string(
            index=False
        )
    )

    print()

    print(
        "BOX-PLOT ADJACENT STRENGTH COMPARISONS"
    )
    print(
        "-" * 112
    )

    printable = (
        format_numeric_columns(
            box_monotonicity,
            [
                "median_score_delta",
                "mean_score_delta"
            ]
        )
    )

    printable = (
        format_percentage_columns(
            printable,
            [
                "stronger_higher_pct"
            ]
        )
    )

    print(
        printable.to_string(
            index=False
        )
    )

    print()

    print(
        "FULL FIVE-LEVEL MONOTONICITY"
    )
    print(
        "-" * 112
    )

    combined_full = (
        bar_full_monotonicity
        .rename(
            columns={
                "fully_monotonic_seed_pct":
                    "mean_bar_pct"
            }
        )
        .merge(
            box_full_monotonicity
            .rename(
                columns={
                    "fully_monotonic_seed_pct":
                        "box_pct"
                }
            ),
            on="family",
            how="outer"
        )
    )

    combined_full = (
        format_percentage_columns(
            combined_full,
            [
                "mean_bar_pct",
                "box_pct"
            ]
        )
    )

    print(
        combined_full.to_string(
            index=False
        )
    )

    print()

    # -----------------------------------
    # Count bars
    # -----------------------------------

    print(
        "COUNT-BAR CATEGORY IMBALANCE"
    )
    print(
        "-" * 112
    )

    printable_count = (
        format_numeric_columns(
            count_summary,
            [
                "median_score",
                "mean_score",
                "p10_score",
                "p90_score",
                "median_signal",
                "median_minimum_group_size"
            ]
        )
    )

    print(
        printable_count.to_string(
            index=False
        )
    )

    print()

    print(
        "COUNT-BAR ADJACENT STRENGTH COMPARISONS"
    )
    print(
        "-" * 112
    )

    printable = (
        format_numeric_columns(
            count_monotonicity,
            [
                "median_score_delta",
                "mean_score_delta"
            ]
        )
    )

    printable = (
        format_percentage_columns(
            printable,
            [
                "stronger_higher_pct"
            ]
        )
    )

    print(
        printable.to_string(
            index=False
        )
    )

    print()

    print(
        "COUNT-BAR FULL FIVE-LEVEL MONOTONICITY"
    )
    print(
        "-" * 112
    )

    printable = (
        format_percentage_columns(
            count_full_monotonicity,
            [
                "fully_monotonic_seed_pct"
            ]
        )
    )

    print(
        printable.to_string(
            index=False
        )
    )

    print()

    # -----------------------------------
    # Robustness
    # -----------------------------------

    for title, dataframe in [
        (
            "GROUP-SUPPORT ROBUSTNESS",
            support_summary
        ),
        (
            "CATEGORY-COUNT / READABILITY ROBUSTNESS",
            category_count_summary_results
        ),
        (
            "MISSINGNESS ROBUSTNESS",
            missingness_summary_results
        )
    ]:

        print(
            title
        )
        print(
            "-" * 112
        )

        printable = (
            format_numeric_columns(
                dataframe,
                [
                    "median_min_group",
                    "median_bar_score",
                    "p10_bar_score",
                    "p90_bar_score",
                    "median_bar_signal",
                    "median_bar_support",
                    "median_box_score",
                    "p10_box_score",
                    "p90_box_score",
                    "median_box_signal",
                    "median_box_support"
                ]
            )
        )

        print(
            printable.to_string(
                index=False
            )
        )

        print()

    # -----------------------------------
    # Null behavior
    # -----------------------------------

    print(
        "NULL GROUP ROBUSTNESS"
    )
    print(
        "-" * 112
    )

    printable_null = (
        format_numeric_columns(
            null_summary_results,
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
        "-" * 112
    )

    print(
        "Look for:"
    )
    print(
        "  1. Random group effects staying near the "
        "bottom of the recommendation scale."
    )
    print(
        "  2. Mean-bar and box scores increasing "
        "smoothly as true group separation increases."
    )
    print(
        "  3. Box plots responding to differential "
        "within-group outliers while mean bars stay "
        "comparatively low."
    )
    print(
        "  4. Box plots responding to meaningful "
        "differences in within-group spread while "
        "mean bars stay comparatively low."
    )
    print(
        "  5. Count-bar scores rising gradually as "
        "category imbalance becomes stronger."
    )
    print(
        "  6. Rare groups reducing support rather than "
        "creating unstable high recommendations."
    )
    print(
        "  7. Category-count penalties matching visual "
        "readability limits."
    )
    print(
        "  8. Missingness degrading scores gradually."
    )
    print(
        "  9. No null bar/box score regularly crossing "
        "the 50-point recommendation threshold."
    )

    print()

    print(
        "=" * 112
    )
    print(
        f"Raw results saved to {RESULTS_DIR}/"
    )
    print(
        "=" * 112
    )
    print()


# ==================================================
# MAIN BENCHMARK
# ==================================================

def run_categorical_calibration(
    seeds=None,
    n=DEFAULT_SAMPLE_SIZE
):
    if seeds is None:
        seeds = DEFAULT_SEEDS

    # -----------------------------------
    # Category + numeric strength
    # -----------------------------------

    group_scenarios = (
        build_group_strength_suite(
            seeds=seeds,
            n=n
        )
    )

    group_rows = []

    for index, scenario in enumerate(
        group_scenarios,
        start=1
    ):

        print(
            "[group "
            f"{index:03d}/"
            f"{len(group_scenarios):03d}] "
            f"{scenario.name}"
        )

        group_rows.append(
            run_group_strength_scenario(
                scenario
            )
        )

    group_results = pd.DataFrame(
        group_rows
    )

    # -----------------------------------
    # Count-bar imbalance
    # -----------------------------------

    count_scenarios = (
        build_count_strength_suite(
            seeds=seeds,
            n=n
        )
    )

    count_rows = []

    for index, scenario in enumerate(
        count_scenarios,
        start=1
    ):

        print(
            "[count "
            f"{index:03d}/"
            f"{len(count_scenarios):03d}] "
            f"{scenario.name}"
        )

        count_rows.append(
            run_count_strength_scenario(
                scenario
            )
        )

    count_results = pd.DataFrame(
        count_rows
    )

    # -----------------------------------
    # Robustness suites
    # -----------------------------------

    support_rows = [
        run_robustness_scenario(
            scenario
        )
        for scenario
        in build_group_support_suite(
            seeds=seeds
        )
    ]

    category_count_rows = [
        run_robustness_scenario(
            scenario
        )
        for scenario
        in build_category_count_suite(
            seeds=seeds
        )
    ]

    missingness_rows = [
        run_robustness_scenario(
            scenario
        )
        for scenario
        in build_missingness_suite(
            seeds=seeds,
            n=n
        )
    ]

    null_rows = [
        run_robustness_scenario(
            scenario
        )
        for scenario
        in build_null_suite(
            seeds=seeds,
            n=n
        )
    ]

    support_results = pd.DataFrame(
        support_rows
    )

    category_count_results = (
        pd.DataFrame(
            category_count_rows
        )
    )

    missingness_results = (
        pd.DataFrame(
            missingness_rows
        )
    )

    null_results = pd.DataFrame(
        null_rows
    )

    # -----------------------------------
    # Summaries
    # -----------------------------------

    group_summary = (
        group_strength_summary(
            group_results
        )
    )

    bar_monotonicity = (
        adjacent_monotonicity(
            group_results,
            "bar_score"
        )
    )

    box_monotonicity = (
        adjacent_monotonicity(
            group_results,
            "box_score"
        )
    )

    bar_full = full_monotonicity(
        group_results,
        "bar_score"
    )

    box_full = full_monotonicity(
        group_results,
        "box_score"
    )

    count_summary_results = (
        count_strength_summary(
            count_results
        )
    )

    count_monotonicity = (
        adjacent_monotonicity(
            count_results,
            "score"
        )
    )

    count_full = full_monotonicity(
        count_results,
        "score"
    )

    support_summary = (
        robustness_summary(
            support_results
        )
    )

    category_count_summary_results = (
        robustness_summary(
            category_count_results
        )
    )

    missingness_summary_results = (
        robustness_summary(
            missingness_results
        )
    )

    null_summary_results = (
        null_summary(
            null_results
        )
    )

    # -----------------------------------
    # Save
    # -----------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    outputs = {
        "group_strength_raw.csv":
            group_results,
        "group_strength_summary.csv":
            group_summary,
        "mean_bar_adjacent_monotonicity.csv":
            bar_monotonicity,
        "box_adjacent_monotonicity.csv":
            box_monotonicity,
        "mean_bar_full_monotonicity.csv":
            bar_full,
        "box_full_monotonicity.csv":
            box_full,
        "count_strength_raw.csv":
            count_results,
        "count_strength_summary.csv":
            count_summary_results,
        "count_adjacent_monotonicity.csv":
            count_monotonicity,
        "count_full_monotonicity.csv":
            count_full,
        "group_support_raw.csv":
            support_results,
        "group_support_summary.csv":
            support_summary,
        "category_count_raw.csv":
            category_count_results,
        "category_count_summary.csv":
            category_count_summary_results,
        "missingness_raw.csv":
            missingness_results,
        "missingness_summary.csv":
            missingness_summary_results,
        "null_raw.csv":
            null_results,
        "null_summary.csv":
            null_summary_results
    }

    for filename, dataframe in (
        outputs.items()
    ):

        dataframe.to_csv(
            RESULTS_DIR
            / filename,
            index=False
        )

    # -----------------------------------
    # Report
    # -----------------------------------

    print_report(
        group_strength_results=
            group_results,
        group_summary=
            group_summary,
        bar_monotonicity=
            bar_monotonicity,
        box_monotonicity=
            box_monotonicity,
        bar_full_monotonicity=
            bar_full,
        box_full_monotonicity=
            box_full,
        count_summary=
            count_summary_results,
        count_monotonicity=
            count_monotonicity,
        count_full_monotonicity=
            count_full,
        support_summary=
            support_summary,
        category_count_summary_results=
            category_count_summary_results,
        missingness_summary_results=
            missingness_summary_results,
        null_summary_results=
            null_summary_results
    )

    return {
        "group_strength_raw":
            group_results,
        "group_strength_summary":
            group_summary,
        "count_strength_raw":
            count_results,
        "count_strength_summary":
            count_summary_results,
        "group_support_summary":
            support_summary,
        "category_count_summary":
            category_count_summary_results,
        "missingness_summary":
            missingness_summary_results,
        "null_summary":
            null_summary_results
    }


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":

    run_categorical_calibration()
