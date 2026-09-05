from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from candidate_generator import (
    generate_candidates
)

from scoring.engine import (
    candidate_signature,
    diversify_recommendations,
    score_all_candidates
)


# ==================================================
# CONFIGURATION
# ==================================================

RESULTS_DIR = Path(
    "evaluation/results/cross_chart_calibration_v2"
)

DEFAULT_SEEDS = range(
    10
)

DEFAULT_SAMPLE_SIZE = 500

STRENGTH_LABELS = {
    0: "ordinary",
    1: "weak",
    2: "moderate",
    3: "strong",
    4: "extreme"
}

TARGET_SIGNATURE_FAMILIES = [
    "histogram",
    "scatter",
    "line",
    "count_bar",
    "group_location"
]

CHART_FORMS = [
    "histogram",
    "scatter",
    "line",
    "count_bar",
    "mean_bar",
    "box"
]

COMPETITIONS = {
    "strong_vs_weak": {
        "focal_level": 3,
        "other_level": 1
    },
    "extreme_vs_moderate": {
        "focal_level": 4,
        "other_level": 2
    }
}


# ==================================================
# DATA STRUCTURES
# ==================================================

@dataclass(frozen=True)
class CrossChartScenario:
    """
    One synthetic multi-pattern dataset.

    Multiple independent planted relationships are
    placed into the same dataset so Visift must rank
    chart candidates from different chart families
    on one shared 0-100 scale.
    """

    name: str
    seed: int
    suite: str
    levels: dict
    dataframe: pd.DataFrame
    profile: dict


# ==================================================
# COMMON HELPERS
# ==================================================

def strength_label(
    strength_index
):
    return STRENGTH_LABELS[
        strength_index
    ]


def rng_for(
    seed,
    stream
):
    """
    Give each planted relationship an independent
    random stream.

    This reduces accidental correlation between the
    intentionally unrelated signal families.
    """

    return np.random.default_rng(
        (
            int(seed)
            + 1
        )
        * 100_003
        + int(stream)
        * 7_919
    )


def standardized(
    values
):
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


def balanced_groups(
    rng,
    n,
    labels
):
    values = np.resize(
        np.asarray(
            labels,
            dtype=object
        ),
        n
    )

    rng.shuffle(
        values
    )

    return values


def manual_profile():
    """
    Fix semantic types so this benchmark isolates
    candidate generation, chart scoring, and global
    ranking rather than semantic inference.
    """

    return {
        "column_profiles": [
            {
                "name": "time",
                "semantic_type": "temporal"
            },
            {
                "name": "temporal_value",
                "semantic_type":
                    "numeric_continuous"
            },
            {
                "name": "scatter_x",
                "semantic_type":
                    "numeric_continuous"
            },
            {
                "name": "scatter_y",
                "semantic_type":
                    "numeric_continuous"
            },
            {
                "name": "distribution_value",
                "semantic_type":
                    "numeric_continuous"
            },
            {
                "name": "group",
                "semantic_type": "categorical"
            },
            {
                "name": "group_value",
                "semantic_type":
                    "numeric_continuous"
            },
            {
                "name": "count_category",
                "semantic_type": "categorical"
            }
        ]
    }


# ==================================================
# PLANTED RELATIONSHIP GENERATORS
# ==================================================

def scatter_values(
    seed,
    level,
    n
):
    """
    Linear scatter relationship.

    Strength schedule mirrors the dedicated scatter
    calibration benchmark.
    """

    rng = rng_for(
        seed,
        1
    )

    relationship = {
        0: 0.00,
        1: 0.15,
        2: 0.30,
        3: 0.55,
        4: 0.85
    }[
        level
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
        relationship
        * x
        + np.sqrt(
            max(
                0.0,
                1.0
                - relationship ** 2
            )
        )
        * noise
    )

    return (
        x,
        y
    )


def temporal_values(
    seed,
    level,
    n
):
    """
    Linear temporal trend using the same effect-size
    schedule as the calibrated line-trend suite.
    """

    rng = rng_for(
        seed,
        2
    )

    relationship = {
        0: 0.00,
        1: 0.15,
        2: 0.30,
        3: 0.55,
        4: 0.85
    }[
        level
    ]

    time_values = np.arange(
        n,
        dtype=float
    )

    time_standardized = standardized(
        time_values
    )

    noise = rng.normal(
        0,
        1,
        n
    )

    y = (
        relationship
        * time_standardized
        + np.sqrt(
            max(
                0.0,
                1.0
                - relationship ** 2
            )
        )
        * noise
    )

    return (
        time_values,
        y
    )


def histogram_values(
    seed,
    level,
    n
):
    """
    Increasingly informative right-tailed
    distributions.

    This family intentionally combines robust shape
    asymmetry with a small amount of one-sided tail
    contamination at higher levels. It is designed
    to exercise the same broad distribution features
    used by Visift's calibrated histogram scorer.
    """

    rng = rng_for(
        seed,
        3
    )

    if level == 0:

        return rng.normal(
            0,
            1,
            n
        )

    sigma = {
        1: 0.20,
        2: 0.40,
        3: 0.65,
        4: 0.90
    }[
        level
    ]

    values = rng.lognormal(
        mean=0,
        sigma=sigma,
        size=n
    )

    outlier_rate = {
        1: 0.00,
        2: 0.01,
        3: 0.03,
        4: 0.06
    }[
        level
    ]

    outlier_multiplier = {
        1: 1.0,
        2: 4.0,
        3: 6.0,
        4: 9.0
    }[
        level
    ]

    outlier_count = int(
        round(
            n
            * outlier_rate
        )
    )

    if outlier_count > 0:

        indices = rng.choice(
            n,
            size=outlier_count,
            replace=False
        )

        values[
            indices
        ] *= outlier_multiplier

    return values


def group_location_values(
    seed,
    level,
    n
):
    """
    Four balanced categories with increasingly
    separated numeric centers.

    Both mean bar and box plots are valid views of
    this same signature, which lets the global engine
    choose which chart form should be primary.
    """

    rng = rng_for(
        seed,
        4
    )

    labels = [
        "A",
        "B",
        "C",
        "D"
    ]

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
        level
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


def count_category_values(
    seed,
    level,
    n
):
    """
    Four-category frequency imbalance using the same
    schedule as the dedicated categorical benchmark.
    """

    rng = rng_for(
        seed,
        5
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
        level
    ]

    return rng.choice(
        [
            "A",
            "B",
            "C",
            "D"
        ],
        size=n,
        p=probabilities
    )


# ==================================================
# DATASET CONSTRUCTION
# ==================================================

def build_dataset(
    seed,
    levels,
    n=DEFAULT_SAMPLE_SIZE
):
    scatter_x, scatter_y = (
        scatter_values(
            seed=seed,
            level=levels[
                "scatter"
            ],
            n=n
        )
    )

    time_values, temporal_y = (
        temporal_values(
            seed=seed,
            level=levels[
                "line"
            ],
            n=n
        )
    )

    distribution = histogram_values(
        seed=seed,
        level=levels[
            "histogram"
        ],
        n=n
    )

    groups, group_y = (
        group_location_values(
            seed=seed,
            level=levels[
                "group_location"
            ],
            n=n
        )
    )

    count_categories = (
        count_category_values(
            seed=seed,
            level=levels[
                "count_bar"
            ],
            n=n
        )
    )

    dataframe = pd.DataFrame({
        "time": time_values,
        "temporal_value": temporal_y,
        "scatter_x": scatter_x,
        "scatter_y": scatter_y,
        "distribution_value":
            distribution,
        "group": groups,
        "group_value": group_y,
        "count_category":
            count_categories
    })

    return (
        dataframe,
        manual_profile()
    )


# ==================================================
# TARGET IDENTIFICATION
# ==================================================

def is_exact_pair(
    candidate,
    x,
    y
):
    return (
        candidate.get(
            "x"
        )
        == x
        and candidate.get(
            "y"
        )
        == y
    )


def chart_form_for_candidate(
    candidate
):
    """
    Return one of the six cross-chart forms when a
    candidate is an intended planted visualization.
    """

    chart = candidate.get(
        "chart"
    )

    if (
        chart == "histogram"
        and candidate.get(
            "x"
        )
        == "distribution_value"
    ):

        return "histogram"

    if (
        chart == "scatter"
        and {
            candidate.get(
                "x"
            ),
            candidate.get(
                "y"
            )
        }
        == {
            "scatter_x",
            "scatter_y"
        }
    ):

        return "scatter"

    if (
        chart == "line"
        and is_exact_pair(
            candidate,
            "time",
            "temporal_value"
        )
    ):

        return "line"

    if (
        chart == "bar"
        and candidate.get(
            "x"
        )
        == "count_category"
        and candidate.get(
            "aggregation"
        )
        == "count"
        and "y" not in candidate
    ):

        return "count_bar"

    if (
        chart == "bar"
        and is_exact_pair(
            candidate,
            "group",
            "group_value"
        )
        and candidate.get(
            "aggregation"
        )
        == "mean"
    ):

        return "mean_bar"

    if (
        chart == "box"
        and is_exact_pair(
            candidate,
            "group",
            "group_value"
        )
    ):

        return "box"

    return None


def signature_family_for_candidate(
    candidate
):
    """
    Collapse mean-bar and box-plot alternatives for
    the same category + numeric relationship into one
    group-location signature.

    This matches Visift's recommendation
    diversification behavior.
    """

    form = chart_form_for_candidate(
        candidate
    )

    if form in {
        "mean_bar",
        "box"
    }:

        return "group_location"

    if form in {
        "histogram",
        "scatter",
        "line",
        "count_bar"
    }:

        return form

    return None


def target_signature(
    family
):
    signatures = {
        "histogram": (
            "distribution_value",
        ),
        "scatter": tuple(
            sorted([
                "scatter_x",
                "scatter_y"
            ])
        ),
        "line": tuple(
            sorted([
                "time",
                "temporal_value"
            ])
        ),
        "count_bar": (
            "count_category",
        ),
        "group_location": tuple(
            sorted([
                "group",
                "group_value"
            ])
        )
    }

    return signatures[
        family
    ]


# ==================================================
# PIPELINE EXECUTION
# ==================================================

def run_pipeline(
    dataframe,
    profile
):
    """
    Run the actual Visift candidate-generation and
    global-ranking stages.
    """

    candidates = generate_candidates(
        dataframe,
        profile
    )

    scored = score_all_candidates(
        dataframe,
        candidates,
        profile
    )

    diversified = (
        diversify_recommendations(
            scored
        )
    )

    return {
        "candidate_count":
            len(
                candidates
            ),
        "scored":
            scored,
        "diversified":
            diversified
    }


def rank_lookup(
    recommendations
):
    return {
        candidate_signature(
            candidate
        ): index
        for index, candidate
        in enumerate(
            recommendations,
            start=1
        )
    }


def score_lookup(
    recommendations
):
    return {
        candidate_signature(
            candidate
        ): candidate[
            "score"
        ]
        for candidate
        in recommendations
    }


def extract_chart_form_rows(
    scenario,
    pipeline
):
    """
    Record raw candidate scores for every intended
    chart form before diversification.
    """

    rows = []

    for index, candidate in enumerate(
        pipeline[
            "scored"
        ],
        start=1
    ):

        form = chart_form_for_candidate(
            candidate
        )

        if form is None:
            continue

        planted_family = (
            "group_location"
            if form in {
                "mean_bar",
                "box"
            }
            else form
        )

        rows.append({
            "scenario":
                scenario.name,
            "suite":
                scenario.suite,
            "seed":
                scenario.seed,
            "chart_form":
                form,
            "planted_family":
                planted_family,
            "strength_index":
                scenario.levels[
                    planted_family
                ],
            "level":
                strength_label(
                    scenario.levels[
                        planted_family
                    ]
                ),
            "score":
                candidate[
                    "score"
                ],
            "signal":
                candidate[
                    "components"
                ][
                    "signal"
                ],
            "visualization_quality":
                candidate[
                    "components"
                ][
                    "visualization_quality"
                ],
            "raw_global_rank":
                index
        })

    return rows


def extract_signature_rows(
    scenario,
    pipeline
):
    """
    Record globally diversified target-signature
    scores and ranks.
    """

    diversified = pipeline[
        "diversified"
    ]

    diversified_rank = (
        rank_lookup(
            diversified
        )
    )

    diversified_score = (
        score_lookup(
            diversified
        )
    )

    rows = []

    for family in (
        TARGET_SIGNATURE_FAMILIES
    ):

        signature = target_signature(
            family
        )

        rank = diversified_rank.get(
            signature
        )

        score = diversified_score.get(
            signature
        )

        primary_chart = None

        if rank is not None:

            primary_chart = (
                diversified[
                    rank - 1
                ][
                    "chart"
                ]
            )

        rows.append({
            "scenario":
                scenario.name,
            "suite":
                scenario.suite,
            "seed":
                scenario.seed,
            "family":
                family,
            "strength_index":
                scenario.levels[
                    family
                ],
            "level":
                strength_label(
                    scenario.levels[
                        family
                    ]
                ),
            "score":
                score,
            "diversified_rank":
                rank,
            "primary_chart":
                primary_chart
        })

    return rows


def distractor_metrics(
    scenario,
    pipeline
):
    """
    Compare planted target signatures against all
    other diversified recommendations.
    """

    target_signatures = {
        target_signature(
            family
        )
        for family in (
            TARGET_SIGNATURE_FAMILIES
        )
    }

    diversified = pipeline[
        "diversified"
    ]

    target_candidates = [
        candidate
        for candidate
        in diversified
        if candidate_signature(
            candidate
        )
        in target_signatures
    ]

    distractors = [
        candidate
        for candidate
        in diversified
        if candidate_signature(
            candidate
        )
        not in target_signatures
    ]

    target_scores = [
        candidate[
            "score"
        ]
        for candidate
        in target_candidates
    ]

    distractor_scores = [
        candidate[
            "score"
        ]
        for candidate
        in distractors
    ]

    weakest_target = (
        min(
            target_scores
        )
        if target_scores
        else np.nan
    )

    best_distractor = (
        max(
            distractor_scores
        )
        if distractor_scores
        else np.nan
    )

    targets_in_top_5 = sum(
        candidate_signature(
            candidate
        )
        in target_signatures
        for candidate
        in diversified[
            :5
        ]
    )

    return {
        "scenario":
            scenario.name,
        "suite":
            scenario.suite,
        "seed":
            scenario.seed,
        "candidate_count":
            pipeline[
                "candidate_count"
            ],
        "diversified_count":
            len(
                diversified
            ),
        "targets_in_top_5":
            targets_in_top_5,
        "weakest_target_score":
            weakest_target,
        "best_distractor_score":
            best_distractor,
        "target_distractor_margin": (
            weakest_target
            - best_distractor
            if (
                np.isfinite(
                    weakest_target
                )
                and np.isfinite(
                    best_distractor
                )
            )
            else np.nan
        )
    }


# ==================================================
# ALIGNED-STRENGTH SUITE
# ==================================================

def aligned_scenario(
    seed,
    level,
    n=DEFAULT_SAMPLE_SIZE
):
    levels = {
        family: level
        for family in (
            TARGET_SIGNATURE_FAMILIES
        )
    }

    dataframe, profile = (
        build_dataset(
            seed=seed,
            levels=levels,
            n=n
        )
    )

    return CrossChartScenario(
        name=(
            "cross_chart_aligned_"
            f"{strength_label(level)}_"
            f"seed_{seed}"
        ),
        seed=seed,
        suite="aligned",
        levels=levels,
        dataframe=dataframe,
        profile=profile
    )


def build_aligned_suite(
    seeds=None,
    n=DEFAULT_SAMPLE_SIZE
):
    if seeds is None:
        seeds = DEFAULT_SEEDS

    scenarios = []

    for seed in seeds:

        for level in range(
            5
        ):

            scenarios.append(
                aligned_scenario(
                    seed=seed,
                    level=level,
                    n=n
                )
            )

    return scenarios


# ==================================================
# MIXED-STRENGTH COMPETITION SUITE
# ==================================================

def mixed_scenario(
    seed,
    focal_family,
    competition,
    n=DEFAULT_SAMPLE_SIZE
):
    specification = (
        COMPETITIONS[
            competition
        ]
    )

    levels = {
        family:
            specification[
                "other_level"
            ]
        for family in (
            TARGET_SIGNATURE_FAMILIES
        )
    }

    levels[
        focal_family
    ] = specification[
        "focal_level"
    ]

    dataframe, profile = (
        build_dataset(
            seed=seed,
            levels=levels,
            n=n
        )
    )

    return CrossChartScenario(
        name=(
            "cross_chart_"
            f"{competition}_"
            f"{focal_family}_"
            f"seed_{seed}"
        ),
        seed=seed,
        suite=competition,
        levels=levels,
        dataframe=dataframe,
        profile=profile
    )


def build_mixed_suite(
    seeds=None,
    n=DEFAULT_SAMPLE_SIZE
):
    if seeds is None:
        seeds = DEFAULT_SEEDS

    scenarios = []

    for competition in (
        COMPETITIONS
    ):

        for focal_family in (
            TARGET_SIGNATURE_FAMILIES
        ):

            for seed in seeds:

                scenarios.append(
                    mixed_scenario(
                        seed=seed,
                        focal_family=
                            focal_family,
                        competition=
                            competition,
                        n=n
                    )
                )

    return scenarios


# ==================================================
# SUMMARY HELPERS
# ==================================================

def percentile_10(
    values
):
    return values.quantile(
        0.10
    )


def percentile_90(
    values
):
    return values.quantile(
        0.90
    )


def chart_form_level_summary(
    rows
):
    return (
        rows
        .groupby(
            [
                "chart_form",
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
            median_quality=(
                "visualization_quality",
                "median"
            ),
            median_raw_rank=(
                "raw_global_rank",
                "median"
            )
        )
        .sort_values(
            [
                "chart_form",
                "strength_index"
            ]
        )
        .reset_index(
            drop=True
        )
    )


def signature_level_summary(
    rows
):
    return (
        rows
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
            median_rank=(
                "diversified_rank",
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


def tier_spread_summary(
    signature_summary
):
    """
    Show how far apart the median score scales are
    across target signature families at the same
    qualitative strength level.
    """

    rows = []

    for level in range(
        5
    ):

        subset = (
            signature_summary[
                signature_summary[
                    "strength_index"
                ]
                == level
            ]
        )

        if subset.empty:
            continue

        minimum_row = subset.loc[
            subset[
                "median_score"
            ].idxmin()
        ]

        maximum_row = subset.loc[
            subset[
                "median_score"
            ].idxmax()
        ]

        rows.append({
            "strength_index":
                level,
            "level":
                strength_label(
                    level
                ),
            "minimum_family":
                minimum_row[
                    "family"
                ],
            "minimum_median_score":
                minimum_row[
                    "median_score"
                ],
            "maximum_family":
                maximum_row[
                    "family"
                ],
            "maximum_median_score":
                maximum_row[
                    "median_score"
                ],
            "cross_family_spread": (
                maximum_row[
                    "median_score"
                ]
                - minimum_row[
                    "median_score"
                ]
            )
        })

    return pd.DataFrame(
        rows
    )


def aligned_distractor_summary(
    rows
):
    aligned = (
        rows[
            rows[
                "suite"
            ]
            == "aligned"
        ]
        .copy()
    )

    # Recover level from the scenario name by
    # grouping against the aligned signature results
    # later; here all five planted targets share one
    # level, so use target scores from each scenario.
    return (
        aligned
        .groupby(
            "scenario",
            as_index=False
        )
        .agg(
            seed=(
                "seed",
                "first"
            ),
            candidate_count=(
                "candidate_count",
                "first"
            ),
            diversified_count=(
                "diversified_count",
                "first"
            ),
            targets_in_top_5=(
                "targets_in_top_5",
                "first"
            ),
            weakest_target_score=(
                "weakest_target_score",
                "first"
            ),
            best_distractor_score=(
                "best_distractor_score",
                "first"
            ),
            target_distractor_margin=(
                "target_distractor_margin",
                "first"
            )
        )
    )


def build_mixed_competition_rows(
    scenario,
    signature_rows,
    pipeline
):
    """
    Evaluate whether the intentionally stronger
    planted signature wins against weaker planted
    signatures and unrelated distractors.
    """

    specification = (
        COMPETITIONS[
            scenario.suite
        ]
    )

    focal_family = next(
        family
        for family in (
            TARGET_SIGNATURE_FAMILIES
        )
        if (
            scenario.levels[
                family
            ]
            == specification[
                "focal_level"
            ]
        )
    )

    frame = pd.DataFrame(
        signature_rows
    )

    focal_row = (
        frame[
            frame[
                "family"
            ]
            == focal_family
        ]
        .iloc[0]
    )

    other_rows = (
        frame[
            frame[
                "family"
            ]
            != focal_family
        ]
    )

    best_other_score = (
        other_rows[
            "score"
        ]
        .max()
    )

    target_signatures = {
        target_signature(
            family
        )
        for family in (
            TARGET_SIGNATURE_FAMILIES
        )
    }

    diversified = pipeline[
        "diversified"
    ]

    top_signature = (
        candidate_signature(
            diversified[0]
        )
        if diversified
        else None
    )

    focal_signature = (
        target_signature(
            focal_family
        )
    )

    distractors_above_focal = sum(
        (
            candidate_signature(
                candidate
            )
            not in target_signatures
        )
        and (
            candidate[
                "score"
            ]
            > focal_row[
                "score"
            ]
        )
        for candidate
        in diversified
    )

    return {
        "scenario":
            scenario.name,
        "competition":
            scenario.suite,
        "seed":
            scenario.seed,
        "focal_family":
            focal_family,
        "focal_level":
            specification[
                "focal_level"
            ],
        "other_level":
            specification[
                "other_level"
            ],
        "focal_score":
            focal_row[
                "score"
            ],
        "best_other_target_score":
            best_other_score,
        "margin_over_best_other": (
            focal_row[
                "score"
            ]
            - best_other_score
        ),
        "focal_best_target": bool(
            focal_row[
                "score"
            ]
            > best_other_score
        ),
        "focal_global_rank":
            focal_row[
                "diversified_rank"
            ],
        "focal_global_top_1": bool(
            top_signature
            == focal_signature
        ),
        "distractors_above_focal":
            distractors_above_focal
    }


def mixed_competition_summary(
    rows
):
    return (
        rows
        .groupby(
            [
                "competition",
                "focal_family",
                "focal_level",
                "other_level"
            ],
            as_index=False
        )
        .agg(
            scenarios=(
                "focal_score",
                "size"
            ),
            median_focal_score=(
                "focal_score",
                "median"
            ),
            median_best_other_score=(
                "best_other_target_score",
                "median"
            ),
            median_margin=(
                "margin_over_best_other",
                "median"
            ),
            focal_best_target_pct=(
                "focal_best_target",
                lambda values:
                values.mean()
                * 100
            ),
            focal_global_top_1_pct=(
                "focal_global_top_1",
                lambda values:
                values.mean()
                * 100
            ),
            median_global_rank=(
                "focal_global_rank",
                "median"
            ),
            mean_distractors_above=(
                "distractors_above_focal",
                "mean"
            )
        )
        .sort_values(
            [
                "competition",
                "focal_family"
            ]
        )
        .reset_index(
            drop=True
        )
    )


def recommendation_band(
    score
):
    if score >= 80:
        return "Excellent"

    if score >= 65:
        return "Strong"

    if score >= 50:
        return "Moderate"

    if score >= 35:
        return "Weak"

    return "Very weak"


def recommendation_band_summary(
    signature_summary
):
    output = (
        signature_summary[
            [
                "family",
                "strength_index",
                "level",
                "median_score"
            ]
        ]
        .copy()
    )

    output[
        "median_recommendation_band"
    ] = output[
        "median_score"
    ].map(
        recommendation_band
    )

    return output


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
    aligned_chart_summary,
    aligned_signature_summary,
    tier_spread,
    band_summary,
    mixed_summary,
    aligned_distractors,
    scenario_count,
    candidate_count
):
    print()
    print(
        "=" * 118
    )
    print(
        "VISIFT CROSS-CHART SCORE CALIBRATION"
    )
    print(
        "=" * 118
    )
    print()

    print(
        f"Multi-pattern datasets tested:    "
        f"{scenario_count}"
    )

    print(
        f"Visualization candidates scored:  "
        f"{candidate_count}"
    )

    print(
        f"Random seeds:                     "
        f"{len(list(DEFAULT_SEEDS))}"
    )

    print(
        f"Observations per dataset:          "
        f"{DEFAULT_SAMPLE_SIZE}"
    )

    print()

    print(
        "PURPOSE"
    )
    print(
        "-" * 118
    )

    print(
        "This benchmark tests whether Visift's chart-"
        "specific 0-100 scores remain comparable after "
        "candidate generation, global scoring, and "
        "recommendation diversification. Multiple "
        "independent planted patterns compete inside "
        "the same dataset alongside many unintended "
        "cross-variable distractor candidates."
    )

    print()

    print(
        "ALIGNED-STRENGTH CHART-FORM SCORES"
    )
    print(
        "-" * 118
    )

    printable = format_numeric_columns(
        aligned_chart_summary,
        [
            "median_score",
            "mean_score",
            "p10_score",
            "p90_score",
            "median_signal",
            "median_quality",
            "median_raw_rank"
        ]
    )

    print(
        printable.to_string(
            index=False
        )
    )

    print()

    print(
        "ALIGNED-STRENGTH DIVERSIFIED SIGNATURE SCORES"
    )
    print(
        "-" * 118
    )

    printable = format_numeric_columns(
        aligned_signature_summary,
        [
            "median_score",
            "mean_score",
            "p10_score",
            "p90_score",
            "median_rank"
        ]
    )

    print(
        printable.to_string(
            index=False
        )
    )

    print()

    print(
        "CROSS-FAMILY SCORE SPREAD BY STRENGTH TIER"
    )
    print(
        "-" * 118
    )

    printable = format_numeric_columns(
        tier_spread,
        [
            "minimum_median_score",
            "maximum_median_score",
            "cross_family_spread"
        ]
    )

    print(
        printable.to_string(
            index=False
        )
    )

    print()

    print(
        "MEDIAN RECOMMENDATION BANDS"
    )
    print(
        "-" * 118
    )

    printable = format_numeric_columns(
        band_summary,
        [
            "median_score"
        ]
    )

    print(
        printable.to_string(
            index=False
        )
    )

    print()

    print(
        "MIXED-STRENGTH COMPETITIONS"
    )
    print(
        "-" * 118
    )

    printable = format_numeric_columns(
        mixed_summary,
        [
            "median_focal_score",
            "median_best_other_score",
            "median_margin",
            "median_global_rank",
            "mean_distractors_above"
        ]
    )

    printable = format_percentage_columns(
        printable,
        [
            "focal_best_target_pct",
            "focal_global_top_1_pct"
        ]
    )

    print(
        printable.to_string(
            index=False
        )
    )

    print()

    print(
        "ALIGNED-SUITE DISTRACTOR PRESSURE"
    )
    print(
        "-" * 118
    )

    print(
        "Targets in top 5:"
    )

    print(
        (
            aligned_distractors[
                "targets_in_top_5"
            ]
            .value_counts()
            .sort_index()
            .to_string()
        )
    )

    print()

    print(
        "Median weakest-target score:       "
        f"{aligned_distractors['weakest_target_score'].median():.2f}"
    )

    print(
        "Median best-distractor score:      "
        f"{aligned_distractors['best_distractor_score'].median():.2f}"
    )

    print(
        "Median target-distractor margin:   "
        f"{aligned_distractors['target_distractor_margin'].median():.2f}"
    )

    print()

    print(
        "INTERPRETATION GUIDE"
    )
    print(
        "-" * 118
    )

    print(
        "Look for:"
    )

    print(
        "  1. Ordinary relationships staying near the "
        "bottom of the recommendation scale across "
        "all chart families."
    )

    print(
        "  2. Similar qualitative tiers producing "
        "reasonably comparable scores across chart "
        "families, without one family dominating every "
        "tier."
    )

    print(
        "  3. Strong-vs-weak and extreme-vs-moderate "
        "competitions being won by the intentionally "
        "stronger planted signature regardless of chart "
        "family."
    )

    print(
        "  4. Focal signatures remaining near global "
        "rank #1 even when many unrelated chart "
        "candidates are generated."
    )

    print(
        "  5. Mean-bar and box-plot scores being "
        "interpreted as alternate views of the same "
        "category + numeric relationship rather than "
        "double-counted recommendations."
    )

    print(
        "  6. Large cross-family score spreads or a "
        "systematic focal-family failure indicating a "
        "global calibration mismatch worth tuning."
    )

    print()

    print(
        "=" * 118
    )

    print(
        f"Raw results saved to {RESULTS_DIR}/"
    )

    print(
        "=" * 118
    )

    print()


# ==================================================
# MAIN BENCHMARK
# ==================================================

def run_cross_chart_calibration(
    seeds=None,
    n=DEFAULT_SAMPLE_SIZE
):
    if seeds is None:
        seeds = DEFAULT_SEEDS

    # -----------------------------------
    # Aligned strength
    # -----------------------------------

    aligned_scenarios = (
        build_aligned_suite(
            seeds=seeds,
            n=n
        )
    )

    aligned_chart_rows = []
    aligned_signature_rows = []
    distractor_rows = []

    total_candidates = 0

    for index, scenario in enumerate(
        aligned_scenarios,
        start=1
    ):

        print(
            "[aligned "
            f"{index:03d}/"
            f"{len(aligned_scenarios):03d}] "
            f"{scenario.name}"
        )

        pipeline = run_pipeline(
            scenario.dataframe,
            scenario.profile
        )

        total_candidates += pipeline[
            "candidate_count"
        ]

        aligned_chart_rows.extend(
            extract_chart_form_rows(
                scenario,
                pipeline
            )
        )

        aligned_signature_rows.extend(
            extract_signature_rows(
                scenario,
                pipeline
            )
        )

        distractor_rows.append(
            distractor_metrics(
                scenario,
                pipeline
            )
        )

    aligned_chart_raw = pd.DataFrame(
        aligned_chart_rows
    )

    aligned_signature_raw = (
        pd.DataFrame(
            aligned_signature_rows
        )
    )

    distractor_raw = pd.DataFrame(
        distractor_rows
    )

    # -----------------------------------
    # Mixed strength
    # -----------------------------------

    mixed_scenarios = (
        build_mixed_suite(
            seeds=seeds,
            n=n
        )
    )

    mixed_rows = []
    mixed_signature_rows = []

    for index, scenario in enumerate(
        mixed_scenarios,
        start=1
    ):

        print(
            "[mixed "
            f"{index:03d}/"
            f"{len(mixed_scenarios):03d}] "
            f"{scenario.name}"
        )

        pipeline = run_pipeline(
            scenario.dataframe,
            scenario.profile
        )

        total_candidates += pipeline[
            "candidate_count"
        ]

        signature_rows = (
            extract_signature_rows(
                scenario,
                pipeline
            )
        )

        mixed_signature_rows.extend(
            signature_rows
        )

        mixed_rows.append(
            build_mixed_competition_rows(
                scenario,
                signature_rows,
                pipeline
            )
        )

    mixed_raw = pd.DataFrame(
        mixed_rows
    )

    mixed_signature_raw = pd.DataFrame(
        mixed_signature_rows
    )

    # -----------------------------------
    # Summaries
    # -----------------------------------

    aligned_chart_summary = (
        chart_form_level_summary(
            aligned_chart_raw
        )
    )

    aligned_signature_summary = (
        signature_level_summary(
            aligned_signature_raw
        )
    )

    tier_spread = (
        tier_spread_summary(
            aligned_signature_summary
        )
    )

    band_summary = (
        recommendation_band_summary(
            aligned_signature_summary
        )
    )

    mixed_summary = (
        mixed_competition_summary(
            mixed_raw
        )
    )

    aligned_distractors = (
        aligned_distractor_summary(
            distractor_raw
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
        "aligned_chart_form_raw.csv":
            aligned_chart_raw,
        "aligned_chart_form_summary.csv":
            aligned_chart_summary,
        "aligned_signature_raw.csv":
            aligned_signature_raw,
        "aligned_signature_summary.csv":
            aligned_signature_summary,
        "tier_spread_summary.csv":
            tier_spread,
        "recommendation_band_summary.csv":
            band_summary,
        "aligned_distractor_raw.csv":
            distractor_raw,
        "mixed_signature_raw.csv":
            mixed_signature_raw,
        "mixed_competition_raw.csv":
            mixed_raw,
        "mixed_competition_summary.csv":
            mixed_summary
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

    scenario_count = (
        len(
            aligned_scenarios
        )
        + len(
            mixed_scenarios
        )
    )

    print_report(
        aligned_chart_summary=
            aligned_chart_summary,
        aligned_signature_summary=
            aligned_signature_summary,
        tier_spread=
            tier_spread,
        band_summary=
            band_summary,
        mixed_summary=
            mixed_summary,
        aligned_distractors=
            aligned_distractors,
        scenario_count=
            scenario_count,
        candidate_count=
            total_candidates
    )

    return {
        "aligned_chart_form_raw":
            aligned_chart_raw,
        "aligned_chart_form_summary":
            aligned_chart_summary,
        "aligned_signature_raw":
            aligned_signature_raw,
        "aligned_signature_summary":
            aligned_signature_summary,
        "tier_spread_summary":
            tier_spread,
        "recommendation_band_summary":
            band_summary,
        "aligned_distractor_raw":
            distractor_raw,
        "mixed_signature_raw":
            mixed_signature_raw,
        "mixed_competition_raw":
            mixed_raw,
        "mixed_competition_summary":
            mixed_summary
    }


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":

    run_cross_chart_calibration()
