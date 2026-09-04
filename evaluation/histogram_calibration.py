from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scoring.histogram import (
    distribution_signal_score,
    score_histogram
)


# ==================================================
# CONFIGURATION
# ==================================================

RESULTS_DIR = Path(
    "evaluation/results/histogram_calibration_v1"
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
# DATA STRUCTURE
# ==================================================

@dataclass(frozen=True)
class HistogramCalibrationScenario:
    """
    One controlled histogram calibration scenario.

    Unlike the hardened ranking benchmark, these
    scenarios isolate the histogram scorer so we can
    study how recommendation scores change as a known
    distribution pattern becomes stronger.
    """

    name: str
    family: str
    level: str
    strength_index: int
    seed: int
    description: str
    dataframe: pd.DataFrame
    target_column: str


# ==================================================
# COMMON HELPERS
# ==================================================

def calibration_dataframe(
    values,
    column_name="calibration_measure"
):
    """
    Build a one-column dataframe for isolated
    histogram scoring.
    """

    return pd.DataFrame({
        column_name: values
    })


def manual_histogram_profile(
    column_name
):
    """
    Create the minimal profile required by
    score_histogram().

    The hardened benchmark already tests semantic
    inference and candidate generation. This
    calibration benchmark intentionally isolates
    score behavior, so semantic type is fixed to
    numeric_continuous.
    """

    return {
        "column_profiles": [
            {
                "name": column_name,
                "semantic_type":
                    "numeric_continuous"
            }
        ]
    }


def strength_label(
    strength_index
):
    return STRENGTH_LABELS[
        strength_index
    ]


# ==================================================
# SKEWNESS FAMILY
# ==================================================

def skewness_values(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE
):
    """
    Generate increasingly right-skewed
    distributions.

    strength 0:
        approximately symmetric normal data

    strengths 1-4:
        lognormal data with progressively larger
        shape parameters
    """

    rng = np.random.default_rng(
        seed
    )

    if strength_index == 0:

        return rng.normal(
            50,
            10,
            n
        )

    sigma_by_strength = {
        1: 0.20,
        2: 0.40,
        3: 0.65,
        4: 0.90
    }

    sigma = sigma_by_strength[
        strength_index
    ]

    return rng.lognormal(
        mean=3.5,
        sigma=sigma,
        size=n
    )


def skewness_scenario(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE
):
    level = strength_label(
        strength_index
    )

    values = skewness_values(
        seed=seed,
        strength_index=strength_index,
        n=n
    )

    return HistogramCalibrationScenario(
        name=(
            f"hist_skewness_{level}_"
            f"seed_{seed}"
        ),
        family="skewness",
        level=level,
        strength_index=strength_index,
        seed=seed,
        description=(
            "Controlled right-skew calibration "
            f"scenario at {level} strength."
        ),
        dataframe=calibration_dataframe(
            values
        ),
        target_column="calibration_measure"
    )


# ==================================================
# SYMMETRIC OUTLIER FAMILY
# ==================================================

def symmetric_outlier_values(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE
):
    """
    Generate a roughly symmetric distribution
    containing increasing proportions of extreme
    observations.

    The contaminated observations are two-sided so
    this family primarily tests outlier sensitivity
    rather than directional skew.
    """

    rng = np.random.default_rng(
        seed
    )

    values = rng.normal(
        50,
        8,
        n
    )

    rate_by_strength = {
        0: 0.00,
        1: 0.01,
        2: 0.03,
        3: 0.06,
        4: 0.10
    }

    rate = rate_by_strength[
        strength_index
    ]

    if rate <= 0:
        return values

    # Generate one deterministic contamination order
    # per seed. Higher-strength scenarios therefore
    # contain the lower-strength contaminated rows as
    # a subset, improving paired comparisons.
    contamination_order = (
        rng.permutation(
            n
        )
    )

    signs = rng.choice(
        [-1, 1],
        size=n
    )

    magnitudes = rng.uniform(
        45,
        70,
        size=n
    )

    outlier_count = max(
        1,
        int(
            n * rate
        )
    )

    indices = contamination_order[
        :outlier_count
    ]

    values[
        indices
    ] += (
        signs[
            indices
        ]
        * magnitudes[
            indices
        ]
    )

    return values


def symmetric_outlier_scenario(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE
):
    level = strength_label(
        strength_index
    )

    values = symmetric_outlier_values(
        seed=seed,
        strength_index=strength_index,
        n=n
    )

    return HistogramCalibrationScenario(
        name=(
            f"hist_outliers_{level}_"
            f"seed_{seed}"
        ),
        family="symmetric_outliers",
        level=level,
        strength_index=strength_index,
        seed=seed,
        description=(
            "Controlled two-sided outlier "
            f"scenario at {level} strength."
        ),
        dataframe=calibration_dataframe(
            values
        ),
        target_column="calibration_measure"
    )


# ==================================================
# COMBINED RIGHT-TAIL FAMILY
# ==================================================

def combined_tail_values(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE
):
    """
    Generate distributions where skewness, outlier
    prevalence, and tail asymmetry occur together.

    This family is particularly important because
    the current histogram scorer combines all three
    signals additively. Real-world long-tailed
    variables can therefore contribute overlapping
    evidence multiple times.
    """

    rng = np.random.default_rng(
        seed
    )

    if strength_index == 0:

        return rng.normal(
            50,
            10,
            n
        )

    sigma_by_strength = {
        1: 0.20,
        2: 0.40,
        3: 0.65,
        4: 0.90
    }

    rate_by_strength = {
        1: 0.01,
        2: 0.03,
        3: 0.06,
        4: 0.10
    }

    sigma = sigma_by_strength[
        strength_index
    ]

    rate = rate_by_strength[
        strength_index
    ]

    values = rng.lognormal(
        mean=3.5,
        sigma=sigma,
        size=n
    )

    contamination_order = (
        rng.permutation(
            n
        )
    )

    outlier_count = max(
        1,
        int(
            n * rate
        )
    )

    indices = contamination_order[
        :outlier_count
    ]

    scale = float(
        np.std(
            values
        )
    )

    # One-sided contamination mimics real-world
    # variables such as durations, transaction sizes,
    # or prices with a very long upper tail.
    extra = rng.uniform(
        5,
        10,
        size=outlier_count
    )

    values[
        indices
    ] += (
        extra
        * scale
    )

    return values


def combined_tail_scenario(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE
):
    level = strength_label(
        strength_index
    )

    values = combined_tail_values(
        seed=seed,
        strength_index=strength_index,
        n=n
    )

    return HistogramCalibrationScenario(
        name=(
            f"hist_combined_tail_{level}_"
            f"seed_{seed}"
        ),
        family="combined_right_tail",
        level=level,
        strength_index=strength_index,
        seed=seed,
        description=(
            "Controlled long-right-tail scenario "
            "combining skewness, outliers, and tail "
            f"asymmetry at {level} strength."
        ),
        dataframe=calibration_dataframe(
            values
        ),
        target_column="calibration_measure"
    )


# ==================================================
# BIMODALITY FAMILY
# ==================================================

def bimodal_values(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE
):
    """
    Generate increasingly separated symmetric
    two-component mixtures.

    Strong bimodality can be highly informative in a
    histogram even when skewness and outlier rates are
    low. This family intentionally tests a pattern the
    current histogram scorer does not explicitly model.
    """

    rng = np.random.default_rng(
        seed
    )

    separation_by_strength = {
        0: 0.0,
        1: 1.5,
        2: 3.0,
        3: 5.0,
        4: 7.0
    }

    separation = (
        separation_by_strength[
            strength_index
        ]
    )

    components = rng.integers(
        0,
        2,
        n
    )

    noise = rng.normal(
        0,
        1,
        n
    )

    means = np.where(
        components == 0,
        -separation / 2,
        separation / 2
    )

    values = (
        means
        + noise
    )

    return values


def bimodality_scenario(
    seed,
    strength_index,
    n=DEFAULT_SAMPLE_SIZE
):
    level = strength_label(
        strength_index
    )

    values = bimodal_values(
        seed=seed,
        strength_index=strength_index,
        n=n
    )

    return HistogramCalibrationScenario(
        name=(
            f"hist_bimodality_{level}_"
            f"seed_{seed}"
        ),
        family="bimodality",
        level=level,
        strength_index=strength_index,
        seed=seed,
        description=(
            "Controlled symmetric bimodal mixture "
            f"at {level} strength."
        ),
        dataframe=calibration_dataframe(
            values
        ),
        target_column="calibration_measure"
    )


# ==================================================
# CALIBRATION SUITE
# ==================================================

def build_histogram_calibration_suite(
    seeds=None,
    n=DEFAULT_SAMPLE_SIZE
):
    """
    Build the histogram calibration suite.

    Default configuration:

        4 signal families
        5 strength levels per family
        25 random seeds
        500 total scenarios

    Families:

        skewness
        symmetric outliers
        combined right-tail irregularity
        bimodality
    """

    if seeds is None:

        seeds = (
            DEFAULT_SEEDS
        )

    scenarios = []

    for seed in seeds:

        for strength_index in range(
            5
        ):

            scenarios.extend([
                skewness_scenario(
                    seed=seed,
                    strength_index=
                        strength_index,
                    n=n
                ),
                symmetric_outlier_scenario(
                    seed=seed,
                    strength_index=
                        strength_index,
                    n=n
                ),
                combined_tail_scenario(
                    seed=seed,
                    strength_index=
                        strength_index,
                    n=n
                ),
                bimodality_scenario(
                    seed=seed,
                    strength_index=
                        strength_index,
                    n=n
                )
            ])

    return scenarios


# ==================================================
# SCORING
# ==================================================

def run_scenario(
    scenario
):
    """
    Score one isolated histogram scenario.
    """

    df = scenario.dataframe

    column = (
        scenario.target_column
    )

    candidate = {
        "chart": "histogram",
        "x": column
    }

    profile = (
        manual_histogram_profile(
            column
        )
    )

    result = score_histogram(
        df=df,
        candidate=candidate,
        profile=profile
    )

    clean = pd.to_numeric(
        df[column],
        errors="coerce"
    ).dropna()

    distribution = (
        distribution_signal_score(
            clean
        )
    )

    return {
        "name": scenario.name,
        "family": scenario.family,
        "level": scenario.level,
        "strength_index":
            scenario.strength_index,
        "seed": scenario.seed,
        "observations": len(clean),

        "score": result[
            "score"
        ],

        "signal": result[
            "components"
        ][
            "signal"
        ],

        "visualization_quality": result[
            "components"
        ][
            "visualization_quality"
        ],

        "skewness": distribution[
            "skewness"
        ],

        "skew_score": distribution[
            "skew_score"
        ],

        "outlier_rate": distribution[
            "outlier_rate"
        ],

        "outlier_score": distribution[
            "outlier_score"
        ],

        "tail_asymmetry": distribution[
            "tail_asymmetry"
        ],

        "tail_asymmetry_score": (
            distribution[
                "tail_asymmetry_score"
            ]
        )
    }


# ==================================================
# SUMMARIES
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


def level_summary(
    results
):
    """
    Summarize score behavior at every family and
    strength level.
    """

    summary = (
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
            median_skewness=(
                "skewness",
                "median"
            ),
            median_outlier_rate=(
                "outlier_rate",
                "median"
            ),
            median_tail_asymmetry=(
                "tail_asymmetry",
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

    summary[
        "median_outlier_pct"
    ] = (
        summary[
            "median_outlier_rate"
        ]
        * 100
    )

    summary = summary.drop(
        columns=[
            "median_outlier_rate"
        ]
    )

    return summary


def adjacent_monotonicity(
    results
):
    """
    Compare each strength level with the next
    stronger level using matched random seeds.

    The paired comparison is more informative than
    comparing only global means because it measures
    whether the scorer reliably increases when the
    planted pattern becomes stronger.
    """

    rows = []

    families = sorted(
        results[
            "family"
        ].unique()
    )

    for family in families:

        family_results = (
            results[
                results[
                    "family"
                ]
                == family
            ]
            .copy()
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

            stronger_higher_pct = (
                (
                    paired[
                        "delta"
                    ]
                    > 0
                )
                .mean()
                * 100
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
                    len(paired),
                "median_score_delta":
                    paired[
                        "delta"
                    ].median(),
                "mean_score_delta":
                    paired[
                        "delta"
                    ].mean(),
                "stronger_higher_pct":
                    stronger_higher_pct
            })

    return pd.DataFrame(
        rows
    )


def full_monotonicity_summary(
    results
):
    """
    Measure the percentage of seeds for which all
    five strength levels increase strictly in order.
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
                values="score",
                aggfunc="first"
            )
        )

        required_columns = [
            0,
            1,
            2,
            3,
            4
        ]

        if not all(
            column in pivot.columns
            for column in required_columns
        ):

            monotonic_pct = (
                float("nan")
            )

        else:

            ordered = (
                pivot[
                    required_columns
                ]
                .dropna()
            )

            if ordered.empty:

                monotonic_pct = (
                    float("nan")
                )

            else:

                monotonic = (
                    (
                        ordered[
                            0
                        ]
                        < ordered[
                            1
                        ]
                    )
                    & (
                        ordered[
                            1
                        ]
                        < ordered[
                            2
                        ]
                    )
                    & (
                        ordered[
                            2
                        ]
                        < ordered[
                            3
                        ]
                    )
                    & (
                        ordered[
                            3
                        ]
                        < ordered[
                            4
                        ]
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


# ==================================================
# PRINTING
# ==================================================

def printable_level_summary(
    summary
):
    output = (
        summary.copy()
    )

    numeric_columns = [
        "median_score",
        "mean_score",
        "p10_score",
        "p90_score",
        "median_signal",
        "median_skewness",
        "median_outlier_pct",
        "median_tail_asymmetry"
    ]

    for column in numeric_columns:

        output[
            column
        ] = output[
            column
        ].map(
            lambda value:
            f"{value:.2f}"
        )

    return output


def printable_monotonicity(
    monotonicity
):
    output = (
        monotonicity.copy()
    )

    for column in [
        "median_score_delta",
        "mean_score_delta"
    ]:

        output[
            column
        ] = output[
            column
        ].map(
            lambda value:
            f"{value:.2f}"
        )

    output[
        "stronger_higher_pct"
    ] = output[
        "stronger_higher_pct"
    ].map(
        lambda value:
        f"{value:.1f}%"
    )

    return output


def printable_full_monotonicity(
    summary
):
    output = (
        summary.copy()
    )

    output[
        "fully_monotonic_seed_pct"
    ] = output[
        "fully_monotonic_seed_pct"
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
    results,
    level_results,
    monotonicity,
    full_monotonicity
):
    """
    Print the histogram calibration report.
    """

    print()
    print(
        "=" * 96
    )
    print(
        "VISIFT HISTOGRAM SCORE CALIBRATION"
    )
    print(
        "=" * 96
    )
    print()

    print(
        f"Scenarios tested:                 "
        f"{len(results)}"
    )

    print(
        f"Signal families:                  "
        f"{results['family'].nunique()}"
    )

    print(
        f"Random seeds:                     "
        f"{results['seed'].nunique()}"
    )

    print(
        f"Observations per scenario:        "
        f"{int(results['observations'].median())}"
    )

    print()

    print(
        "PURPOSE"
    )
    print(
        "-" * 96
    )
    print(
        "This benchmark does not test ranking accuracy. "
        "It tests whether histogram scores behave "
        "sensibly as planted distribution structure "
        "becomes stronger."
    )
    print()
    print(
        "No hard numeric target bands are enforced in "
        "this baseline run. First inspect monotonicity, "
        "score separation, saturation, and blind spots; "
        "then tune the scorer against those findings."
    )
    print()

    print(
        "LEVEL SUMMARY"
    )
    print(
        "-" * 96
    )

    print(
        printable_level_summary(
            level_results
        ).to_string(
            index=False
        )
    )

    print()

    print(
        "ADJACENT STRENGTH COMPARISONS"
    )
    print(
        "-" * 96
    )

    print(
        printable_monotonicity(
            monotonicity
        ).to_string(
            index=False
        )
    )

    print()

    print(
        "FULL FIVE-LEVEL MONOTONICITY"
    )
    print(
        "-" * 96
    )

    print(
        printable_full_monotonicity(
            full_monotonicity
        ).to_string(
            index=False
        )
    )

    print()

    print(
        "INTERPRETATION GUIDE"
    )
    print(
        "-" * 96
    )

    print(
        "Look for:"
    )
    print(
        "  1. Ordinary distributions staying near the "
        "bottom of the recommendation scale."
    )
    print(
        "  2. Weak < moderate < strong < extreme within "
        "each signal family."
    )
    print(
        "  3. Meaningful score separation between "
        "adjacent levels."
    )
    print(
        "  4. Combined tail evidence not saturating the "
        "score too quickly."
    )
    print(
        "  5. Strong bimodality eventually receiving a "
        "meaningful signal rather than looking ordinary."
    )

    print()

    print(
        "=" * 96
    )
    print(
        f"Raw results saved to {RESULTS_DIR}/"
    )
    print(
        "=" * 96
    )
    print()


# ==================================================
# MAIN BENCHMARK
# ==================================================

def run_histogram_calibration(
    seeds=None,
    n=DEFAULT_SAMPLE_SIZE
):
    """
    Run the complete histogram score-calibration
    benchmark.
    """

    scenarios = (
        build_histogram_calibration_suite(
            seeds=seeds,
            n=n
        )
    )

    rows = []

    for index, scenario in enumerate(
        scenarios,
        start=1
    ):

        print(
            f"[{index:03d}/{len(scenarios):03d}] "
            f"{scenario.name}"
        )

        rows.append(
            run_scenario(
                scenario
            )
        )

    results = pd.DataFrame(
        rows
    )

    level_results = (
        level_summary(
            results
        )
    )

    monotonicity = (
        adjacent_monotonicity(
            results
        )
    )

    full_monotonicity = (
        full_monotonicity_summary(
            results
        )
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        RESULTS_DIR
        / "raw_results.csv",
        index=False
    )

    level_results.to_csv(
        RESULTS_DIR
        / "level_summary.csv",
        index=False
    )

    monotonicity.to_csv(
        RESULTS_DIR
        / "adjacent_monotonicity.csv",
        index=False
    )

    full_monotonicity.to_csv(
        RESULTS_DIR
        / "full_monotonicity.csv",
        index=False
    )

    print_report(
        results=results,
        level_results=level_results,
        monotonicity=monotonicity,
        full_monotonicity=
            full_monotonicity
    )

    return {
        "raw_results": results,
        "level_summary":
            level_results,
        "adjacent_monotonicity":
            monotonicity,
        "full_monotonicity":
            full_monotonicity
    }


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":

    run_histogram_calibration()
