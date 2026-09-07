from pathlib import Path

import numpy as np
import pandas as pd

from scoring.bar import score_bar_chart


# ==================================================
# CONFIGURATION
# ==================================================

SEEDS = range(25)

RESULTS_DIR = Path(
    "evaluation/results/binary_rate_calibration_v1"
)

GROUPS = np.array([
    "A",
    "B",
    "C",
    "D"
])


# ==================================================
# VISIFT HELPERS
# ==================================================

def binary_rate_profile():
    """
    Minimal profile required by the bar scorer.

    The calibration intentionally isolates the scoring layer
    rather than semantic inference or candidate generation.
    """

    return {
        "column_profiles": [
            {
                "name": "segment",
                "semantic_type": "categorical"
            },
            {
                "name": "outcome",
                "semantic_type": "boolean"
            }
        ]
    }


def rate_candidate():
    """
    Category -> binary-rate mean-bar candidate.
    """

    return {
        "chart": "bar",
        "x": "segment",
        "y": "outcome",
        "aggregation": "mean"
    }


def encode_outcome(
    values,
    encoding
):
    """
    Represent the same binary outcome several ways.

    This checks that Visift's text-boolean normalization is
    representation-invariant.
    """

    values = np.asarray(
        values,
        dtype=int
    )

    if encoding == "int":

        return values

    if encoding == "bool":

        return values.astype(
            bool
        )

    if encoding == "yes_no":

        return np.where(
            values == 1,
            "yes",
            "no"
        )

    if encoding == "true_false":

        return np.where(
            values == 1,
            "true",
            "false"
        )

    raise ValueError(
        f"Unknown outcome encoding: {encoding}"
    )


def make_rate_dataset(
    seed,
    rates,
    n=2000,
    encoding="yes_no",
    missing_fraction=0.0
):
    """
    Create a categorical predictor with a binary outcome whose
    positive probability differs by group.

    Rates correspond to GROUPS in order.
    """

    rng = np.random.default_rng(
        seed
    )

    rates = np.asarray(
        rates,
        dtype=float
    )

    if len(rates) != len(GROUPS):

        raise ValueError(
            "rates must contain one value per group."
        )

    segment = rng.choice(
        GROUPS,
        size=n,
        replace=True
    )

    rate_map = {
        group: rate
        for group, rate in zip(
            GROUPS,
            rates
        )
    }

    probabilities = np.array([
        rate_map[value]
        for value in segment
    ])

    binary = rng.binomial(
        1,
        probabilities
    )

    outcome = encode_outcome(
        binary,
        encoding
    )

    df = pd.DataFrame({
        "segment": segment,
        "outcome": outcome
    })

    if missing_fraction > 0:

        missing_count = int(
            round(
                n * missing_fraction
            )
        )

        if missing_count > 0:

            missing_indices = rng.choice(
                df.index,
                size=missing_count,
                replace=False
            )

            df.loc[
                missing_indices,
                "outcome"
            ] = pd.NA

    return (
        df,
        binary
    )


def score_rate_dataset(
    df
):
    """
    Run the current Visift mean-bar scorer.
    """

    result = score_bar_chart(
        df,
        rate_candidate(),
        binary_rate_profile()
    )

    return result


# ==================================================
# EFFECT-SIZE DIAGNOSTICS
# ==================================================

def numeric_binary(
    series
):
    """
    Convert supported binary representations to 0/1 solely for
    calibration diagnostics.

    This does not modify Visift scoring behavior.
    """

    if pd.api.types.is_bool_dtype(
        series
    ):

        return series.astype(
            float
        )

    if pd.api.types.is_numeric_dtype(
        series
    ):

        return pd.to_numeric(
            series,
            errors="coerce"
        )

    normalized = (
        series
        .astype("string")
        .str.strip()
        .str.lower()
    )

    mapping = {
        "yes": 1.0,
        "true": 1.0,
        "y": 1.0,
        "on": 1.0,
        "1": 1.0,
        "no": 0.0,
        "false": 0.0,
        "n": 0.0,
        "off": 0.0,
        "0": 0.0
    }

    return (
        normalized
        .map(mapping)
        .astype(float)
    )


def binary_rate_diagnostics(
    df
):
    """
    Calculate descriptive quantities that may be useful when a
    dedicated binary-rate signal is designed later.

    None of these diagnostics currently affect Visift scores.
    """

    clean = (
        df[
            [
                "segment",
                "outcome"
            ]
        ]
        .copy()
    )

    clean[
        "outcome_numeric"
    ] = numeric_binary(
        clean["outcome"]
    )

    clean = clean.dropna(
        subset=[
            "segment",
            "outcome_numeric"
        ]
    )

    if clean.empty:

        return {
            "observations": 0,
            "positive_count": 0,
            "overall_rate": np.nan,
            "min_group_rate": np.nan,
            "max_group_rate": np.nan,
            "rate_range_pp": np.nan,
            "max_lift_vs_overall": np.nan,
            "cramers_v": np.nan,
            "min_group_size": 0,
            "min_group_positive_count": 0
        }

    grouped = (
        clean
        .groupby(
            "segment"
        )["outcome_numeric"]
        .agg(
            [
                "count",
                "sum",
                "mean"
            ]
        )
    )

    overall_rate = float(
        clean[
            "outcome_numeric"
        ].mean()
    )

    min_rate = float(
        grouped["mean"].min()
    )

    max_rate = float(
        grouped["mean"].max()
    )

    rate_range_pp = (
        max_rate
        - min_rate
    ) * 100

    max_lift = (
        max_rate
        / overall_rate
        if overall_rate > 0
        else np.nan
    )

    # -----------------------------------
    # Cramer's V for category x binary
    # -----------------------------------

    observed = np.column_stack([
        grouped["sum"].to_numpy(
            dtype=float
        ),
        (
            grouped["count"]
            - grouped["sum"]
        ).to_numpy(
            dtype=float
        )
    ])

    n = observed.sum()

    if (
        n <= 0
        or observed.shape[0] <= 1
    ):

        cramers_v = 0.0

    else:

        row_totals = observed.sum(
            axis=1,
            keepdims=True
        )

        column_totals = observed.sum(
            axis=0,
            keepdims=True
        )

        expected = (
            row_totals
            @ column_totals
            / n
        )

        valid = expected > 0

        chi_square = (
            (
                (
                    observed
                    - expected
                ) ** 2
                / np.where(
                    valid,
                    expected,
                    1
                )
            )[valid]
            .sum()
        )

        # For a category x binary table:
        # min(r - 1, c - 1) = 1
        cramers_v = float(
            np.sqrt(
                chi_square / n
            )
        )

    return {
        "observations": int(
            len(clean)
        ),
        "positive_count": int(
            clean[
                "outcome_numeric"
            ].sum()
        ),
        "overall_rate": overall_rate,
        "min_group_rate": min_rate,
        "max_group_rate": max_rate,
        "rate_range_pp": rate_range_pp,
        "max_lift_vs_overall": (
            float(max_lift)
            if np.isfinite(max_lift)
            else np.nan
        ),
        "cramers_v": cramers_v,
        "min_group_size": int(
            grouped["count"].min()
        ),
        "min_group_positive_count": int(
            grouped["sum"].min()
        )
    }


def calibration_row(
    family,
    level,
    strength_index,
    seed,
    df,
    planted_rates,
    encoding,
    n,
    missing_fraction=0.0
):
    """
    Score one scenario and collect current Visift output plus
    descriptive binary-rate diagnostics.
    """

    result = score_rate_dataset(
        df
    )

    diagnostics = (
        binary_rate_diagnostics(
            df
        )
    )

    return {
        "family": family,
        "level": level,
        "strength_index": strength_index,
        "seed": seed,
        "n": n,
        "encoding": encoding,
        "missing_fraction": missing_fraction,
        "planted_min_rate": float(
            min(planted_rates)
        ),
        "planted_max_rate": float(
            max(planted_rates)
        ),
        "planted_range_pp": float(
            (
                max(planted_rates)
                - min(planted_rates)
            )
            * 100
        ),
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
        "sample_support": result[
            "components"
        ][
            "sample_support"
        ],
        **diagnostics
    }


# ==================================================
# SCENARIO FAMILIES
# ==================================================

COMMON_RATE_LEVELS = (
    (
        "ordinary",
        (
            0.50,
            0.50,
            0.50,
            0.50
        )
    ),
    (
        "weak",
        (
            0.44,
            0.48,
            0.52,
            0.56
        )
    ),
    (
        "moderate",
        (
            0.35,
            0.45,
            0.55,
            0.65
        )
    ),
    (
        "strong",
        (
            0.25,
            0.40,
            0.60,
            0.75
        )
    ),
    (
        "extreme",
        (
            0.10,
            0.30,
            0.70,
            0.90
        )
    ),
)


IMBALANCED_RATE_LEVELS = (
    (
        "ordinary",
        (
            0.10,
            0.10,
            0.10,
            0.10
        )
    ),
    (
        "weak",
        (
            0.07,
            0.09,
            0.11,
            0.13
        )
    ),
    (
        "moderate",
        (
            0.04,
            0.08,
            0.14,
            0.20
        )
    ),
    (
        "strong",
        (
            0.02,
            0.06,
            0.18,
            0.30
        )
    ),
    (
        "extreme",
        (
            0.01,
            0.03,
            0.22,
            0.45
        )
    ),
)


RARE_RATE_LEVELS = (
    (
        "ordinary",
        (
            0.02,
            0.02,
            0.02,
            0.02
        )
    ),
    (
        "weak",
        (
            0.01,
            0.015,
            0.025,
            0.035
        )
    ),
    (
        "moderate",
        (
            0.005,
            0.015,
            0.035,
            0.065
        )
    ),
    (
        "strong",
        (
            0.003,
            0.012,
            0.055,
            0.10
        )
    ),
    (
        "extreme",
        (
            0.001,
            0.008,
            0.08,
            0.16
        )
    ),
)


def run_strength_scenarios():
    """
    Compare the current score curve at common, imbalanced,
    and rare base rates.
    """

    rows = []

    families = (
        (
            "common_prevalence",
            COMMON_RATE_LEVELS
        ),
        (
            "imbalanced_prevalence",
            IMBALANCED_RATE_LEVELS
        ),
        (
            "rare_prevalence",
            RARE_RATE_LEVELS
        ),
    )

    for (
        family,
        levels
    ) in families:

        for (
            strength_index,
            (
                level,
                rates
            )
        ) in enumerate(
            levels
        ):

            for seed in SEEDS:

                df, _ = make_rate_dataset(
                    seed=seed,
                    rates=rates,
                    n=2000,
                    encoding="yes_no"
                )

                rows.append(
                    calibration_row(
                        family=family,
                        level=level,
                        strength_index=(
                            strength_index
                        ),
                        seed=seed,
                        df=df,
                        planted_rates=rates,
                        encoding="yes_no",
                        n=2000
                    )
                )

    return pd.DataFrame(
        rows
    )


def run_sample_size_scenarios():
    """
    Hold a meaningful rate pattern approximately fixed while
    changing n.

    This exposes whether confidence reacts to actual event
    evidence rather than only raw group size.
    """

    rows = []

    rates = (
        0.02,
        0.06,
        0.18,
        0.30
    )

    sample_sizes = (
        40,
        80,
        200,
        500,
        2000,
        10000
    )

    for n in sample_sizes:

        for seed in SEEDS:

            df, _ = make_rate_dataset(
                seed=seed,
                rates=rates,
                n=n,
                encoding="yes_no"
            )

            rows.append(
                calibration_row(
                    family="sample_size",
                    level=str(n),
                    strength_index=np.nan,
                    seed=seed,
                    df=df,
                    planted_rates=rates,
                    encoding="yes_no",
                    n=n
                )
            )

    return pd.DataFrame(
        rows
    )


def run_representation_scenarios():
    """
    The same underlying binary outcome should receive the same
    recommendation score whether stored as 0/1, bool, yes/no,
    or true/false.
    """

    rows = []

    rates = (
        0.04,
        0.08,
        0.14,
        0.20
    )

    encodings = (
        "int",
        "bool",
        "yes_no",
        "true_false"
    )

    for seed in SEEDS:

        # Use one fixed segment/outcome realization, then change
        # only its storage representation.
        base_df, binary = (
            make_rate_dataset(
                seed=seed,
                rates=rates,
                n=2000,
                encoding="int"
            )
        )

        for encoding in encodings:

            df = base_df.copy()

            df["outcome"] = (
                encode_outcome(
                    binary,
                    encoding
                )
            )

            rows.append(
                calibration_row(
                    family="representation",
                    level=encoding,
                    strength_index=np.nan,
                    seed=seed,
                    df=df,
                    planted_rates=rates,
                    encoding=encoding,
                    n=2000
                )
            )

    return pd.DataFrame(
        rows
    )


def run_missingness_scenarios():
    """
    Test graceful degradation when binary outcomes are missing.
    """

    rows = []

    rates = (
        0.02,
        0.06,
        0.18,
        0.30
    )

    missing_levels = (
        0.0,
        0.20,
        0.40,
        0.60,
        0.80
    )

    for missing_fraction in (
        missing_levels
    ):

        for seed in SEEDS:

            df, _ = make_rate_dataset(
                seed=seed,
                rates=rates,
                n=2000,
                encoding="yes_no",
                missing_fraction=(
                    missing_fraction
                )
            )

            rows.append(
                calibration_row(
                    family="missingness",
                    level=(
                        f"{missing_fraction:.0%}"
                    ),
                    strength_index=np.nan,
                    seed=seed,
                    df=df,
                    planted_rates=rates,
                    encoding="yes_no",
                    n=2000,
                    missing_fraction=(
                        missing_fraction
                    )
                )
            )

    return pd.DataFrame(
        rows
    )


def run_null_scenarios():
    """
    Independent group membership and binary outcomes.

    Null behavior is checked across common, imbalanced, and
    rare outcomes and across multiple sample sizes.
    """

    rows = []

    null_settings = (
        (
            "common_n500",
            0.50,
            500
        ),
        (
            "imbalanced_n500",
            0.10,
            500
        ),
        (
            "rare_n500",
            0.02,
            500
        ),
        (
            "common_n5000",
            0.50,
            5000
        ),
        (
            "imbalanced_n5000",
            0.10,
            5000
        ),
        (
            "rare_n5000",
            0.02,
            5000
        ),
    )

    for (
        label,
        rate,
        n
    ) in null_settings:

        rates = (
            rate,
            rate,
            rate,
            rate
        )

        for seed in SEEDS:

            df, _ = make_rate_dataset(
                seed=seed,
                rates=rates,
                n=n,
                encoding="yes_no"
            )

            rows.append(
                calibration_row(
                    family="null",
                    level=label,
                    strength_index=0,
                    seed=seed,
                    df=df,
                    planted_rates=rates,
                    encoding="yes_no",
                    n=n
                )
            )

    return pd.DataFrame(
        rows
    )


# ==================================================
# SUMMARIES
# ==================================================

def percentile_10(
    series
):
    return float(
        series.quantile(
            0.10
        )
    )


def percentile_90(
    series
):
    return float(
        series.quantile(
            0.90
        )
    )


def summarize_strength(
    df
):
    return (
        df
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
            median_rate_range_pp=(
                "rate_range_pp",
                "median"
            ),
            median_cramers_v=(
                "cramers_v",
                "median"
            ),
            median_positive_count=(
                "positive_count",
                "median"
            ),
            median_min_group_positives=(
                "min_group_positive_count",
                "median"
            )
        )
        .sort_values(
            [
                "family",
                "strength_index"
            ]
        )
    )


def summarize_simple(
    df,
    setting_column
):
    return (
        df
        .groupby(
            setting_column,
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
            p10_score=(
                "score",
                percentile_10
            ),
            p90_score=(
                "score",
                percentile_90
            ),
            highest_score=(
                "score",
                "max"
            ),
            median_signal=(
                "signal",
                "median"
            ),
            median_rate_range_pp=(
                "rate_range_pp",
                "median"
            ),
            median_cramers_v=(
                "cramers_v",
                "median"
            ),
            median_positive_count=(
                "positive_count",
                "median"
            ),
            median_min_group_positives=(
                "min_group_positive_count",
                "median"
            ),
            median_support=(
                "sample_support",
                "median"
            )
        )
    )


def summarize_nulls(
    df
):
    result = (
        df
        .groupby(
            "level",
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
            )
        )
    )

    threshold_rows = []

    for level, group in (
        df.groupby(
            "level"
        )
    ):

        threshold_rows.append({
            "level": level,
            "pct_above_50": (
                100
                * (
                    group["score"]
                    >= 50
                ).mean()
            ),
            "pct_above_65": (
                100
                * (
                    group["score"]
                    >= 65
                ).mean()
            )
        })

    thresholds = pd.DataFrame(
        threshold_rows
    )

    return result.merge(
        thresholds,
        on="level",
        how="left"
    )


def representation_consistency(
    df
):
    """
    Score spread within a seed after changing only the storage
    representation of the binary target.
    """

    rows = []

    for seed, group in (
        df.groupby(
            "seed"
        )
    ):

        rows.append({
            "seed": seed,
            "minimum_score": (
                group["score"].min()
            ),
            "maximum_score": (
                group["score"].max()
            ),
            "score_range": (
                group["score"].max()
                - group["score"].min()
            ),
            "minimum_signal": (
                group["signal"].min()
            ),
            "maximum_signal": (
                group["signal"].max()
            ),
            "signal_range": (
                group["signal"].max()
                - group["signal"].min()
            )
        })

    return pd.DataFrame(
        rows
    )


# ==================================================
# DISPLAY
# ==================================================

def display_table(
    df,
    percent_columns=()
):
    printable = df.copy()

    numeric_columns = (
        printable
        .select_dtypes(
            include=[
                np.number
            ]
        )
        .columns
    )

    for column in numeric_columns:

        if (
            column in {
                "scenarios",
                "strength_index"
            }
            or column in percent_columns
        ):

            continue

        printable[column] = (
            printable[column]
            .map(
                lambda value:
                (
                    f"{value:.2f}"
                    if pd.notna(value)
                    else "N/A"
                )
            )
        )

    for column in percent_columns:

        if column in printable.columns:

            printable[column] = (
                printable[column]
                .map(
                    lambda value:
                    (
                        f"{value:.1f}%"
                        if pd.notna(value)
                        else "N/A"
                    )
                )
            )

    print(
        printable.to_string(
            index=False
        )
    )


# ==================================================
# MAIN
# ==================================================

def run_calibration():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    strength = (
        run_strength_scenarios()
    )

    sample_size = (
        run_sample_size_scenarios()
    )

    representation = (
        run_representation_scenarios()
    )

    missingness = (
        run_missingness_scenarios()
    )

    nulls = (
        run_null_scenarios()
    )

    strength_summary = (
        summarize_strength(
            strength
        )
    )

    sample_summary = (
        summarize_simple(
            sample_size,
            "n"
        )
        .sort_values("n")
    )

    representation_summary = (
        summarize_simple(
            representation,
            "encoding"
        )
    )

    representation_ranges = (
        representation_consistency(
            representation
        )
    )

    missing_summary = (
        summarize_simple(
            missingness,
            "missing_fraction"
        )
        .sort_values(
            "missing_fraction"
        )
    )

    null_summary = (
        summarize_nulls(
            nulls
        )
    )

    # -----------------------------------
    # Save raw + summary outputs
    # -----------------------------------

    strength.to_csv(
        RESULTS_DIR
        / "strength_raw.csv",
        index=False
    )

    strength_summary.to_csv(
        RESULTS_DIR
        / "strength_summary.csv",
        index=False
    )

    sample_size.to_csv(
        RESULTS_DIR
        / "sample_size_raw.csv",
        index=False
    )

    sample_summary.to_csv(
        RESULTS_DIR
        / "sample_size_summary.csv",
        index=False
    )

    representation.to_csv(
        RESULTS_DIR
        / "representation_raw.csv",
        index=False
    )

    representation_summary.to_csv(
        RESULTS_DIR
        / "representation_summary.csv",
        index=False
    )

    representation_ranges.to_csv(
        RESULTS_DIR
        / "representation_consistency.csv",
        index=False
    )

    missingness.to_csv(
        RESULTS_DIR
        / "missingness_raw.csv",
        index=False
    )

    missing_summary.to_csv(
        RESULTS_DIR
        / "missingness_summary.csv",
        index=False
    )

    nulls.to_csv(
        RESULTS_DIR
        / "null_raw.csv",
        index=False
    )

    null_summary.to_csv(
        RESULTS_DIR
        / "null_summary.csv",
        index=False
    )

    # -----------------------------------
    # Console report
    # -----------------------------------

    print()
    print(
        "=" * 112
    )

    print(
        "VISIFT BINARY-RATE SCORE CALIBRATION"
    )

    print(
        "=" * 112
    )

    print()
    print(
        f"Strength scenarios scored:         "
        f"{len(strength)}"
    )

    print(
        f"Sample-size scenarios scored:      "
        f"{len(sample_size)}"
    )

    print(
        f"Representation scenarios scored:   "
        f"{len(representation)}"
    )

    print(
        f"Missingness scenarios scored:      "
        f"{len(missingness)}"
    )

    print(
        f"Null scenarios scored:             "
        f"{len(nulls)}"
    )

    print(
        f"Random seeds:                      "
        f"{len(list(SEEDS))}"
    )

    print()
    print(
        "PURPOSE"
    )

    print(
        "-" * 112
    )

    print(
        "This benchmark isolates category -> binary-outcome "
        "mean bars. It measures the current score curve across "
        "common, imbalanced, and rare outcomes before Visift "
        "receives a dedicated binary-rate scoring branch."
    )

    print()
    print(
        "BINARY-RATE STRENGTH"
    )

    print(
        "-" * 112
    )

    display_table(
        strength_summary
    )

    print()
    print(
        "SAMPLE-SIZE / EVENT-EVIDENCE ROBUSTNESS"
    )

    print(
        "-" * 112
    )

    display_table(
        sample_summary
    )

    print()
    print(
        "BOOLEAN REPRESENTATION INVARIANCE"
    )

    print(
        "-" * 112
    )

    display_table(
        representation_summary
    )

    print()
    print(
        "Per-seed maximum representation score delta: "
        f"{representation_ranges['score_range'].max():.4f}"
    )

    print(
        "Per-seed maximum representation signal delta: "
        f"{representation_ranges['signal_range'].max():.4f}"
    )

    print()
    print(
        "MISSINGNESS ROBUSTNESS"
    )

    print(
        "-" * 112
    )

    display_table(
        missing_summary
    )

    print()
    print(
        "NULL BINARY-RATE ROBUSTNESS"
    )

    print(
        "-" * 112
    )

    display_table(
        null_summary,
        percent_columns=(
            "pct_above_50",
            "pct_above_65"
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
        "  1. Null category/binary relationships staying "
        "near the bottom of the recommendation scale."
    )

    print(
        "  2. Scores increasing smoothly as group rate "
        "differences become stronger."
    )

    print(
        "  3. Similar business-relevant rate differences not "
        "collapsing merely because the overall positive rate "
        "is imbalanced."
    )

    print(
        "  4. Rare-event scenarios requiring enough actual "
        "positive observations before receiving high confidence."
    )

    print(
        "  5. Small samples reducing confidence even when the "
        "observed percentage-point spread is large."
    )

    print(
        "  6. 0/1, bool, yes/no, and true/false encodings "
        "producing effectively identical scores."
    )

    print(
        "  7. Missingness degrading recommendation strength "
        "gradually rather than creating instability."
    )

    print(
        "  8. No null binary-rate family regularly crossing "
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

    return {
        "strength": strength,
        "sample_size": sample_size,
        "representation": representation,
        "missingness": missingness,
        "nulls": nulls
    }


if __name__ == "__main__":
    run_calibration()
