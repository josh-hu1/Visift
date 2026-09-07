from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2


# ==================================================
# CONFIGURATION
# ==================================================

RESULTS_DIR = Path(
    "evaluation/results/binary_rate_calibration_v1"
)

OUTPUT_DIR = Path(
    "evaluation/results/binary_rate_formula_experiment_v1"
)

GROUP_COUNT = 4

CANDIDATES = {
    "conservative": {
        "absolute_scale_pp": 60.0,
        "relative_scale": 2.5,
        "minority_events_per_group_full": 15.0
    },
    "responsive": {
        "absolute_scale_pp": 60.0,
        "relative_scale": 2.0,
        "minority_events_per_group_full": 15.0
    }
}


# ==================================================
# HELPERS
# ==================================================

def significance_reliability(
    p_value
):
    """
    Convert a chi-square p-value into a smooth reliability
    multiplier.

    p >= 0.10  -> 0 reliability
    p <= 0.001 -> full reliability

    Values in between are interpolated on a log10 scale.
    """

    if not np.isfinite(
        p_value
    ):

        return 0.0

    if p_value >= 0.10:

        return 0.0

    if p_value <= 0.001:

        return 1.0

    numerator = (
        np.log10(0.10)
        - np.log10(
            p_value
        )
    )

    denominator = (
        np.log10(0.10)
        - np.log10(0.001)
    )

    return float(
        np.clip(
            numerator
            / denominator,
            0.0,
            1.0
        )
    )


def formula_components(
    row,
    parameters
):
    """
    Calculate a candidate binary-rate signal from calibration
    diagnostics without modifying the production Visift scorer.
    """

    rate_range_pp = float(
        row[
            "rate_range_pp"
        ]
    )

    lift = float(
        row[
            "max_lift_vs_overall"
        ]
    )

    observations = float(
        row[
            "observations"
        ]
    )

    positives = float(
        row[
            "positive_count"
        ]
    )

    cramers_v = float(
        row[
            "cramers_v"
        ]
    )

    visualization_quality = float(
        row[
            "visualization_quality"
        ]
    )

    # -----------------------------------
    # Absolute rate-separation component
    # -----------------------------------

    absolute_signal = (
        100.0
        * (
            1.0
            - np.exp(
                -rate_range_pp
                / parameters[
                    "absolute_scale_pp"
                ]
            )
        )
    )

    # -----------------------------------
    # Relative lift component
    # -----------------------------------

    if (
        np.isfinite(
            lift
        )
        and lift > 1.0
    ):

        log2_lift = (
            np.log2(
                lift
            )
        )

    else:

        log2_lift = 0.0

    relative_signal = (
        100.0
        * (
            1.0
            - np.exp(
                -log2_lift
                / parameters[
                    "relative_scale"
                ]
            )
        )
    )

    effect_signal = max(
        absolute_signal,
        relative_signal
    )

    # -----------------------------------
    # Event-evidence reliability
    # -----------------------------------

    negatives = max(
        observations
        - positives,
        0.0
    )

    minority_events = min(
        positives,
        negatives
    )

    minority_events_per_group = (
        minority_events
        / GROUP_COUNT
    )

    event_reliability = np.sqrt(
        min(
            1.0,
            minority_events_per_group
            / parameters[
                "minority_events_per_group_full"
            ]
        )
    )

    # -----------------------------------
    # Statistical reliability
    # -----------------------------------

    # For category x binary tables:
    #
    # Cramer's V = sqrt(chi_square / n)
    #
    # because min(r - 1, c - 1) = 1.
    chi_square = (
        observations
        * (
            cramers_v ** 2
        )
    )

    degrees_of_freedom = (
        GROUP_COUNT
        - 1
    )

    p_value = float(
        chi2.sf(
            chi_square,
            degrees_of_freedom
        )
    )

    p_reliability = (
        significance_reliability(
            p_value
        )
    )

    statistical_reliability = (
        np.sqrt(
            p_reliability
        )
    )

    reliability = (
        event_reliability
        * statistical_reliability
    )

    signal = (
        effect_signal
        * reliability
    )

    final_score = (
        visualization_quality
        * (
            0.35
            + 0.65
            * (
                signal / 100.0
            )
        )
    )

    return {
        "absolute_signal": absolute_signal,
        "relative_signal": relative_signal,
        "effect_signal": effect_signal,
        "minority_events": minority_events,
        "minority_events_per_group": (
            minority_events_per_group
        ),
        "event_reliability": event_reliability,
        "chi_square": chi_square,
        "p_value": p_value,
        "p_reliability": p_reliability,
        "statistical_reliability": (
            statistical_reliability
        ),
        "reliability": reliability,
        "candidate_signal": signal,
        "candidate_score": final_score
    }


def apply_formula(
    df,
    name,
    parameters
):
    """
    Apply one candidate formula to a raw calibration table.
    """

    rows = []

    for _, row in df.iterrows():

        components = (
            formula_components(
                row,
                parameters
            )
        )

        rows.append({
            **row.to_dict(),
            "formula": name,
            **components
        })

    return pd.DataFrame(
        rows
    )


def p10(
    series
):
    return float(
        series.quantile(
            0.10
        )
    )


def p90(
    series
):
    return float(
        series.quantile(
            0.90
        )
    )


def display_table(
    df,
    percent_columns=()
):
    """
    Print compact numeric tables without mutating percentage
    columns before their final formatting step.
    """

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
                    if pd.notna(
                        value
                    )
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
                        f"{float(value):.1f}%"
                        if pd.notna(
                            value
                        )
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
# SUMMARIES
# ==================================================

def strength_summary(
    df
):
    return (
        df
        .groupby(
            [
                "formula",
                "family",
                "strength_index",
                "level"
            ],
            as_index=False
        )
        .agg(
            scenarios=(
                "candidate_score",
                "size"
            ),
            median_score=(
                "candidate_score",
                "median"
            ),
            p10_score=(
                "candidate_score",
                p10
            ),
            p90_score=(
                "candidate_score",
                p90
            ),
            median_signal=(
                "candidate_signal",
                "median"
            ),
            median_effect_signal=(
                "effect_signal",
                "median"
            ),
            median_reliability=(
                "reliability",
                "median"
            )
        )
        .sort_values(
            [
                "formula",
                "family",
                "strength_index"
            ]
        )
    )


def sample_size_summary(
    df
):
    return (
        df
        .groupby(
            [
                "formula",
                "n"
            ],
            as_index=False
        )
        .agg(
            scenarios=(
                "candidate_score",
                "size"
            ),
            median_score=(
                "candidate_score",
                "median"
            ),
            p10_score=(
                "candidate_score",
                p10
            ),
            p90_score=(
                "candidate_score",
                p90
            ),
            highest_score=(
                "candidate_score",
                "max"
            ),
            median_signal=(
                "candidate_signal",
                "median"
            ),
            median_reliability=(
                "reliability",
                "median"
            ),
            median_minority_events_per_group=(
                "minority_events_per_group",
                "median"
            )
        )
        .sort_values(
            [
                "formula",
                "n"
            ]
        )
    )


def missingness_summary(
    df
):
    return (
        df
        .groupby(
            [
                "formula",
                "missing_fraction"
            ],
            as_index=False
        )
        .agg(
            scenarios=(
                "candidate_score",
                "size"
            ),
            median_score=(
                "candidate_score",
                "median"
            ),
            p10_score=(
                "candidate_score",
                p10
            ),
            p90_score=(
                "candidate_score",
                p90
            ),
            highest_score=(
                "candidate_score",
                "max"
            ),
            median_signal=(
                "candidate_signal",
                "median"
            ),
            median_reliability=(
                "reliability",
                "median"
            )
        )
        .sort_values(
            [
                "formula",
                "missing_fraction"
            ]
        )
    )


def null_summary(
    df
):
    grouped_rows = []

    for (
        formula,
        level
    ), group in df.groupby(
        [
            "formula",
            "level"
        ]
    ):

        grouped_rows.append({
            "formula": formula,
            "level": level,
            "scenarios": len(
                group
            ),
            "mean_score": (
                group[
                    "candidate_score"
                ].mean()
            ),
            "median_score": (
                group[
                    "candidate_score"
                ].median()
            ),
            "p90_score": (
                group[
                    "candidate_score"
                ].quantile(
                    0.90
                )
            ),
            "highest_score": (
                group[
                    "candidate_score"
                ].max()
            ),
            "pct_above_50": (
                100
                * (
                    group[
                        "candidate_score"
                    ]
                    >= 50
                ).mean()
            ),
            "pct_above_65": (
                100
                * (
                    group[
                        "candidate_score"
                    ]
                    >= 65
                ).mean()
            )
        })

    return (
        pd.DataFrame(
            grouped_rows
        )
        .sort_values(
            [
                "formula",
                "level"
            ]
        )
    )


def representation_consistency(
    df
):
    rows = []

    for (
        formula,
        seed
    ), group in df.groupby(
        [
            "formula",
            "seed"
        ]
    ):

        rows.append({
            "formula": formula,
            "seed": seed,
            "score_range": (
                group[
                    "candidate_score"
                ].max()
                - group[
                    "candidate_score"
                ].min()
            ),
            "signal_range": (
                group[
                    "candidate_signal"
                ].max()
                - group[
                    "candidate_signal"
                ].min()
            )
        })

    return pd.DataFrame(
        rows
    )


# ==================================================
# MAIN
# ==================================================

def run_experiment():
    required_files = {
        "strength": (
            RESULTS_DIR
            / "strength_raw.csv"
        ),
        "sample_size": (
            RESULTS_DIR
            / "sample_size_raw.csv"
        ),
        "representation": (
            RESULTS_DIR
            / "representation_raw.csv"
        ),
        "missingness": (
            RESULTS_DIR
            / "missingness_raw.csv"
        ),
        "null": (
            RESULTS_DIR
            / "null_raw.csv"
        )
    }

    missing = [
        str(path)
        for path in required_files.values()
        if not path.exists()
    ]

    if missing:

        raise FileNotFoundError(
            "Run "
            "'python -m evaluation.binary_rate_calibration' "
            "first. Missing files: "
            + ", ".join(
                missing
            )
        )

    source_tables = {
        name: pd.read_csv(
            path
        )
        for name, path in (
            required_files.items()
        )
    }

    candidate_tables = {
        name: []
        for name in source_tables
    }

    for (
        formula_name,
        parameters
    ) in CANDIDATES.items():

        for (
            table_name,
            source_df
        ) in source_tables.items():

            candidate_tables[
                table_name
            ].append(
                apply_formula(
                    source_df,
                    formula_name,
                    parameters
                )
            )

    candidate_tables = {
        name: pd.concat(
            frames,
            ignore_index=True
        )
        for name, frames in (
            candidate_tables.items()
        )
    }

    strength = (
        strength_summary(
            candidate_tables[
                "strength"
            ]
        )
    )

    sample_size = (
        sample_size_summary(
            candidate_tables[
                "sample_size"
            ]
        )
    )

    representation = (
        representation_consistency(
            candidate_tables[
                "representation"
            ]
        )
    )

    missingness = (
        missingness_summary(
            candidate_tables[
                "missingness"
            ]
        )
    )

    nulls = (
        null_summary(
            candidate_tables[
                "null"
            ]
        )
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    strength.to_csv(
        OUTPUT_DIR
        / "strength_summary.csv",
        index=False
    )

    sample_size.to_csv(
        OUTPUT_DIR
        / "sample_size_summary.csv",
        index=False
    )

    representation.to_csv(
        OUTPUT_DIR
        / "representation_consistency.csv",
        index=False
    )

    missingness.to_csv(
        OUTPUT_DIR
        / "missingness_summary.csv",
        index=False
    )

    nulls.to_csv(
        OUTPUT_DIR
        / "null_summary.csv",
        index=False
    )

    print()
    print(
        "=" * 112
    )
    print(
        "VISIFT BINARY-RATE FORMULA EXPERIMENT"
    )
    print(
        "=" * 112
    )

    print()
    print(
        "PURPOSE"
    )
    print(
        "-" * 112
    )
    print(
        "Compare two candidate binary-rate signal formulas "
        "against the already-generated calibration scenarios. "
        "This does not modify Visift's production scorer."
    )

    print()
    print(
        "CANDIDATE PARAMETERS"
    )
    print(
        "-" * 112
    )

    for (
        name,
        parameters
    ) in CANDIDATES.items():

        print(
            f"{name}: {parameters}"
        )

    print()
    print(
        "STRENGTH CALIBRATION"
    )
    print(
        "-" * 112
    )
    display_table(
        strength
    )

    print()
    print(
        "SAMPLE-SIZE / EVENT-EVIDENCE ROBUSTNESS"
    )
    print(
        "-" * 112
    )
    display_table(
        sample_size
    )

    print()
    print(
        "BOOLEAN REPRESENTATION INVARIANCE"
    )
    print(
        "-" * 112
    )

    representation_summary = (
        representation
        .groupby(
            "formula",
            as_index=False
        )
        .agg(
            maximum_score_delta=(
                "score_range",
                "max"
            ),
            maximum_signal_delta=(
                "signal_range",
                "max"
            )
        )
    )

    display_table(
        representation_summary
    )

    print()
    print(
        "MISSINGNESS ROBUSTNESS"
    )
    print(
        "-" * 112
    )
    display_table(
        missingness
    )

    print()
    print(
        "NULL ROBUSTNESS"
    )
    print(
        "-" * 112
    )
    display_table(
        nulls,
        percent_columns=(
            "pct_above_50",
            "pct_above_65"
        )
    )

    print()
    print(
        "DECISION CRITERIA"
    )
    print(
        "-" * 112
    )

    print(
        "Prefer a formula that:"
    )

    print(
        "  1. Keeps all null families safely below 50."
    )

    print(
        "  2. Produces a smooth ordinary < weak < moderate "
        "< strong < extreme progression."
    )

    print(
        "  3. Does not collapse strong imbalanced or rare "
        "binary-rate relationships solely because prevalence "
        "is low."
    )

    print(
        "  4. Suppresses n=40 and n=80 apparent effects when "
        "there are too few minority-class events."
    )

    print(
        "  5. Preserves exact representation invariance."
    )

    print(
        "  6. Degrades gradually under missingness."
    )

    print()
    print(
        "=" * 112
    )

    print(
        f"Results saved to {OUTPUT_DIR}/"
    )

    print(
        "=" * 112
    )

    print()


if __name__ == "__main__":
    run_experiment()
