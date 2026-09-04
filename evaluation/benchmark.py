from pathlib import Path
import time

import pandas as pd

from profiler import profile_dataset
from candidate_generator import generate_candidates

from scoring.engine import (
    score_all_candidates,
    diversify_recommendations
)

from evaluation.datasets import (
    build_benchmark_suite
)

from evaluation.metrics import (
    evaluate_positive_scenario,
    evaluate_null_scenario,
    summarize_positive_results,
    summarize_null_results,
    relationship_breakdown,
    null_relationship_breakdown
)


# ==================================================
# CONFIGURATION
# ==================================================

RESULTS_DIR = Path(
    "evaluation/results/hardened_v1_2b"
)


# ==================================================
# SINGLE SCENARIO
# ==================================================

def run_scenario(
    scenario
):
    """
    Run one synthetic dataset through the complete
    Visift recommendation pipeline.
    """

    start = (
        time.perf_counter()
    )

    df = (
        scenario.dataframe
    )

    profile = profile_dataset(
        df
    )

    candidates = generate_candidates(
        df,
        profile
    )

    scored = score_all_candidates(
        df,
        candidates,
        profile
    )

    recommendations = (
        diversify_recommendations(
            scored
        )
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    if scenario.is_null:

        result = evaluate_null_scenario(
            scenario=scenario,
            scored_candidates=scored,
            elapsed_seconds=elapsed
        )

    else:

        result = evaluate_positive_scenario(
            scenario=scenario,
            recommendations=recommendations,
            candidate_count=len(
                candidates
            ),
            elapsed_seconds=elapsed
        )

    return (
        result,
        scored,
        recommendations
    )


# ==================================================
# FORMATTING
# ==================================================

def format_percentage(
    value
):
    return (
        f"{value:.1f}%"
    )


def format_optional_number(
    value,
    decimals=2
):
    if value is None:
        return "N/A"

    if pd.isna(
        value
    ):
        return "N/A"

    return (
        f"{value:.{decimals}f}"
    )


def printable_percentage_columns(
    dataframe,
    columns
):
    """
    Return a copy with selected numeric percentage
    columns formatted for terminal output.
    """

    output = (
        dataframe.copy()
    )

    for column in columns:

        if (
            column
            in output.columns
        ):

            output[
                column
            ] = output[
                column
            ].map(
                lambda value:
                f"{value:.1f}%"
            )

    return output


def printable_numeric_columns(
    dataframe,
    columns,
    decimals=2
):
    """
    Format numeric columns for terminal output.
    """

    output = (
        dataframe.copy()
    )

    for column in columns:

        if (
            column
            in output.columns
        ):

            output[
                column
            ] = output[
                column
            ].map(
                lambda value:
                (
                    f"{value:.{decimals}f}"
                    if pd.notna(value)
                    else "N/A"
                )
            )

    return output


# ==================================================
# REPORT
# ==================================================

def print_report(
    positive_results,
    null_results,
    positive_breakdown,
    null_breakdown,
    total_candidates,
    total_elapsed
):
    """
    Print the hardened benchmark report.
    """

    positive_summary = (
        summarize_positive_results(
            positive_results
        )
    )

    null_summary = (
        summarize_null_results(
            null_results
        )
    )

    total_datasets = (
        len(positive_results)
        + len(null_results)
    )

    print()

    print(
        "=" * 88
    )

    print(
        "VISIFT HARDENED BENCHMARK"
    )

    print(
        "=" * 88
    )

    print()

    print(
        f"Datasets tested:                  "
        f"{total_datasets}"
    )

    print(
        f"Positive datasets:                "
        f"{len(positive_results)}"
    )

    print(
        f"Null datasets:                    "
        f"{len(null_results)}"
    )

    print(
        f"Visualization candidates:         "
        f"{total_candidates}"
    )

    print(
        f"Total runtime:                    "
        f"{total_elapsed:.2f}s"
    )

    print()

    # -----------------------------------
    # Positive retrieval
    # -----------------------------------

    print(
        "GROUND-TRUTH RELATIONSHIP RETRIEVAL"
    )

    print(
        "-" * 88
    )

    print(
        f"Top-1 accuracy:                   "
        f"{format_percentage(positive_summary['top_1_accuracy'])}"
    )

    print(
        f"Top-3 accuracy:                   "
        f"{format_percentage(positive_summary['top_3_accuracy'])}"
    )

    print(
        f"Top-5 accuracy:                   "
        f"{format_percentage(positive_summary['top_5_accuracy'])}"
    )

    print(
        f"Mean reciprocal rank:             "
        f"{positive_summary['mean_reciprocal_rank']:.3f}"
    )

    print(
        f"Median global rank:               "
        f"{format_optional_number(positive_summary['median_global_rank'], 1)}"
    )

    print(
        f"Median chart-type rank:           "
        f"{format_optional_number(positive_summary['median_type_rank'], 1)}"
    )

    print(
        f"Mean expected score:              "
        f"{format_optional_number(positive_summary['mean_expected_score'])}"
    )

    print()

    # -----------------------------------
    # Null performance
    # -----------------------------------

    print(
        "NULL / FALSE-POSITIVE PERFORMANCE"
    )

    print(
        "-" * 88
    )

    print(
        f"Average evaluated null score:     "
        f"{null_summary['mean_candidate_score']:.2f}"
    )

    print(
        f"Average maximum null score:       "
        f"{null_summary['mean_max_score']:.2f}"
    )

    print(
        f"Highest null score observed:      "
        f"{null_summary['highest_noise_score']:.2f}"
    )

    print(
        f"Null candidates scoring >= 50:    "
        f"{format_percentage(null_summary['mean_pct_above_50'])}"
    )

    print(
        f"Null candidates scoring >= 65:    "
        f"{format_percentage(null_summary['mean_pct_above_65'])}"
    )

    print()

    # -----------------------------------
    # Positive breakdown
    # -----------------------------------

    print(
        "PER-RELATIONSHIP PERFORMANCE"
    )

    print(
        "-" * 88
    )

    if positive_breakdown.empty:

        print(
            "No positive benchmark results."
        )

    else:

        printable = (
            positive_breakdown.copy()
        )

        printable = (
            printable_percentage_columns(
                printable,
                [
                    "top_1_pct",
                    "top_3_pct",
                    "top_5_pct"
                ]
            )
        )

        printable = (
            printable_numeric_columns(
                printable,
                [
                    "mean_reciprocal_rank",
                    "median_rank",
                    "median_type_rank",
                    "mean_score"
                ],
                decimals=2
            )
        )

        print(
            printable.to_string(
                index=False
            )
        )

    print()

    # -----------------------------------
    # Null breakdown
    # -----------------------------------

    print(
        "PER-NULL-SCENARIO PERFORMANCE"
    )

    print(
        "-" * 88
    )

    if null_breakdown.empty:

        print(
            "No null benchmark results."
        )

    else:

        printable_null = (
            null_breakdown.copy()
        )

        printable_null = (
            printable_percentage_columns(
                printable_null,
                [
                    "pct_above_50",
                    "pct_above_65"
                ]
            )
        )

        printable_null = (
            printable_numeric_columns(
                printable_null,
                [
                    "mean_score",
                    "mean_max_score",
                    "highest_score"
                ],
                decimals=2
            )
        )

        print(
            printable_null.to_string(
                index=False
            )
        )

    print()

    print(
        "=" * 88
    )

    print(
        f"Raw results saved to "
        f"{RESULTS_DIR}/"
    )

    print(
        "=" * 88
    )

    print()


# ==================================================
# BENCHMARK
# ==================================================

def run_benchmark(
    seeds=None
):
    """
    Run the complete hardened benchmark suite.
    """

    scenarios = build_benchmark_suite(
        seeds=seeds
    )

    positive_results = []
    null_results = []

    total_candidates = 0

    suite_start = (
        time.perf_counter()
    )

    for index, scenario in enumerate(
        scenarios,
        start=1
    ):

        print(
            f"[{index:03d}/{len(scenarios):03d}] "
            f"{scenario.name}"
        )

        (
            result,
            scored,
            recommendations
        ) = run_scenario(
            scenario
        )

        total_candidates += len(
            scored
        )

        if scenario.is_null:

            null_results.append(
                result
            )

        else:

            positive_results.append(
                result
            )

    total_elapsed = (
        time.perf_counter()
        - suite_start
    )

    # -----------------------------------
    # Convert to dataframes
    # -----------------------------------

    positive_df = pd.DataFrame(
        positive_results
    )

    null_df = pd.DataFrame(
        null_results
    )

    positive_breakdown = (
        relationship_breakdown(
            positive_results
        )
    )

    null_breakdown = (
        null_relationship_breakdown(
            null_results
        )
    )

    # -----------------------------------
    # Save results
    # -----------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    positive_df.to_csv(
        RESULTS_DIR
        / "positive_scenarios.csv",
        index=False
    )

    null_df.to_csv(
        RESULTS_DIR
        / "null_scenarios.csv",
        index=False
    )

    positive_breakdown.to_csv(
        RESULTS_DIR
        / "relationship_breakdown.csv",
        index=False
    )

    null_breakdown.to_csv(
        RESULTS_DIR
        / "null_breakdown.csv",
        index=False
    )

    # -----------------------------------
    # Terminal report
    # -----------------------------------

    print_report(
        positive_results=positive_results,
        null_results=null_results,
        positive_breakdown=positive_breakdown,
        null_breakdown=null_breakdown,
        total_candidates=total_candidates,
        total_elapsed=total_elapsed
    )

    return {
        "positive_results": (
            positive_df
        ),

        "null_results": (
            null_df
        ),

        "relationship_breakdown": (
            positive_breakdown
        ),

        "null_breakdown": (
            null_breakdown
        )
    }


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":

    run_benchmark()