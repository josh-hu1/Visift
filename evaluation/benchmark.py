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
    relationship_breakdown
)


# ==================================================
# CONFIGURATION
# ==================================================

RESULTS_DIR = Path(
    "evaluation/results"
)


# ==================================================
# SINGLE SCENARIO
# ==================================================

def run_scenario(
    scenario,
):
    """
    Run one synthetic dataset through the complete
    Dataviz Engine recommendation pipeline.
    """

    start = time.perf_counter()

    df = scenario.dataframe

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
    value,
):
    return f"{value:.1f}%"


def format_optional_number(
    value,
    decimals=2,
):
    if value is None:
        return "N/A"

    if pd.isna(value):
        return "N/A"

    return f"{value:.{decimals}f}"


# ==================================================
# REPORT
# ==================================================

def print_report(
    positive_results,
    null_results,
    breakdown,
    total_candidates,
    total_elapsed,
):
    """
    Print a readable terminal benchmark report.
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
        "=" * 72
    )

    print(
        "DATAVIZ ENGINE BENCHMARK"
    )

    print(
        "=" * 72
    )

    print()

    print(
        f"Datasets tested:                  "
        f"{total_datasets}"
    )

    print(
        f"Visualization candidates:         "
        f"{total_candidates}"
    )

    print(
        f"Total runtime:                     "
        f"{total_elapsed:.2f}s"
    )

    print()

    # -----------------------------------
    # Retrieval
    # -----------------------------------

    print(
        "GROUND-TRUTH RELATIONSHIP RETRIEVAL"
    )

    print(
        "-" * 72
    )

    print(
        f"Positive datasets:                 "
        f"{positive_summary['datasets']}"
    )

    print(
        f"Top-1 accuracy:                    "
        f"{format_percentage(positive_summary['top_1_accuracy'])}"
    )

    print(
        f"Top-3 accuracy:                    "
        f"{format_percentage(positive_summary['top_3_accuracy'])}"
    )

    print(
        f"Top-5 accuracy:                    "
        f"{format_percentage(positive_summary['top_5_accuracy'])}"
    )

    print(
        f"Mean reciprocal rank:              "
        f"{positive_summary['mean_reciprocal_rank']:.3f}"
    )

    print(
        f"Median global rank:                "
        f"{format_optional_number(positive_summary['median_global_rank'], 1)}"
    )

    print(
        f"Median chart-type rank:            "
        f"{format_optional_number(positive_summary['median_type_rank'], 1)}"
    )

    print(
        f"Mean expected score:               "
        f"{format_optional_number(positive_summary['mean_expected_score'])}"
    )

    print()

    # -----------------------------------
    # Noise rejection
    # -----------------------------------

    print(
        "NOISE REJECTION"
    )

    print(
        "-" * 72
    )

    print(
        f"Null datasets:                     "
        f"{null_summary['datasets']}"
    )

    print(
        f"Average noise candidate score:     "
        f"{null_summary['mean_candidate_score']:.2f}"
    )

    print(
        f"Average maximum noise score:       "
        f"{null_summary['mean_max_score']:.2f}"
    )

    print(
        f"Highest noise score observed:      "
        f"{null_summary['highest_noise_score']:.2f}"
    )

    print(
        f"Noise candidates scoring >= 50:    "
        f"{format_percentage(null_summary['mean_pct_above_50'])}"
    )

    print(
        f"Noise candidates scoring >= 65:    "
        f"{format_percentage(null_summary['mean_pct_above_65'])}"
    )

    print()

    # -----------------------------------
    # Breakdown
    # -----------------------------------

    print(
        "PER-RELATIONSHIP PERFORMANCE"
    )

    print(
        "-" * 72
    )

    if breakdown.empty:

        print(
            "No positive benchmark results."
        )

    else:

        printable = (
            breakdown.copy()
        )

        printable[
            "top_1_pct"
        ] = printable[
            "top_1_pct"
        ].map(
            lambda value:
            f"{value:.1f}%"
        )

        printable[
            "top_3_pct"
        ] = printable[
            "top_3_pct"
        ].map(
            lambda value:
            f"{value:.1f}%"
        )

        printable[
            "top_5_pct"
        ] = printable[
            "top_5_pct"
        ].map(
            lambda value:
            f"{value:.1f}%"
        )

        printable[
            "mean_reciprocal_rank"
        ] = printable[
            "mean_reciprocal_rank"
        ].map(
            lambda value:
            f"{value:.3f}"
        )

        printable[
            "median_rank"
        ] = printable[
            "median_rank"
        ].map(
            lambda value:
            f"{value:.1f}"
        )

        printable[
            "mean_score"
        ] = printable[
            "mean_score"
        ].map(
            lambda value:
            f"{value:.2f}"
        )

        print(
            printable.to_string(
                index=False
            )
        )

    print()

    print(
        "=" * 72
    )

    print(
        "Raw results saved to evaluation/results/"
    )

    print(
        "=" * 72
    )

    print()


# ==================================================
# BENCHMARK
# ==================================================

def run_benchmark(
    seeds=None,
    n=500,
):
    """
    Run the complete benchmark suite.
    """

    scenarios = build_benchmark_suite(
        seeds=seeds,
        n=n
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
            f"[{index:02d}/{len(scenarios):02d}] "
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
    # Dataframes
    # -----------------------------------

    positive_df = pd.DataFrame(
        positive_results
    )

    null_df = pd.DataFrame(
        null_results
    )

    breakdown = (
        relationship_breakdown(
            positive_results
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

    breakdown.to_csv(
        RESULTS_DIR
        / "relationship_breakdown.csv",
        index=False
    )

    # -----------------------------------
    # Terminal report
    # -----------------------------------

    print_report(
        positive_results=positive_results,
        null_results=null_results,
        breakdown=breakdown,
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
            breakdown
        )
    }


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":

    run_benchmark()