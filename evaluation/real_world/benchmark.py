import argparse
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.real_world.expectations import DATASET_DIR, build_real_world_suite

# ==================================================
# CONFIGURATION
# ==================================================

RESULTS_DIR = Path("evaluation/results/real_world_v1")

# Target mode: each dataset's pre-registered target, drivers ranked first.
TARGET_RESULTS_DIR = Path("evaluation/results/real_world_target_v1")

# Effect-size-only baseline (evaluation/real_world/baseline.py), both modes.
BASELINE_RESULTS_DIR = Path("evaluation/results/real_world_baseline_v1")
BASELINE_TARGET_RESULTS_DIR = Path("evaluation/results/real_world_baseline_target_v1")

# The engine at the end of Phase 1, before target-aware ranking (#7), scored
# with this harness against the current expectations. It had no target
# mode, so its plain ranking is the 'before' number for both modes.
PRE_PHASE2_COMMIT = "4aa4729"
PRE_PHASE2_RESULTS_DIR = Path("evaluation/results/real_world_pre_phase2")

RESULTS_DIRS = {
    ("visift", False): RESULTS_DIR,
    ("visift", True): TARGET_RESULTS_DIR,
    ("baseline", False): BASELINE_RESULTS_DIR,
    ("baseline", True): BASELINE_TARGET_RESULTS_DIR,
    ("legacy", False): PRE_PHASE2_RESULTS_DIR,
}

TOP_K_VALUES = (3, 5, 10)

UNMATCHED_RECOMMENDATIONS_TO_SAVE = 10


# ==================================================
# CANDIDATE HELPERS
# ==================================================


def candidate_variables(candidate):
    """
    Return the variables represented by a recommendation.

    Variable order is intentionally ignored for the initial
    relationship match. Optional required_x / required_y fields
    on an expectation can then enforce meaningful orientation.
    """

    variables = []

    if candidate.get("x") is not None:
        variables.append(candidate["x"])

    if candidate.get("y") is not None:
        variables.append(candidate["y"])

    return frozenset(variables)


def chart_variant_matches(variant, expectation):
    """
    Check chart type plus any pre-registered orientation and
    aggregation constraints.
    """

    if variant.get("chart") not in expectation.acceptable_charts:
        return False

    if (
        expectation.aggregation is not None
        and variant.get("aggregation") != expectation.aggregation
    ):
        return False

    if (
        expectation.required_x is not None
        and variant.get("x") != expectation.required_x
    ):
        return False

    if (
        expectation.required_y is not None
        and variant.get("y") != expectation.required_y
    ):
        return False

    return True


def acceptable_chart_present(candidate, expectation):
    """
    Determine whether the primary recommendation or one of its
    diversified alternative views satisfies the expected chart
    form, orientation, and aggregation.
    """

    if chart_variant_matches(candidate, expectation):
        return True

    for alternative in candidate.get("alternative_charts", []):
        # Alternatives may contain only the fields that differ
        # from the primary recommendation. Overlay them so that
        # x/y/aggregation can fall back to the primary candidate.
        variant = dict(candidate)

        variant.update(alternative)

        if chart_variant_matches(variant, expectation):
            return True

    return False


def recommendation_matches_expectation(candidate, expectation):
    """
    Check whether a diversified recommendation surfaces one
    pre-registered real-world insight.
    """

    expected_variables = frozenset(expectation.variables)

    if candidate_variables(candidate) != expected_variables:
        return False

    return acceptable_chart_present(candidate, expectation)


def find_expectation_rank(recommendations, expectation):
    """
    Return the first 1-indexed global rank at which an expected
    insight appears, plus the matched recommendation.
    """

    for rank, candidate in enumerate(recommendations, start=1):
        if recommendation_matches_expectation(candidate, expectation):
            return (rank, candidate)

    return (None, None)


def candidate_label(candidate):
    """
    Produce a compact human-readable label for CSV output.
    """

    chart = candidate.get("chart", "unknown")

    x = candidate.get("x")

    y = candidate.get("y")

    aggregation = candidate.get("aggregation")

    pieces = [chart]

    if aggregation:
        pieces.append(aggregation)

    variable_text = " vs ".join(str(value) for value in (x, y) if value is not None)

    if variable_text:
        pieces.append(variable_text)

    return " | ".join(pieces)


# ==================================================
# SEMANTIC EVALUATION
# ==================================================


def semantic_type_map(profile):
    """
    Map column name to Visift semantic type.
    """

    return {
        column["name"]: column["semantic_type"]
        for column in profile.get("column_profiles", [])
    }


def evaluate_semantics(dataset_spec, profile):
    """
    Compare Visift's inferred semantic types with the
    pre-registered real-world semantic expectations.
    """

    inferred = semantic_type_map(profile)

    rows = []

    for expectation in dataset_spec.expected_semantics:
        inferred_type = inferred.get(expectation.column)

        matches = inferred_type in expectation.acceptable_types

        rows.append(
            {
                "dataset": dataset_spec.name,
                "column": expectation.column,
                "inferred_type": inferred_type,
                "acceptable_types": " | ".join(expectation.acceptable_types),
                "matches": bool(matches),
                "rationale": expectation.rationale,
            }
        )

    semantic_df = pd.DataFrame(rows)

    if semantic_df.empty:
        accuracy = np.nan

    else:
        accuracy = float(semantic_df["matches"].mean())

    return (semantic_df, accuracy)


# ==================================================
# DATASET EVALUATION
# ==================================================


def _legacy_analysis(df):
    """The Phase 1 pipeline: profile, candidates, scores, diversification."""

    from visift.candidate_generator import generate_candidates
    from visift.profiler import profile_dataset
    from visift.scoring.engine import diversify_recommendations, score_all_candidates

    profile = profile_dataset(df)
    candidates = generate_candidates(df, profile)
    scored = score_all_candidates(df, candidates, profile)

    return {
        "profile": profile,
        "candidates": candidates,
        "recommendations": diversify_recommendations(scored),
        "noted_charts": [],
    }


def evaluate_dataset(dataset_spec, target=None, ranker="visift"):
    """
    Run one real dataset through the complete Visift pipeline.

    Unlike the synthetic benchmark, unannotated recommendations
    are NOT labeled false positives. Real datasets do not have
    exhaustive ground truth. Instead, unmatched high-ranking
    recommendations are saved for manual review.

    With a target, candidates include the target-oriented charts and the
    ranking is "drivers of target" first, then everything else. Expected
    insight ranks are measured against that combined list.

    ranker="baseline" ranks the same candidates by plain effect size
    (evaluation/real_world/baseline.py) instead of Visift's scores.
    """

    path = DATASET_DIR / dataset_spec.filename

    if not path.exists():
        raise FileNotFoundError(
            f"Missing real-world dataset: {path}\n"
            "Run:\n"
            "python -m evaluation.real_world.download_datasets"
        )

    df = pd.read_csv(path)

    start = time.perf_counter()

    # Imports are local so the 'legacy' ranker can run against an older
    # visift package that has no pipeline module.
    if ranker == "visift":
        from visift.pipeline import analyze

        analysis = analyze(df, target=target)

    elif ranker == "baseline":
        from evaluation.real_world.baseline import baseline_analysis

        analysis = baseline_analysis(df, target=target)

    elif ranker == "legacy":
        analysis = _legacy_analysis(df)

    else:
        raise ValueError(f"Unknown ranker: {ranker!r}")

    profile = analysis["profile"]

    candidates = analysis["candidates"]

    recommendations = analysis["recommendations"]

    noted_pairs = {
        frozenset((c["x"], c["y"])) for c in analysis["noted_charts"] if "y" in c
    }

    elapsed = time.perf_counter() - start

    (semantic_df, semantic_accuracy) = evaluate_semantics(dataset_spec, profile)

    expectation_rows = []
    matched_recommendation_indices = set()

    for expectation in dataset_spec.expected_insights:
        rank, matched = find_expectation_rank(recommendations, expectation)

        if rank is not None:
            matched_recommendation_indices.add(rank - 1)

        expectation_rows.append(
            {
                "dataset": dataset_spec.name,
                "domain": dataset_spec.domain,
                "expected_insight": expectation.name,
                "variables": " | ".join(expectation.variables),
                "acceptable_charts": " | ".join(expectation.acceptable_charts),
                "required_x": expectation.required_x,
                "required_y": expectation.required_y,
                "required_aggregation": expectation.aggregation,
                "rank": rank,
                "score": (matched.get("score") if matched is not None else np.nan),
                "matched_chart": (
                    matched.get("chart") if matched is not None else None
                ),
                "matched_aggregation": (
                    matched.get("aggregation") if matched is not None else None
                ),
                "in_dataset_notes": frozenset(expectation.variables) in noted_pairs,
                "rationale": expectation.rationale,
            }
        )

    unmatched_rows = []

    for index, candidate in enumerate(recommendations):
        if index in matched_recommendation_indices:
            continue

        rank = index + 1

        unmatched_rows.append(
            {
                "dataset": dataset_spec.name,
                "rank": rank,
                "score": candidate.get("score"),
                "signal": candidate.get("components", {}).get("signal"),
                "chart": candidate.get("chart"),
                "aggregation": candidate.get("aggregation"),
                "x": candidate.get("x"),
                "y": candidate.get("y"),
                "label": candidate_label(candidate),
            }
        )

        if len(unmatched_rows) >= UNMATCHED_RECOMMENDATIONS_TO_SAVE:
            break

    expectation_df = pd.DataFrame(expectation_rows)

    dataset_summary = summarize_dataset(
        dataset_spec=dataset_spec,
        df=df,
        candidates=candidates,
        recommendations=recommendations,
        expectation_df=expectation_df,
        semantic_df=semantic_df,
        semantic_accuracy=semantic_accuracy,
        elapsed_seconds=elapsed,
    )

    return (
        dataset_summary,
        expectation_df,
        pd.DataFrame(unmatched_rows),
        semantic_df,
        profile,
        recommendations,
    )


def summarize_dataset(
    dataset_spec,
    df,
    candidates,
    recommendations,
    expectation_df,
    semantic_df,
    semantic_accuracy,
    elapsed_seconds,
):
    """
    Compute dataset-level real-world retrieval metrics.
    """

    total_expected = len(expectation_df)

    found_mask = expectation_df["rank"].notna()

    found_count = int(found_mask.sum())

    ranks = expectation_df.loc[found_mask, "rank"].astype(float)

    summary = {
        "dataset": dataset_spec.name,
        "domain": dataset_spec.domain,
        "rows": len(df),
        "columns": len(df.columns),
        "candidate_count": len(candidates),
        "recommendation_count": len(recommendations),
        "expected_insights": total_expected,
        "found_anywhere": found_count,
        "coverage_anywhere": (
            found_count / total_expected if total_expected else np.nan
        ),
        "mean_reciprocal_expected_rank": (
            float(
                np.mean(
                    [
                        (1.0 / float(rank) if pd.notna(rank) else 0.0)
                        for rank in expectation_df["rank"]
                    ]
                )
            )
            if total_expected
            else np.nan
        ),
        "median_found_rank": (float(ranks.median()) if not ranks.empty else np.nan),
        "semantic_expectations": len(semantic_df),
        "semantic_matches": (
            int(semantic_df["matches"].sum()) if not semantic_df.empty else 0
        ),
        "semantic_accuracy": (semantic_accuracy),
        "elapsed_seconds": (elapsed_seconds),
    }

    for k in TOP_K_VALUES:
        top_k_found = int(
            (expectation_df["rank"].notna() & (expectation_df["rank"] <= k)).sum()
        )

        summary[f"recall_at_{k}"] = (
            top_k_found / total_expected if total_expected else np.nan
        )

    return summary


# ==================================================
# PROFILE OUTPUT
# ==================================================


def profile_rows(dataset_name, profile):
    """
    Flatten Visift's inferred column profile.
    """

    rows = []

    for column in profile.get("column_profiles", []):
        rows.append(
            {
                "dataset": dataset_name,
                "column": column.get("name"),
                "semantic_type": column.get("semantic_type"),
                "pandas_dtype": column.get("pandas_dtype"),
                "unique_count": column.get("unique_count"),
                "missing_count": column.get("missing_count"),
                "missing_percent": column.get("missing_percent"),
            }
        )

    return rows


# ==================================================
# REPORTING
# ==================================================


def format_pct(value):
    """
    Format a 0-1 proportion as a percentage.
    """

    if pd.isna(value):
        return "N/A"

    return f"{100 * value:.1f}%"


def print_dataset_report(summary, expectation_df, unmatched_df, semantic_df):
    """
    Print one concise real-world evaluation report.
    """

    print()
    print(summary["dataset"].upper())
    print("-" * 80)

    print(f"Rows:                              {summary['rows']}")

    print(f"Columns:                           {summary['columns']}")

    print(f"Candidates generated:              {summary['candidate_count']}")

    print(f"Diversified recommendations:       {summary['recommendation_count']}")

    print(f"Expected insights:                 {summary['expected_insights']}")

    print(f"Recall@3:                          {format_pct(summary['recall_at_3'])}")

    print(f"Recall@5:                          {format_pct(summary['recall_at_5'])}")

    print(f"Recall@10:                         {format_pct(summary['recall_at_10'])}")

    print(
        f"Coverage anywhere:                 {format_pct(summary['coverage_anywhere'])}"
    )

    print(
        f"Mean reciprocal expected rank:     "
        f"{summary['mean_reciprocal_expected_rank']:.3f}"
    )

    if pd.notna(summary["median_found_rank"]):
        print(f"Median found rank:                 {summary['median_found_rank']:.1f}")

    else:
        print("Median found rank:                 N/A")

    print(
        f"Semantic expectation accuracy:     {format_pct(summary['semantic_accuracy'])}"
    )

    print(f"Runtime:                           {summary['elapsed_seconds']:.2f}s")

    print()
    print("SEMANTIC INFERENCE")

    if semantic_df.empty:
        print("No semantic expectations registered.")

    else:
        semantic_printable = semantic_df[
            ["column", "inferred_type", "acceptable_types", "matches"]
        ].copy()

        semantic_printable["matches"] = semantic_printable["matches"].map(
            lambda value: "PASS" if value else "FAIL"
        )

        print(semantic_printable.to_string(index=False))

    print()
    print("EXPECTED INSIGHT RANKS")

    printable = expectation_df[
        ["expected_insight", "variables", "rank", "score", "matched_chart"]
    ].copy()

    printable["rank"] = printable["rank"].map(
        lambda value: f"{int(value)}" if pd.notna(value) else "NOT FOUND"
    )

    printable["score"] = printable["score"].map(
        lambda value: f"{value:.2f}" if pd.notna(value) else "N/A"
    )

    print(printable.to_string(index=False))

    print()
    print("TOP UNMATCHED RECOMMENDATIONS")
    print("(review queue; NOT automatically labeled false positives)")

    if unmatched_df.empty:
        print("None.")

    else:
        printable_unmatched = unmatched_df[["rank", "score", "signal", "label"]].copy()

        for column in ("score", "signal"):
            printable_unmatched[column] = printable_unmatched[column].map(
                lambda value: f"{value:.2f}" if pd.notna(value) else "N/A"
            )

        print(printable_unmatched.to_string(index=False))


def print_dataset_comparison(summaries):
    """
    Print a side-by-side comparison across real datasets.
    """

    summary_df = pd.DataFrame(summaries)

    columns = [
        "dataset",
        "domain",
        "rows",
        "columns",
        "expected_insights",
        "recall_at_3",
        "recall_at_5",
        "recall_at_10",
        "mean_reciprocal_expected_rank",
        "semantic_accuracy",
        "elapsed_seconds",
    ]

    printable = summary_df[columns].copy()

    for column in ("recall_at_3", "recall_at_5", "recall_at_10", "semantic_accuracy"):
        printable[column] = printable[column].map(format_pct)

    printable["mean_reciprocal_expected_rank"] = printable[
        "mean_reciprocal_expected_rank"
    ].map(lambda value: f"{value:.3f}")

    printable["elapsed_seconds"] = printable["elapsed_seconds"].map(
        lambda value: f"{value:.2f}s"
    )

    print("DATASET COMPARISON")
    print("-" * 80)

    print(printable.to_string(index=False))

    print()


def print_suite_report(summaries):
    """
    Print aggregate metrics across all registered real-world
    datasets.
    """

    summary_df = pd.DataFrame(summaries)

    print()
    print("=" * 80)
    print("VISIFT REAL-WORLD EVALUATION")
    print("=" * 80)

    print()
    print(f"Datasets evaluated:                {len(summary_df)}")

    print(
        f"Total expected insights:           "
        f"{int(summary_df['expected_insights'].sum())}"
    )

    expected_total = summary_df["expected_insights"].sum()

    for k in TOP_K_VALUES:
        recovered = 0

        for _, row in summary_df.iterrows():
            recovered += row[f"recall_at_{k}"] * row["expected_insights"]

        weighted_recall = recovered / expected_total if expected_total else np.nan

        print(
            f"Overall Recall@{k}:"
            f"{' ' * max(1, 22 - len(str(k)))}"
            f"{format_pct(weighted_recall)}"
        )

    overall_mrr = np.average(
        summary_df["mean_reciprocal_expected_rank"],
        weights=summary_df["expected_insights"],
    )

    print(f"Mean reciprocal expected rank:     {overall_mrr:.3f}")

    semantic_total = int(summary_df["semantic_expectations"].sum())

    semantic_matches = int(summary_df["semantic_matches"].sum())

    overall_semantic_accuracy = (
        semantic_matches / semantic_total if semantic_total else np.nan
    )

    print(f"Semantic expectation accuracy:     {format_pct(overall_semantic_accuracy)}")

    print(
        f"Total runtime:                     {summary_df['elapsed_seconds'].sum():.2f}s"
    )

    print()

    print_dataset_comparison(summaries)


# ==================================================
# SAVE RESULTS
# ==================================================


def save_results(
    summaries,
    expectation_frames,
    unmatched_frames,
    semantic_frames,
    profile_records,
    results_dir=RESULTS_DIR,
):
    """
    Save raw real-world benchmark outputs for inspection.
    """

    results_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summaries)

    expectation_df = pd.concat(expectation_frames, ignore_index=True)

    unmatched_df = pd.concat(unmatched_frames, ignore_index=True)

    semantic_df = pd.concat(semantic_frames, ignore_index=True)

    profile_df = pd.DataFrame(profile_records)

    summary_df.to_csv(results_dir / "dataset_summary.csv", index=False)

    expectation_df.to_csv(results_dir / "expected_insight_ranks.csv", index=False)

    unmatched_df.to_csv(results_dir / "unmatched_top_recommendations.csv", index=False)

    semantic_df.to_csv(results_dir / "semantic_expectations.csv", index=False)

    profile_df.to_csv(results_dir / "inferred_column_profiles.csv", index=False)

    return (summary_df, expectation_df, unmatched_df, semantic_df, profile_df)


# ==================================================
# BENCHMARK
# ==================================================


def run_benchmark(target_mode=False, ranker="visift"):
    """
    Run the complete registered real-world benchmark suite.

    target_mode=False reproduces the original benchmark exactly.
    target_mode=True gives each dataset its pre-registered target.
    ranker: "visift" (default), "baseline" (effect size only), or "legacy"
    (whatever visift package is importable; used by run_pre_phase2).
    """

    suite = build_real_world_suite()

    results_dir = RESULTS_DIRS[(ranker, target_mode)]

    summaries = []
    expectation_frames = []
    unmatched_frames = []
    semantic_frames = []
    profile_records = []

    print()
    print(
        f"Running real-world benchmark [{ranker}]"
        + (" (target mode)..." if target_mode else "...")
    )

    for index, dataset_spec in enumerate(suite, start=1):
        target = dataset_spec.target if target_mode else None

        print(
            f"[{index:02d}/{len(suite):02d}] {dataset_spec.name}"
            + (f" (target: {target})" if target_mode else "")
        )

        (summary, expectation_df, unmatched_df, semantic_df, profile, _) = (
            evaluate_dataset(dataset_spec, target=target, ranker=ranker)
        )

        if target_mode:
            summary["target"] = target

        summaries.append(summary)

        expectation_frames.append(expectation_df)

        unmatched_frames.append(unmatched_df)

        semantic_frames.append(semantic_df)

        profile_records.extend(profile_rows(dataset_spec.name, profile))

        print_dataset_report(summary, expectation_df, unmatched_df, semantic_df)

    print_suite_report(summaries)

    save_results(
        summaries=summaries,
        expectation_frames=expectation_frames,
        unmatched_frames=unmatched_frames,
        semantic_frames=semantic_frames,
        profile_records=profile_records,
        results_dir=results_dir,
    )

    print(f"Raw results saved to {results_dir}/")

    print()

    return summaries


def compare_modes():
    """
    Side-by-side no-target vs target-mode results from the saved CSVs.
    """

    metrics = [
        "recall_at_3",
        "recall_at_5",
        "recall_at_10",
        "mean_reciprocal_expected_rank",
    ]

    before = pd.read_csv(RESULTS_DIR / "dataset_summary.csv")
    after = pd.read_csv(TARGET_RESULTS_DIR / "dataset_summary.csv")

    summary = before[["dataset", "expected_insights", *metrics]].merge(
        after[["dataset", "target", *metrics]],
        on="dataset",
        suffixes=("_no_target", "_target"),
    )

    ranks = pd.read_csv(RESULTS_DIR / "expected_insight_ranks.csv")[
        ["dataset", "expected_insight", "rank"]
    ].merge(
        pd.read_csv(TARGET_RESULTS_DIR / "expected_insight_ranks.csv")[
            ["dataset", "expected_insight", "rank"]
        ],
        on=["dataset", "expected_insight"],
        suffixes=("_no_target", "_target"),
    )

    summary.to_csv(TARGET_RESULTS_DIR / "mode_comparison.csv", index=False)
    ranks.to_csv(TARGET_RESULTS_DIR / "rank_comparison.csv", index=False)

    printable = summary.copy()

    for column in printable.columns:
        if column.startswith("recall_at_"):
            printable[column] = printable[column].map(format_pct)

        elif column.startswith("mean_reciprocal"):
            printable[column] = printable[column].map(lambda value: f"{value:.3f}")

    print("=" * 80)
    print("NO-TARGET VS TARGET MODE")
    print("=" * 80)
    print(printable.to_string(index=False))
    print()
    print(ranks.to_string(index=False))
    print()

    return summary, ranks


def run_pre_phase2():
    """
    Score the Phase 1 engine (PRE_PHASE2_COMMIT) with this harness and the
    current expectations: export that commit's src/visift to a temporary
    directory and run the 'legacy' ranker in a subprocess that imports it
    first.
    """

    archive = subprocess.run(
        ["git", "archive", PRE_PHASE2_COMMIT, "src/visift"],
        check=True,
        capture_output=True,
    ).stdout

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(fileobj=BytesIO(archive)) as tar:
            if hasattr(tarfile, "data_filter"):  # Python >= 3.12 / patched 3.10+
                tar.extractall(tmp, filter="data")
            else:
                tar.extractall(tmp)  # trusted: our own git archive

        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join([str(Path(tmp) / "src"), os.getcwd()]),
        }

        subprocess.run(
            [
                sys.executable,
                "-c",
                "import visift, evaluation.real_world.benchmark as b; "
                f"assert visift.__file__.startswith({tmp!r}); "
                "b.run_benchmark(target_mode=False, ranker='legacy')",
            ],
            check=True,
            env=env,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--mode",
        choices=("no-target", "target", "both"),
        default="both",
        help="no-target: original benchmark; target: pre-registered targets; "
        "both (default): run both and compare.",
    )

    parser.add_argument(
        "--ranker",
        choices=("visift", "baseline", "pre-phase2", "all"),
        default="visift",
        help="visift (default); baseline: effect size only; pre-phase2: the "
        f"engine at {PRE_PHASE2_COMMIT}; all: everything, then the comparison "
        "table (python -m evaluation.real_world.report).",
    )

    args = parser.parse_args()
    mode = args.mode

    rankers = (
        ("visift", "baseline", "pre-phase2") if args.ranker == "all" else (args.ranker,)
    )

    for ranker in rankers:
        if ranker == "pre-phase2":
            run_pre_phase2()
            continue

        if mode in ("no-target", "both"):
            run_benchmark(target_mode=False, ranker=ranker)

        if mode in ("target", "both"):
            run_benchmark(target_mode=True, ranker=ranker)

    if mode == "both" and "visift" in rankers:
        compare_modes()

    if args.ranker == "all":
        from evaluation.real_world.report import main as report

        report()


if __name__ == "__main__":
    main()
