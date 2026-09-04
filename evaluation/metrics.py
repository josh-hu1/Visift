import numpy as np
import pandas as pd


# ==================================================
# MATCHING
# ==================================================

def candidate_matches_expected(
    candidate,
    expected,
):
    """
    Determine whether a scored candidate matches
    a ground-truth visualization specification.
    """

    if candidate.get("chart") != expected.chart:
        return False

    # -----------------------------------
    # Univariate visualization
    # -----------------------------------

    if expected.y is None:

        if candidate.get("x") != expected.x:
            return False

    # -----------------------------------
    # Two-variable visualization
    # -----------------------------------

    else:

        candidate_x = candidate.get(
            "x"
        )

        candidate_y = candidate.get(
            "y"
        )

        if expected.unordered:

            candidate_columns = {
                candidate_x,
                candidate_y
            }

            expected_columns = {
                expected.x,
                expected.y
            }

            if (
                candidate_columns
                != expected_columns
            ):
                return False

        else:

            if candidate_x != expected.x:
                return False

            if candidate_y != expected.y:
                return False

    # -----------------------------------
    # Optional aggregation requirement
    # -----------------------------------

    if expected.aggregation is not None:

        if (
            candidate.get(
                "aggregation"
            )
            != expected.aggregation
        ):
            return False

    return True


def matches_any_expected(
    candidate,
    expected_visualizations,
):
    """
    Return True when a candidate matches any
    acceptable ground-truth visualization.
    """

    return any(
        candidate_matches_expected(
            candidate,
            expected
        )
        for expected
        in expected_visualizations
    )


# ==================================================
# POSITIVE SCENARIO METRICS
# ==================================================

def find_expected_rank(
    recommendations,
    expected_visualizations,
):
    """
    Find the best global ranking position of an
    acceptable expected visualization.
    """

    for rank, candidate in enumerate(
        recommendations,
        start=1
    ):

        if matches_any_expected(
            candidate,
            expected_visualizations
        ):

            return rank

    return None


def find_expected_score(
    recommendations,
    expected_visualizations,
):
    """
    Return the score of the highest-ranked
    acceptable expected visualization.
    """

    for candidate in recommendations:

        if matches_any_expected(
            candidate,
            expected_visualizations
        ):

            return candidate.get(
                "score"
            )

    return None


def find_type_specific_rank(
    recommendations,
    expected_visualizations,
):
    """
    Rank the target only among chart types that
    could satisfy the expected result.

    This is particularly useful for histogram
    benchmarks where strong unrelated relational
    charts may legitimately rank above the
    histogram globally.
    """

    accepted_chart_types = {
        expected.chart
        for expected
        in expected_visualizations
    }

    same_type = [
        candidate
        for candidate in recommendations
        if candidate.get("chart")
        in accepted_chart_types
    ]

    for rank, candidate in enumerate(
        same_type,
        start=1
    ):

        if matches_any_expected(
            candidate,
            expected_visualizations
        ):

            return rank

    return None


def evaluate_positive_scenario(
    scenario,
    recommendations,
    candidate_count,
    elapsed_seconds,
):
    """
    Evaluate one scenario containing a known
    planted relationship.
    """

    rank = find_expected_rank(
        recommendations,
        scenario.expected
    )

    type_rank = (
        find_type_specific_rank(
            recommendations,
            scenario.expected
        )
    )

    expected_score = (
        find_expected_score(
            recommendations,
            scenario.expected
        )
    )

    return {
        "scenario": scenario.name,
        "relationship_type": (
            scenario.relationship_type
        ),
        "is_null": False,
        "candidate_count": candidate_count,
        "expected_rank": rank,
        "type_specific_rank": type_rank,
        "expected_score": expected_score,
        "hit_at_1": (
            rank is not None
            and rank <= 1
        ),
        "hit_at_3": (
            rank is not None
            and rank <= 3
        ),
        "hit_at_5": (
            rank is not None
            and rank <= 5
        ),
        "reciprocal_rank": (
            1 / rank
            if rank is not None
            else 0
        ),
        "elapsed_seconds": elapsed_seconds
    }


# ==================================================
# NULL SCENARIO METRICS
# ==================================================

def evaluate_null_scenario(
    scenario,
    scored_candidates,
    elapsed_seconds,
):
    """
    Evaluate false-positive behavior on a dataset
    containing only independent noise.
    """

    scores = [
        float(
            candidate.get(
                "score",
                0
            )
        )
        for candidate
        in scored_candidates
    ]

    if not scores:

        return {
            "scenario": scenario.name,
            "relationship_type": (
                scenario.relationship_type
            ),
            "is_null": True,
            "candidate_count": 0,
            "mean_score": 0,
            "max_score": 0,
            "pct_above_50": 0,
            "pct_above_65": 0,
            "elapsed_seconds": elapsed_seconds
        }

    score_array = np.array(
        scores,
        dtype=float
    )

    return {
        "scenario": scenario.name,
        "relationship_type": (
            scenario.relationship_type
        ),
        "is_null": True,
        "candidate_count": len(scores),
        "mean_score": float(
            score_array.mean()
        ),
        "max_score": float(
            score_array.max()
        ),
        "pct_above_50": float(
            (
                score_array >= 50
            ).mean()
            * 100
        ),
        "pct_above_65": float(
            (
                score_array >= 65
            ).mean()
            * 100
        ),
        "elapsed_seconds": elapsed_seconds
    }


# ==================================================
# SUMMARY
# ==================================================

def summarize_positive_results(
    positive_results,
):
    """
    Produce aggregate retrieval metrics.
    """

    if not positive_results:

        return {}

    df = pd.DataFrame(
        positive_results
    )

    ranks = pd.to_numeric(
        df["expected_rank"],
        errors="coerce"
    )

    type_ranks = pd.to_numeric(
        df["type_specific_rank"],
        errors="coerce"
    )

    return {
        "datasets": len(df),
        "top_1_accuracy": (
            df["hit_at_1"].mean()
            * 100
        ),
        "top_3_accuracy": (
            df["hit_at_3"].mean()
            * 100
        ),
        "top_5_accuracy": (
            df["hit_at_5"].mean()
            * 100
        ),
        "mean_reciprocal_rank": (
            df["reciprocal_rank"].mean()
        ),
        "median_global_rank": (
            float(
                ranks.median()
            )
            if ranks.notna().any()
            else None
        ),
        "median_type_rank": (
            float(
                type_ranks.median()
            )
            if type_ranks.notna().any()
            else None
        ),
        "mean_expected_score": (
            pd.to_numeric(
                df["expected_score"],
                errors="coerce"
            ).mean()
        )
    }


def summarize_null_results(
    null_results,
):
    """
    Aggregate noise rejection metrics.
    """

    if not null_results:

        return {}

    df = pd.DataFrame(
        null_results
    )

    return {
        "datasets": len(df),
        "mean_candidate_score": (
            df["mean_score"].mean()
        ),
        "mean_max_score": (
            df["max_score"].mean()
        ),
        "highest_noise_score": (
            df["max_score"].max()
        ),
        "mean_pct_above_50": (
            df["pct_above_50"].mean()
        ),
        "mean_pct_above_65": (
            df["pct_above_65"].mean()
        )
    }


def relationship_breakdown(
    positive_results,
):
    """
    Calculate retrieval performance separately
    for each relationship type.
    """

    if not positive_results:

        return pd.DataFrame()

    df = pd.DataFrame(
        positive_results
    )

    rows = []

    for (
        relationship_type,
        group
    ) in df.groupby(
        "relationship_type"
    ):

        ranks = pd.to_numeric(
            group["expected_rank"],
            errors="coerce"
        )

        scores = pd.to_numeric(
            group["expected_score"],
            errors="coerce"
        )

        rows.append({
            "relationship_type": (
                relationship_type
            ),
            "datasets": len(group),
            "top_1_pct": (
                group["hit_at_1"].mean()
                * 100
            ),
            "top_3_pct": (
                group["hit_at_3"].mean()
                * 100
            ),
            "top_5_pct": (
                group["hit_at_5"].mean()
                * 100
            ),
            "mean_reciprocal_rank": (
                group[
                    "reciprocal_rank"
                ].mean()
            ),
            "median_rank": (
                ranks.median()
                if ranks.notna().any()
                else np.nan
            ),
            "mean_score": (
                scores.mean()
            )
        })

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "top_3_pct",
                "mean_reciprocal_rank"
            ],
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )