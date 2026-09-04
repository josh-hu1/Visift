import numpy as np
import pandas as pd


# ==================================================
# MATCHING
# ==================================================

def candidate_matches_expected(
    candidate,
    expected
):
    """
    Determine whether a scored candidate matches
    a ground-truth visualization specification.
    """

    if (
        candidate.get(
            "chart"
        )
        != expected.chart
    ):
        return False

    # -----------------------------------
    # Univariate visualization
    # -----------------------------------

    if expected.y is None:

        if (
            candidate.get(
                "x"
            )
            != expected.x
        ):
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
    # Optional aggregation
    # -----------------------------------

    if (
        expected.aggregation
        is not None
    ):

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
    expected_visualizations
):
    """
    Return True when a candidate matches any
    acceptable expected visualization.
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
# POSITIVE SCENARIOS
# ==================================================

def find_expected_rank(
    recommendations,
    expected_visualizations
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
    expected_visualizations
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
    expected_visualizations
):
    """
    Rank the expected recommendation only among
    acceptable chart types.
    """

    accepted_chart_types = {
        expected.chart
        for expected
        in expected_visualizations
    }

    same_type = [
        candidate
        for candidate
        in recommendations
        if (
            candidate.get(
                "chart"
            )
            in accepted_chart_types
        )
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
    elapsed_seconds
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
        "scenario": (
            scenario.name
        ),

        "relationship_type": (
            scenario.relationship_type
        ),

        "rows": len(
            scenario.dataframe
        ),

        "is_null": False,

        "candidate_count": (
            candidate_count
        ),

        "expected_rank": rank,

        "type_specific_rank": (
            type_rank
        ),

        "expected_score": (
            expected_score
        ),

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

        "elapsed_seconds": (
            elapsed_seconds
        )
    }


# ==================================================
# NULL SCENARIOS
# ==================================================

def filter_null_candidates(
    scenario,
    scored_candidates
):
    """
    Restrict null evaluation to chart types that
    should represent a null relationship.

    This avoids penalizing the engine for detecting
    legitimate univariate structure in datasets
    designed to test relational false positives.
    """

    if (
        scenario.null_chart_types
        is None
    ):

        return scored_candidates

    accepted_types = set(
        scenario.null_chart_types
    )

    return [
        candidate
        for candidate
        in scored_candidates
        if (
            candidate.get(
                "chart"
            )
            in accepted_types
        )
    ]


def evaluate_null_scenario(
    scenario,
    scored_candidates,
    elapsed_seconds
):
    """
    Evaluate false-positive behavior for one
    null dataset.
    """

    evaluated = filter_null_candidates(
        scenario,
        scored_candidates
    )

    scores = [
        float(
            candidate.get(
                "score",
                0
            )
        )
        for candidate
        in evaluated
    ]

    evaluated_types = (
        "all"
        if scenario.null_chart_types is None
        else ",".join(
            scenario.null_chart_types
        )
    )

    if not scores:

        return {
            "scenario": (
                scenario.name
            ),

            "relationship_type": (
                scenario.relationship_type
            ),

            "rows": len(
                scenario.dataframe
            ),

            "is_null": True,

            "candidate_count": len(
                scored_candidates
            ),

            "evaluated_candidate_count": 0,

            "evaluated_chart_types": (
                evaluated_types
            ),

            "mean_score": 0,

            "max_score": 0,

            "pct_above_50": 0,

            "pct_above_65": 0,

            "elapsed_seconds": (
                elapsed_seconds
            )
        }

    score_array = np.array(
        scores,
        dtype=float
    )

    return {
        "scenario": (
            scenario.name
        ),

        "relationship_type": (
            scenario.relationship_type
        ),

        "rows": len(
            scenario.dataframe
        ),

        "is_null": True,

        "candidate_count": len(
            scored_candidates
        ),

        "evaluated_candidate_count": (
            len(scores)
        ),

        "evaluated_chart_types": (
            evaluated_types
        ),

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

        "elapsed_seconds": (
            elapsed_seconds
        )
    }


# ==================================================
# POSITIVE SUMMARY
# ==================================================

def summarize_positive_results(
    positive_results
):
    """
    Produce aggregate ground-truth retrieval metrics.
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

    scores = pd.to_numeric(
        df["expected_score"],
        errors="coerce"
    )

    return {
        "datasets": len(
            df
        ),

        "top_1_accuracy": (
            df[
                "hit_at_1"
            ].mean()
            * 100
        ),

        "top_3_accuracy": (
            df[
                "hit_at_3"
            ].mean()
            * 100
        ),

        "top_5_accuracy": (
            df[
                "hit_at_5"
            ].mean()
            * 100
        ),

        "mean_reciprocal_rank": (
            df[
                "reciprocal_rank"
            ].mean()
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
            float(
                scores.mean()
            )
            if scores.notna().any()
            else None
        )
    }


# ==================================================
# NULL SUMMARY
# ==================================================

def summarize_null_results(
    null_results
):
    """
    Aggregate false-positive metrics.
    """

    if not null_results:

        return {}

    df = pd.DataFrame(
        null_results
    )

    return {
        "datasets": len(
            df
        ),

        "mean_candidate_score": (
            df[
                "mean_score"
            ].mean()
        ),

        "mean_max_score": (
            df[
                "max_score"
            ].mean()
        ),

        "highest_noise_score": (
            df[
                "max_score"
            ].max()
        ),

        "mean_pct_above_50": (
            df[
                "pct_above_50"
            ].mean()
        ),

        "mean_pct_above_65": (
            df[
                "pct_above_65"
            ].mean()
        )
    }


# ==================================================
# POSITIVE BREAKDOWN
# ==================================================

def relationship_breakdown(
    positive_results
):
    """
    Calculate retrieval performance separately for
    each planted relationship type.
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
            group[
                "expected_rank"
            ],
            errors="coerce"
        )

        type_ranks = pd.to_numeric(
            group[
                "type_specific_rank"
            ],
            errors="coerce"
        )

        scores = pd.to_numeric(
            group[
                "expected_score"
            ],
            errors="coerce"
        )

        rows.append({
            "relationship_type": (
                relationship_type
            ),

            "datasets": len(
                group
            ),

            "rows_per_dataset": int(
                group[
                    "rows"
                ].median()
            ),

            "top_1_pct": (
                group[
                    "hit_at_1"
                ].mean()
                * 100
            ),

            "top_3_pct": (
                group[
                    "hit_at_3"
                ].mean()
                * 100
            ),

            "top_5_pct": (
                group[
                    "hit_at_5"
                ].mean()
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

            "median_type_rank": (
                type_ranks.median()
                if type_ranks.notna().any()
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
                "top_1_pct",
                "top_3_pct",
                "mean_reciprocal_rank"
            ],
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )


# ==================================================
# NULL BREAKDOWN
# ==================================================

def null_relationship_breakdown(
    null_results
):
    """
    Summarize false-positive behavior separately for
    each type of null scenario.
    """

    if not null_results:

        return pd.DataFrame()

    df = pd.DataFrame(
        null_results
    )

    rows = []

    for (
        relationship_type,
        group
    ) in df.groupby(
        "relationship_type"
    ):

        rows.append({
            "relationship_type": (
                relationship_type
            ),

            "datasets": len(
                group
            ),

            "rows_per_dataset": int(
                group[
                    "rows"
                ].median()
            ),

            "chart_scope": (
                group[
                    "evaluated_chart_types"
                ].iloc[0]
            ),

            "mean_score": (
                group[
                    "mean_score"
                ].mean()
            ),

            "mean_max_score": (
                group[
                    "max_score"
                ].mean()
            ),

            "highest_score": (
                group[
                    "max_score"
                ].max()
            ),

            "pct_above_50": (
                group[
                    "pct_above_50"
                ].mean()
            ),

            "pct_above_65": (
                group[
                    "pct_above_65"
                ].mean()
            )
        })

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "highest_score",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )