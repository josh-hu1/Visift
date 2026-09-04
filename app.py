import pandas as pd

from profiler import profile_dataset
from candidate_generator import generate_candidates

from scoring.engine import (
    score_all_candidates,
    diversify_recommendations,
    visualization_title,
    chart_display_name,
    recommendation_strength,
    summary_reasons
)


def print_dataset_summary(
    profile
):
    """
    Print a concise summary of the dataset
    and inferred semantic types.
    """

    print()
    print("DATASET SUMMARY")
    print("=" * 60)

    print(
        f"Rows: {profile['rows']}"
    )

    print(
        f"Columns: {profile['columns']}"
    )

    print()
    print("Column types:")

    for column in (
        profile["column_profiles"]
    ):

        print(
            f"  {column['name']:<20} "
            f"{column['semantic_type']}"
        )


def print_candidate_summary(
    candidates
):
    """
    Print the number of generated visualization
    candidates by chart type.
    """

    counts = {}

    for candidate in candidates:

        chart = candidate["chart"]

        counts[chart] = (
            counts.get(chart, 0)
            + 1
        )

    print()
    print("CANDIDATES GENERATED")
    print("=" * 60)

    print(
        f"Total: {len(candidates)}"
    )

    for chart, count in sorted(
        counts.items()
    ):

        print(
            f"  {chart_display_name(chart):<15} "
            f"{count}"
        )


def print_top_recommendations(
    recommendations,
    top_n=15
):
    """
    Print the highest-ranked diversified
    visualization recommendations.
    """

    print()

    print(
        "TOP VISUALIZATION RECOMMENDATIONS"
    )

    print("=" * 60)

    for rank, candidate in enumerate(
        recommendations[:top_n],
        start=1
    ):

        title = visualization_title(
            candidate
        )

        chart_name = chart_display_name(
            candidate["chart"]
        )

        strength = recommendation_strength(
            candidate["score"]
        )

        print()

        print(
            f"{rank}. {title}"
        )

        print(
            f"   Type: {chart_name}"
        )

        print(
            f"   Score: "
            f"{candidate['score']}/100"
        )

        print(
            f"   Recommendation: "
            f"{strength}"
        )

        print(
            f"   Signal: "
            f"{candidate['components']['signal']}/100"
        )

        print(
            f"   Visualization quality: "
            f"{candidate['components']['visualization_quality']}/100"
        )

        print(
            "   Why:"
        )

        for reason in summary_reasons(
            candidate
        ):

            print(
                f"      - {reason}"
            )

        alternatives = candidate.get(
            "alternative_charts",
            []
        )

        if alternatives:

            print(
                "   Alternative views:"
            )

            for alternative in alternatives:

                print(
                    f"      - "
                    f"{alternative['chart_name']} "
                    f"({alternative['score']}/100)"
                )


def main():
    """
    Run the visualization recommendation engine.
    """

    file_path = (
        "synthetic_sales.csv"
    )

    # -----------------------------------
    # Load dataset
    # -----------------------------------

    df = pd.read_csv(
        file_path
    )

    # -----------------------------------
    # Profile dataset
    # -----------------------------------

    profile = profile_dataset(
        df
    )

    # -----------------------------------
    # Generate valid candidates
    # -----------------------------------

    candidates = generate_candidates(
        df,
        profile
    )

    # -----------------------------------
    # Score all candidates
    # -----------------------------------

    scored_recommendations = (
        score_all_candidates(
            df,
            candidates,
            profile
        )
    )

    # -----------------------------------
    # Remove redundant recommendations
    # -----------------------------------

    recommendations = (
        diversify_recommendations(
            scored_recommendations
        )
    )

    # -----------------------------------
    # Output
    # -----------------------------------

    print_dataset_summary(
        profile
    )

    print_candidate_summary(
        candidates
    )

    print_top_recommendations(
        recommendations,
        top_n=15
    )


if __name__ == "__main__":
    main()