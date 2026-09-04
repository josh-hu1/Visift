from scoring.bar import score_bar_chart
from scoring.scatter import score_scatter_chart
from scoring.line import score_line_chart
from scoring.histogram import score_histogram
from scoring.box import score_box_chart


SCORERS = {
    "bar": score_bar_chart,
    "scatter": score_scatter_chart,
    "line": score_line_chart,
    "histogram": score_histogram,
    "box": score_box_chart
}


def score_candidate(
    df,
    candidate,
    profile
):
    """
    Score a single visualization candidate using
    the appropriate chart-specific scoring function.
    """

    chart_type = candidate["chart"]

    scoring_function = SCORERS.get(
        chart_type
    )

    if scoring_function is None:
        return None

    result = scoring_function(
        df,
        candidate,
        profile
    )

    return {
        **candidate,
        **result
    }


def score_all_candidates(
    df,
    candidates,
    profile
):
    """
    Score every supported visualization candidate
    and return one globally ranked list.
    """

    scored = []

    for candidate in candidates:

        result = score_candidate(
            df,
            candidate,
            profile
        )

        if result is not None:
            scored.append(result)

    scored.sort(
        key=lambda item: item["score"],
        reverse=True
    )

    return scored


def candidate_signature(candidate):
    """
    Create a signature representing the variables
    involved in a visualization.

    Visualizations using the same variables are
    considered alternate views of the same
    underlying relationship.

    Examples:

        revenue vs rating
        rating vs revenue

    both become:

        ('rating', 'revenue')
    """

    variables = []

    if "x" in candidate:
        variables.append(
            candidate["x"]
        )

    if "y" in candidate:
        variables.append(
            candidate["y"]
        )

    return tuple(
        sorted(variables)
    )


def diversify_recommendations(
    recommendations
):
    """
    Remove redundant recommendations involving
    the exact same variable combination.

    The highest-scoring visualization is kept as
    the primary recommendation.

    Lower-scoring chart types using the same
    variables are stored as alternatives.
    """

    diversified = []

    signature_to_index = {}

    for candidate in recommendations:

        signature = candidate_signature(
            candidate
        )

        if signature not in signature_to_index:

            candidate_copy = dict(
                candidate
            )

            candidate_copy[
                "alternative_charts"
            ] = []

            signature_to_index[
                signature
            ] = len(diversified)

            diversified.append(
                candidate_copy
            )

        else:

            primary_index = (
                signature_to_index[
                    signature
                ]
            )

            primary = diversified[
                primary_index
            ]

            primary[
                "alternative_charts"
            ].append({
                "chart": candidate["chart"],
                "chart_name": chart_display_name(
                    candidate["chart"]
                ),
                "title": visualization_title(
                    candidate
                ),
                "score": candidate["score"]
            })

    return diversified


def visualization_title(candidate):
    """
    Generate a human-readable title for a
    visualization candidate.
    """

    chart = candidate["chart"]

    if chart == "scatter":

        return (
            f"{candidate['x']} vs "
            f"{candidate['y']}"
        )

    if chart == "line":

        return (
            f"{candidate['y']} over "
            f"{candidate['x']}"
        )

    if chart == "histogram":

        return (
            f"{candidate['x']} distribution"
        )

    if chart == "box":

        return (
            f"{candidate['y']} by "
            f"{candidate['x']}"
        )

    if chart == "bar":

        if "y" in candidate:

            return (
                f"{candidate['y']} by "
                f"{candidate['x']}"
            )

        return (
            f"{candidate['x']} distribution"
        )

    return chart


def chart_display_name(chart_type):
    """
    Convert internal chart identifiers into
    user-facing chart names.
    """

    names = {
        "bar": "Bar chart",
        "scatter": "Scatterplot",
        "line": "Line chart",
        "histogram": "Histogram",
        "box": "Box plot"
    }

    return names.get(
        chart_type,
        chart_type
    )


def recommendation_strength(score):
    """
    Convert a numerical recommendation score
    into a descriptive category.

    Thresholds are provisional and can be
    calibrated later.
    """

    if score >= 80:
        return "Excellent"

    if score >= 65:
        return "Strong"

    if score >= 50:
        return "Moderate"

    if score >= 35:
        return "Weak"

    return "Very weak"


def summary_reasons(candidate):
    """
    Return the most useful explanation lines for
    the global recommendation view.

    Different chart types have different important
    statistics, so we select reasons accordingly.
    """

    chart = candidate["chart"]

    reasons = candidate[
        "reasons"
    ]

    # -----------------------------------
    # Scatterplot
    # -----------------------------------

    if chart == "scatter":

        statistics = candidate[
            "statistics"
        ]

        return [
            (
                "Pearson correlation: "
                f"{statistics['pearson']:.3f}."
            ),
            (
                "Spearman correlation: "
                f"{statistics['spearman']:.3f}."
            ),
            (
                "Relationship signal: "
                f"{candidate['components']['signal']:.1f}/100."
            )
        ]

    # -----------------------------------
    # Line chart
    # -----------------------------------

    if chart == "line":

        statistics = candidate[
            "statistics"
        ]

        return [
            (
                "Trend direction: "
                f"{statistics['direction']}."
            ),
            (
                "Pearson time correlation: "
                f"{statistics['pearson']:.3f}."
            ),
            (
                "Trend signal: "
                f"{candidate['components']['signal']:.1f}/100."
            )
        ]

    # -----------------------------------
    # Histogram
    # -----------------------------------

    if chart == "histogram":

        statistics = candidate[
            "statistics"
        ]

        return [
            (
                "Skewness: "
                f"{statistics['skewness']:.3f}."
            ),
            (
                "Potential outlier rate: "
                f"{statistics['outlier_rate'] * 100:.1f}%."
            ),
            (
                "Distribution signal: "
                f"{candidate['components']['signal']:.1f}/100."
            )
        ]

    # -----------------------------------
    # Box plot
    # -----------------------------------

    if chart == "box":

        statistics = candidate[
            "statistics"
        ]

        return [
            (
                "Group separation: "
                f"{statistics['eta_squared_score']:.1f}/100."
            ),
            (
                "Median separation: "
                f"{statistics['median_separation_score']:.1f}/100."
            ),
            (
                "Distribution signal: "
                f"{candidate['components']['signal']:.1f}/100."
            )
        ]

    # -----------------------------------
    # Bar chart
    # -----------------------------------

    if chart == "bar":

        if "y" in candidate:

            return [
                reasons[-3],
                reasons[-2],
                reasons[-1]
            ]

        return [
            reasons[-3],
            reasons[-2],
            reasons[-1]
        ]

    return reasons[-3:]