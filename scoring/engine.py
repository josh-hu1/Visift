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
    into Visift's calibrated recommendation bands.
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


def _overall_signal(candidate):
    """
    Return the final chart-specific insight signal
    used by Visift's global ranking layer.
    """

    return candidate.get(
        "components",
        {}
    ).get(
        "signal",
        0
    )


def _safe_float(
    mapping,
    key,
    default=0
):
    """
    Read a numeric statistic defensively so the
    explanation layer remains compatible with older
    saved benchmark outputs and partial scorer
    results.
    """

    value = mapping.get(
        key,
        default
    )

    try:
        return float(
            value
        )
    except (
        TypeError,
        ValueError
    ):
        return float(
            default
        )


def _format_pattern_name(pattern):
    """
    Convert internal pattern identifiers into
    concise user-facing labels.
    """

    names = {
        "trend": "Trend",
        "seasonality": "Seasonality",
        "level_shift": "Level shift",
        "volatility_shift": "Volatility shift"
    }

    return names.get(
        pattern,
        str(
            pattern
        ).replace(
            "_",
            " "
        ).title()
    )


def summary_reasons(candidate):
    """
    Return the most useful explanation lines for
    the global recommendation view.

    The explanation should describe the evidence
    that actually drove the chart's score, rather
    than always displaying the same generic metrics.
    """

    chart = candidate[
        "chart"
    ]

    reasons = candidate.get(
        "reasons",
        []
    )

    signal = _overall_signal(
        candidate
    )

    # -----------------------------------
    # Scatterplot
    # -----------------------------------

    if chart == "scatter":

        statistics = candidate.get(
            "statistics",
            {}
        )

        linear_signal = _safe_float(
            statistics,
            "linear_signal"
        )

        nonlinear_signal = _safe_float(
            statistics,
            "nonlinear_signal"
        )

        if (
            nonlinear_signal
            > linear_signal
        ):

            return [
                (
                    "Dominant relationship: "
                    "nonlinear dependence."
                ),
                (
                    "Nonlinear signal: "
                    f"{nonlinear_signal:.1f}/100 "
                    f"(MI={_safe_float(statistics, 'mutual_information'):.3f})."
                ),
                (
                    "Overall relationship signal: "
                    f"{signal:.1f}/100."
                )
            ]

        selected_method = statistics.get(
            "selected_correlation_method",
            "correlation"
        )

        if selected_method == "pearson":
            selected_value = _safe_float(
                statistics,
                "pearson"
            )
            method_label = "Pearson"
        elif selected_method == "spearman":
            selected_value = _safe_float(
                statistics,
                "spearman"
            )
            method_label = "Spearman"
        else:
            pearson = _safe_float(
                statistics,
                "pearson"
            )
            spearman = _safe_float(
                statistics,
                "spearman"
            )

            if abs(
                pearson
            ) >= abs(
                spearman
            ):
                selected_value = pearson
                method_label = "Pearson"
            else:
                selected_value = spearman
                method_label = "Spearman"

        reliability = _safe_float(
            statistics,
            "linear_reliability",
            1
        )

        return [
            (
                "Dominant relationship: "
                "linear/monotonic association."
            ),
            (
                f"{method_label} correlation: "
                f"{selected_value:.3f} "
                f"(reliability {reliability * 100:.0f}%)."
            ),
            (
                "Overall relationship signal: "
                f"{signal:.1f}/100."
            )
        ]

    # -----------------------------------
    # Line chart
    # -----------------------------------

    if chart == "line":

        statistics = candidate.get(
            "statistics",
            {}
        )

        dominant = statistics.get(
            "dominant_pattern",
            "trend"
        )

        pattern_label = (
            _format_pattern_name(
                dominant
            )
        )

        if dominant == "seasonality":

            period = _safe_float(
                statistics,
                "seasonal_period_temporal_units"
            )

            autocorrelation = _safe_float(
                statistics,
                "seasonal_autocorrelation"
            )

            return [
                (
                    "Dominant temporal pattern: "
                    f"{pattern_label}."
                ),
                (
                    "Estimated repeating period: "
                    f"{period:.1f} temporal units "
                    f"(autocorrelation {autocorrelation:.3f})."
                ),
                (
                    "Overall temporal signal: "
                    f"{signal:.1f}/100."
                )
            ]

        if dominant == "level_shift":

            split_fraction = _safe_float(
                statistics,
                "level_shift_split_fraction"
            )

            effect_size = _safe_float(
                statistics,
                "level_shift_effect_size"
            )

            reliability = _safe_float(
                statistics,
                "level_shift_reliability"
            )

            return [
                (
                    "Dominant temporal pattern: "
                    f"{pattern_label}."
                ),
                (
                    "Abrupt change detected around "
                    f"{split_fraction * 100:.0f}% through the series "
                    f"(effect {effect_size:.2f}, "
                    f"reliability {reliability * 100:.0f}%)."
                ),
                (
                    "Overall temporal signal: "
                    f"{signal:.1f}/100."
                )
            ]

        if dominant == "volatility_shift":

            split_fraction = _safe_float(
                statistics,
                "volatility_split_fraction"
            )

            dispersion_ratio = _safe_float(
                statistics,
                "volatility_dispersion_ratio"
            )

            reliability = _safe_float(
                statistics,
                "volatility_reliability"
            )

            return [
                (
                    "Dominant temporal pattern: "
                    f"{pattern_label}."
                ),
                (
                    "Dispersion changes by about "
                    f"{dispersion_ratio:.2f}× around "
                    f"{split_fraction * 100:.0f}% through the series "
                    f"(reliability {reliability * 100:.0f}%)."
                ),
                (
                    "Overall temporal signal: "
                    f"{signal:.1f}/100."
                )
            ]

        direction = statistics.get(
            "direction",
            "flat"
        )

        pearson = _safe_float(
            statistics,
            "pearson"
        )

        spearman = _safe_float(
            statistics,
            "spearman"
        )

        strongest = (
            pearson
            if abs(
                pearson
            ) >= abs(
                spearman
            )
            else spearman
        )

        reliability = _safe_float(
            statistics,
            "trend_reliability",
            1
        )

        return [
            (
                "Dominant temporal pattern: "
                f"{pattern_label} ({direction})."
            ),
            (
                "Strongest time correlation: "
                f"{strongest:.3f} "
                f"(reliability {reliability * 100:.0f}%)."
            ),
            (
                "Overall temporal signal: "
                f"{signal:.1f}/100."
            )
        ]

    # -----------------------------------
    # Histogram
    # -----------------------------------

    if chart == "histogram":

        statistics = candidate.get(
            "statistics",
            {}
        )

        shape_score = _safe_float(
            statistics,
            "shape_score"
        )

        outlier_score = _safe_float(
            statistics,
            "outlier_score"
        )

        multimodality_score = _safe_float(
            statistics,
            "multimodality_score"
        )

        dominant = max(
            [
                (
                    "shape",
                    shape_score
                ),
                (
                    "outliers",
                    outlier_score
                ),
                (
                    "multimodality",
                    multimodality_score
                )
            ],
            key=lambda item:
                item[1]
        )[0]

        if dominant == "multimodality":

            separation = _safe_float(
                statistics,
                "multimodality_component_separation"
            )

            component_weight = _safe_float(
                statistics,
                "multimodality_smaller_component_weight"
            )

            return [
                (
                    "Dominant distribution pattern: "
                    "multiple modes."
                ),
                (
                    "Mode separation: "
                    f"{separation:.2f} SD; smaller component "
                    f"{component_weight * 100:.0f}% of observations."
                ),
                (
                    "Overall distribution signal: "
                    f"{signal:.1f}/100."
                )
            ]

        if dominant == "outliers":

            outlier_rate = _safe_float(
                statistics,
                "outlier_rate"
            )

            return [
                (
                    "Dominant distribution pattern: "
                    "unusual tail observations."
                ),
                (
                    "Potential outlier rate: "
                    f"{outlier_rate * 100:.1f}%."
                ),
                (
                    "Overall distribution signal: "
                    f"{signal:.1f}/100."
                )
            ]

        tail_asymmetry = _safe_float(
            statistics,
            "tail_asymmetry"
        )

        skewness = _safe_float(
            statistics,
            "skewness"
        )

        return [
            (
                "Dominant distribution pattern: "
                "asymmetric shape."
            ),
            (
                "Robust tail asymmetry: "
                f"{tail_asymmetry:.3f} "
                f"(moment skewness {skewness:.3f})."
            ),
            (
                "Overall distribution signal: "
                f"{signal:.1f}/100."
            )
        ]

    # -----------------------------------
    # Box plot
    # -----------------------------------

    if chart == "box":

        statistics = candidate.get(
            "statistics",
            {}
        )

        location_score = _safe_float(
            statistics,
            "location_score"
        )

        dispersion_score = _safe_float(
            statistics,
            "dispersion_score"
        )

        outlier_score = _safe_float(
            statistics,
            "outlier_score"
        )

        dominant = max(
            [
                (
                    "location",
                    location_score
                ),
                (
                    "dispersion",
                    dispersion_score
                ),
                (
                    "outliers",
                    outlier_score
                )
            ],
            key=lambda item:
                item[1]
        )[0]

        if dominant == "dispersion":

            dispersion_ratio = _safe_float(
                statistics,
                "dispersion_ratio"
            )

            return [
                (
                    "Dominant group difference: "
                    "within-group spread."
                ),
                (
                    "Largest-to-smallest group IQR ratio: "
                    f"{dispersion_ratio:.2f}×."
                ),
                (
                    "Overall grouped-distribution signal: "
                    f"{signal:.1f}/100."
                )
            ]

        if dominant == "outliers":

            disparity = _safe_float(
                statistics,
                "outlier_rate_disparity"
            )

            maximum_rate = _safe_float(
                statistics,
                "maximum_group_outlier_rate"
            )

            return [
                (
                    "Dominant group difference: "
                    "outlier pattern."
                ),
                (
                    "Group outlier-rate disparity: "
                    f"{disparity * 100:.1f} percentage points; "
                    f"highest group {maximum_rate * 100:.1f}%."
                ),
                (
                    "Overall grouped-distribution signal: "
                    f"{signal:.1f}/100."
                )
            ]

        eta_squared = _safe_float(
            statistics,
            "eta_squared_score"
        )

        median_separation = _safe_float(
            statistics,
            "median_separation_score"
        )

        return [
            (
                "Dominant group difference: "
                "location/central tendency."
            ),
            (
                "Group separation: "
                f"{eta_squared:.1f}/100; "
                f"median separation {median_separation:.1f}/100."
            ),
            (
                "Overall grouped-distribution signal: "
                f"{signal:.1f}/100."
            )
        ]

    # -----------------------------------
    # Bar chart
    # -----------------------------------

    if chart == "bar":

        if "y" in candidate:

            if len(
                reasons
            ) >= 3:

                return [
                    reasons[-3],
                    reasons[-2],
                    reasons[-1]
                ]

        else:

            # The tuned count-bar scorer ends with:
            # frequency-imbalance effect,
            # statistical reliability,
            # final category-frequency signal.
            if len(
                reasons
            ) >= 3:

                return [
                    reasons[-3],
                    reasons[-2],
                    reasons[-1]
                ]

    return reasons[-3:]

