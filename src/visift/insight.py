"""
Plain-English takeaways for recommendations ("Average Bill Length (mm) is
highest for Chinstrap and lowest for Adelie.").

Used by the app, the HTML report, the CLI and the notebook API, so every
surface says the same thing about a chart.
"""

from collections.abc import Mapping
from typing import Any

import pandas as pd

from visift.value_normalization import coerce_numeric_or_boolean
from visift.visualization.renderer import humanize_column_name, is_currency_column

__all__ = ["plain_english_insight"]


def relationship_strength(value: float) -> str:
    """
    Convert an absolute relationship value
    into plain-English wording.
    """

    value = abs(value)

    if value >= 0.70:
        return "strong"

    if value >= 0.40:
        return "moderate"

    if value >= 0.20:
        return "weak"

    return "very weak"


def signal_strength(value: float) -> str:
    """
    Convert a calibrated 0-100 pattern signal into
    concise plain-English wording.
    """

    if value >= 80:
        return "strong"

    if value >= 60:
        return "moderate"

    if value >= 40:
        return "noticeable"

    if value >= 20:
        return "weak"

    return "very weak"


# ==================================================
# PLAIN-ENGLISH INSIGHT GENERATION
# ==================================================


def scatter_insight(candidate: Mapping[str, Any]) -> str:
    """
    Generate a plain-English explanation for
    a scatterplot recommendation.

    The takeaway follows the signal that actually
    drives the scorer: linear/monotonic association
    or nonlinear dependence.
    """

    x = humanize_column_name(candidate["x"])

    y = humanize_column_name(candidate["y"])

    statistics = candidate.get("statistics", {})

    linear_signal = statistics.get("linear_signal", 0)

    nonlinear_signal = statistics.get("nonlinear_signal", 0)

    # -----------------------------------
    # Nonlinear relationship
    # -----------------------------------

    if nonlinear_signal > linear_signal:
        strength = signal_strength(nonlinear_signal)

        pearson = statistics.get("pearson", 0)

        spearman = statistics.get("spearman", 0)

        strongest_linear = max(abs(pearson), abs(spearman))

        if strongest_linear < 0.20:
            return (
                f"{x} and {y} show a {strength} nonlinear "
                f"relationship that is not well summarized "
                f"by a straight-line trend."
            )

        return (
            f"{x} and {y} show a {strength} nonlinear "
            f"relationship, which is more informative than "
            f"their linear correlation alone."
        )

    # -----------------------------------
    # Linear / monotonic relationship
    # -----------------------------------

    pearson = statistics.get("pearson", 0)

    spearman = statistics.get("spearman", 0)

    selected_method = statistics.get("selected_correlation_method")

    if selected_method == "pearson":
        strongest = pearson
    elif selected_method == "spearman":
        strongest = spearman
    else:
        strongest = pearson if abs(pearson) >= abs(spearman) else spearman

    strength = relationship_strength(strongest)

    if strongest > 0:
        return (
            f"Higher {x} tends to be associated with "
            f"higher {y}, with a {strength} positive "
            f"relationship."
        )

    if strongest < 0:
        return (
            f"Higher {x} tends to be associated with "
            f"lower {y}, with a {strength} negative "
            f"relationship."
        )

    return f"{x} and {y} show little evidence of a consistent relationship."


def line_insight(candidate: Mapping[str, Any]) -> str:
    """
    Generate a plain-English explanation for
    a line-chart recommendation.

    Supports:
        - trend
        - seasonality
        - level shift
        - volatility shift
    """

    x = humanize_column_name(candidate["x"])

    y = humanize_column_name(candidate["y"])

    statistics = candidate["statistics"]

    signal = candidate["components"]["signal"]

    dominant_pattern = statistics.get("dominant_pattern", "trend")

    # -----------------------------------
    # General strength wording
    # -----------------------------------

    if signal >= 80:
        strength = "strong"

    elif signal >= 60:
        strength = "moderate"

    elif signal >= 40:
        strength = "noticeable"

    else:
        strength = "weak"

    # ==================================================
    # SEASONALITY
    # ==================================================

    if dominant_pattern == "seasonality":
        period_observations = statistics.get("seasonal_period_observations", 0)

        autocorrelation = statistics.get("seasonal_autocorrelation", 0)

        if period_observations and period_observations > 0:
            period_display = f"{period_observations:.0f}"

            return (
                f"{y} shows a {strength} repeating pattern "
                f"over {x}, recurring approximately every "
                f"{period_display} observations "
                f"(autocorrelation {autocorrelation:.2f})."
            )

        return f"{y} shows a {strength} repeating seasonal pattern over {x}."

    # ==================================================
    # VOLATILITY SHIFT
    # ==================================================

    if dominant_pattern == "volatility_shift":
        dispersion_ratio = statistics.get("volatility_dispersion_ratio", 1)

        split_fraction = statistics.get("volatility_split_fraction", 0)

        if split_fraction and split_fraction > 0:
            split_percent = split_fraction * 100

            return (
                f"{y} shows a {strength} change in variability "
                f"over {x}. Dispersion differs by about "
                f"{dispersion_ratio:.1f}× across temporal regimes, "
                f"with the strongest shift near "
                f"{split_percent:.0f}% of the way through the series."
            )

        return (
            f"{y} shows a {strength} change in variability "
            f"over {x}, with dispersion differing by about "
            f"{dispersion_ratio:.1f}× across temporal regimes."
        )

    # ==================================================
    # LEVEL SHIFT
    # ==================================================

    if dominant_pattern == "level_shift":
        effect_size = statistics.get("level_shift_effect_size", 0)

        split_fraction = statistics.get("level_shift_split_fraction", 0)

        if split_fraction and split_fraction > 0:
            split_percent = split_fraction * 100

            return (
                f"{y} shows a {strength} persistent change in level "
                f"over {x}, with the strongest shift occurring near "
                f"{split_percent:.0f}% of the way through the series "
                f"(standardized effect {effect_size:.2f})."
            )

        return f"{y} shows a {strength} persistent level shift over {x}."

    # ==================================================
    # TREND
    # ==================================================

    direction = statistics.get("direction", "flat")

    pearson = statistics.get("pearson", 0)

    spearman = statistics.get("spearman", 0)

    strongest_correlation = max(abs(pearson), abs(spearman))

    if direction == "upward":
        return (
            f"{y} shows a {strength} upward trend over {x}, "
            f"with a temporal correlation of approximately "
            f"{strongest_correlation:.2f}."
        )

    if direction == "downward":
        return (
            f"{y} shows a {strength} downward trend over {x}, "
            f"with a temporal correlation of approximately "
            f"{strongest_correlation:.2f}."
        )

    return f"{y} shows little evidence of a consistent directional trend over {x}."


def format_group_label(value: object) -> str:
    """
    Format category labels cleanly.

    Float values that represent whole numbers
    are shown without a decimal.
    """

    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))

    return str(value)


def box_insight(df: pd.DataFrame, candidate: Mapping[str, Any]) -> str:
    """
    Generate a plain-English explanation for
    a grouped box plot.

    The takeaway follows the dominant grouped-
    distribution signal when available:
        - location
        - dispersion
        - differential outliers
    """

    x = candidate["x"]
    y = candidate["y"]

    x_name = humanize_column_name(x)

    y_name = humanize_column_name(y)

    statistics = candidate.get("statistics", {})

    location_score = statistics.get("location_score", 0)

    dispersion_score = statistics.get("dispersion_score", 0)

    outlier_score = statistics.get("outlier_score", 0)

    dominant = max(
        [
            ("location", location_score),
            ("dispersion", dispersion_score),
            ("outliers", outlier_score),
        ],
        key=lambda item: item[1],
    )[0]

    # -----------------------------------
    # Dispersion differences
    # -----------------------------------

    if dominant == "dispersion" and dispersion_score > 0:
        dispersion_ratio = statistics.get("dispersion_ratio", 1)

        return (
            f"{y_name} varies much more within some "
            f"{x_name} groups than others, with the "
            f"largest group IQR about "
            f"{dispersion_ratio:.1f}× the smallest."
        )

    # -----------------------------------
    # Differential outlier behavior
    # -----------------------------------

    if dominant == "outliers" and outlier_score > 0:
        disparity = statistics.get("outlier_rate_disparity", 0)

        maximum_rate = statistics.get("maximum_group_outlier_rate", 0)

        return (
            f"Potential outliers are unevenly distributed "
            f"across {x_name} groups: the highest group "
            f"outlier rate is about "
            f"{maximum_rate * 100:.1f}%, with a "
            f"{disparity * 100:.1f}-point disparity "
            f"between groups."
        )

    # -----------------------------------
    # Location / median differences
    # -----------------------------------

    clean = df[[x, y]].dropna().copy()

    if clean.empty:
        return (
            "The available data is insufficient to describe the grouped distributions."
        )

    medians = clean.groupby(x)[y].median().sort_values()

    if len(medians) <= 1:
        return f"There is only one usable {x_name} group."

    lowest_group = format_group_label(medians.index[0])

    highest_group = format_group_label(medians.index[-1])

    lowest_value = medians.iloc[0]

    highest_value = medians.iloc[-1]

    if is_currency_column(y):
        # Escape dollar signs because Streamlit
        # Markdown otherwise interprets them as math.
        lowest_display = f"\\${lowest_value:,.0f}"

        highest_display = f"\\${highest_value:,.0f}"

    else:
        lowest_display = f"{lowest_value:,.1f}"

        highest_display = f"{highest_value:,.1f}"

    return (
        f"Group medians for {y_name} vary across {x_name}, "
        f"ranging from about {lowest_display} for "
        f"{lowest_group} to {highest_display} for "
        f"{highest_group}."
    )


def histogram_insight(candidate: Mapping[str, Any]) -> str:
    """
    Generate a plain-English explanation for
    a histogram.

    The takeaway follows the strongest calibrated
    distribution signal:
        - asymmetric shape
        - unusual tail observations
        - multimodality
    """

    x = humanize_column_name(candidate["x"])

    statistics = candidate.get("statistics", {})

    signal = candidate.get("components", {}).get("signal", 0)

    if signal < 15:
        return (
            f"The distribution of {x} appears fairly ordinary, "
            f"with no strong distributional pattern detected."
        )

    shape_score = statistics.get("shape_score", 0)

    outlier_score = statistics.get("outlier_score", 0)

    multimodality_score = statistics.get("multimodality_score", 0)

    # -----------------------------------
    # Multimodality
    # -----------------------------------

    if (
        multimodality_score >= max(shape_score, outlier_score)
        and multimodality_score > 0
    ):
        strength = signal_strength(multimodality_score)

        separation = statistics.get("multimodality_component_separation", 0)

        smaller_weight = statistics.get("multimodality_smaller_component_weight", 0)

        if separation > 0 and smaller_weight > 0:
            return (
                f"The distribution of {x} shows {strength} "
                f"evidence of multiple modes, with the "
                f"fitted components separated by about "
                f"{separation:.1f} standard deviations."
            )

        return (
            f"The distribution of {x} shows {strength} "
            f"evidence of multiple modes or clusters."
        )

    # -----------------------------------
    # Outlier / tail behavior
    # -----------------------------------

    if outlier_score > shape_score and outlier_score > 0:
        outlier_rate = statistics.get("outlier_rate", 0) * 100

        return (
            f"The distribution of {x} contains unusual "
            f"tail observations, with about "
            f"{outlier_rate:.1f}% flagged as potential "
            f"outliers by the current rule."
        )

    # -----------------------------------
    # Shape / asymmetry
    # -----------------------------------

    skewness = statistics.get("skewness", 0)

    tail_asymmetry = statistics.get("tail_asymmetry", 0)

    outlier_rate = statistics.get("outlier_rate", 0) * 100

    # Prefer robust tail asymmetry for direction when
    # it is meaningfully different from zero.
    direction_value = tail_asymmetry if abs(tail_asymmetry) >= 0.10 else skewness

    if direction_value >= 0.75:
        shape = "noticeably right-skewed"

    elif direction_value <= -0.75:
        shape = "noticeably left-skewed"

    elif abs(direction_value) >= 0.30:
        shape = "slightly asymmetric"

    else:
        shape = "fairly symmetric"

    if outlier_rate >= 5:
        outlier_text = (
            f"and contains a notable share of potential outliers ({outlier_rate:.1f}%)."
        )

    elif outlier_rate > 0:
        outlier_text = f"with relatively few potential outliers ({outlier_rate:.1f}%)."

    else:
        outlier_text = "with no potential outliers detected by the current rule."

    return f"The distribution of {x} is {shape} {outlier_text}"


def bar_insight(df: pd.DataFrame, candidate: Mapping[str, Any]) -> str:
    """
    Generate a plain-English explanation for
    a bar-chart recommendation.
    """

    x = candidate["x"]

    x_name = humanize_column_name(x)

    aggregation = candidate.get("aggregation", "count")

    # -----------------------------------
    # Count bar
    # -----------------------------------

    if aggregation == "count":
        counts = df[x].dropna().value_counts()

        if counts.empty:
            return f"No usable {x_name} values were available."

        largest = counts.index[0]

        smallest = counts.index[-1]

        if counts.max() == counts.min():
            return f"{x_name} categories appear evenly represented."

        signal = candidate.get("components", {}).get("signal", 0)

        strength = signal_strength(signal)

        return (
            f"{x_name} categories show a {strength} "
            f"frequency imbalance: {largest} is the most "
            f"common, while {smallest} is the least common."
        )

    # -----------------------------------
    # Mean bar
    # -----------------------------------

    y = candidate["y"]

    y_name = humanize_column_name(y)

    clean = df[[x, y]].copy()

    clean[y] = coerce_numeric_or_boolean(clean[y])

    clean = clean.dropna(subset=[x, y])

    means = clean.groupby(x)[y].mean().sort_values()

    if means.empty:
        return "No usable values were available for this comparison."

    lowest_group = means.index[0]
    highest_group = means.index[-1]

    return (
        f"Average {y_name} is highest for "
        f"{highest_group} and lowest for "
        f"{lowest_group}."
    )


def plain_english_insight(df: pd.DataFrame, candidate: Mapping[str, Any]) -> str:
    """
    Generate the most useful plain-English
    takeaway for a recommendation.
    """

    chart = candidate["chart"]

    if chart == "scatter":
        return scatter_insight(candidate)

    if chart == "line":
        return line_insight(candidate)

    if chart == "box":
        return box_insight(df, candidate)

    if chart == "histogram":
        return histogram_insight(candidate)

    if chart == "bar":
        return bar_insight(df, candidate)

    return (
        "This visualization was selected because "
        "it combines strong chart suitability with "
        "a potentially useful pattern."
    )
