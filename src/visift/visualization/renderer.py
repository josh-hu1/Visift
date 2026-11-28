import math
from collections.abc import Mapping
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from visift.value_normalization import coerce_numeric_or_boolean

# ==================================================
# THEME
# ==================================================

NEON_GREEN = "#39FF14"
NEON_GREEN_SOFT = "#7CFF5B"
TEXT_LIGHT = "#E8FFE8"
TEXT_MUTED = "#9CB39C"
BG_BLACK = "#000000"
BG_PANEL = "#050505"
GRID_COLOR = "#1A1A1A"

# Distinct, bright-on-black colors for grouped charts (one per group).
GROUP_COLORS = (
    "#39FF14",  # neon green
    "#FF4FD8",  # magenta
    "#33C7FF",  # sky blue
    "#FFB000",  # amber
    "#B388FF",  # lavender
    "#FF5C5C",  # coral
    "#00E5C0",  # teal
    "#F4F4F4",  # near white
)


# ==================================================
# HISTOGRAM DISPLAY SETTINGS
# ==================================================

# Fraction trimmed from a tail when an extreme
# outlier would otherwise destroy the chart scale.
HISTOGRAM_TRIM_QUANTILE = 0.01

# A tail is considered extreme only when the
# distance from the percentile boundary to the
# extreme value is much larger than the central
# 98% span.
HISTOGRAM_EXTREME_TAIL_MULTIPLIER = 4.0

# Do not automatically trim very small datasets.
HISTOGRAM_MIN_TRIM_SAMPLE = 100


def apply_plot_theme(fig: go.Figure) -> go.Figure:
    """
    Apply the shared black + neon-green
    Plotly appearance.
    """

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=BG_BLACK,
        plot_bgcolor=BG_BLACK,
        font=dict(color=TEXT_LIGHT),
        title_font=dict(color=NEON_GREEN, size=18),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_LIGHT)),
        xaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            linecolor=GRID_COLOR,
            tickfont=dict(color=TEXT_MUTED),
            title_font=dict(color=TEXT_LIGHT),
        ),
        yaxis=dict(
            gridcolor=GRID_COLOR,
            zerolinecolor=GRID_COLOR,
            linecolor=GRID_COLOR,
            tickfont=dict(color=TEXT_MUTED),
            title_font=dict(color=TEXT_LIGHT),
        ),
    )

    return fig


# ==================================================
# DISPLAY HELPERS
# ==================================================

CURRENCY_TERMS = {
    "revenue",
    "profit",
    "price",
    "cost",
    "sales",
    "spend",
    "income",
    "salary",
    "amount",
    "budget",
    "expense",
}


# A trailing unit token is shown in parentheses: body_mass_g -> "Body Mass (g)".
# Conservative on purpose: single letters are ambiguous ("x", "h"), except g,
# and "min" is left out because temp_min / temp_max usually mean minimum.
UNIT_SUFFIXES = {
    "mm": "mm",
    "cm": "cm",
    "km": "km",
    "g": "g",
    "kg": "kg",
    "mg": "mg",
    "lb": "lb",
    "lbs": "lbs",
    "oz": "oz",
    "ms": "ms",
    "sec": "sec",
    "hr": "hr",
    "hrs": "hrs",
    "pct": "%",
    "usd": "USD",
    "kwh": "kWh",
}


def humanize_column_name(name: object) -> str:
    """
    Convert a raw column name into a cleaner human-readable label. A trailing
    unit (from UNIT_SUFFIXES) becomes a parenthesized suffix:
    bill_depth_mm -> "Bill Depth (mm)", growth_pct -> "Growth (%)".
    """

    words = str(name).replace("_", " ").split()
    unit = UNIT_SUFFIXES.get(words[-1].lower()) if len(words) >= 2 else None

    if unit is None:
        return str(name).replace("_", " ").strip().title()

    return f"{' '.join(words[:-1]).title()} ({unit})"


def is_currency_column(column_name: object) -> bool:
    """
    Infer whether a numeric column likely
    represents currency based on its name.
    """

    name = str(column_name).lower()

    return any(term in name for term in CURRENCY_TERMS)


def visualization_title(candidate: Mapping[str, Any]) -> str:
    """
    Generate a user-facing visualization title.
    """

    chart = candidate["chart"]

    x = humanize_column_name(candidate["x"])

    y = humanize_column_name(candidate["y"]) if "y" in candidate else None

    if chart == "scatter":
        return f"{y} vs {x}"

    if chart == "line":
        return f"{y} Over {x}"

    if chart == "histogram":
        return f"Distribution of {x}"

    if chart == "box":
        return f"{y} by {x}"

    if chart == "bar":
        if "y" in candidate:
            return f"Average {y} by {x}"

        return f"{x} Distribution"

    return "Visualization"


# ==================================================
# NUMBER FORMATTING
# ==================================================


def numeric_tick_format(series: pd.Series) -> str:
    """
    Choose an axis tick format based on the
    magnitude of the numeric values.
    """

    clean = pd.to_numeric(series, errors="coerce").dropna()

    if clean.empty:
        return ",.2f"

    maximum = clean.abs().max()

    if maximum >= 10000:
        return ".3s"

    if maximum >= 1000:
        return ",.0f"

    if maximum >= 100:
        return ",.0f"

    if maximum >= 10:
        return ",.1f"

    return ",.2f"


def hover_number_format(series: pd.Series) -> str:
    """
    Choose a more precise number format for
    hover tooltips.
    """

    clean = pd.to_numeric(series, errors="coerce").dropna()

    if clean.empty:
        return ",.2f"

    integer_like = (clean - clean.round()).abs().max() < 1e-9

    if integer_like:
        return ",.0f"

    return ",.2f"


def hover_numeric_value(column_name: str, series: pd.Series, variable: str) -> str:
    """
    Construct a Plotly hover-template value
    for a numeric variable.
    """

    number_format = hover_number_format(series)

    prefix = "$" if is_currency_column(column_name) else ""

    return f"{prefix}%{{{variable}:{number_format}}}"


def apply_numeric_axis_format(
    fig: go.Figure, axis: str, column_name: str, series: pd.Series
) -> None:
    """
    Apply clean number formatting to one
    Plotly axis.
    """

    tick_format = numeric_tick_format(series)

    prefix = "$" if is_currency_column(column_name) else ""

    settings = {"tickformat": tick_format, "tickprefix": prefix}

    if axis == "x":
        fig.update_xaxes(**settings)

    elif axis == "y":
        fig.update_yaxes(**settings)


# ==================================================
# SCATTERPLOT
# ==================================================


def calculate_linear_trend(data: pd.DataFrame, x: str, y: str) -> dict[str, Any] | None:
    """
    Calculate a simple OLS linear trendline
    and R-squared without requiring statsmodels.
    """

    if len(data) < 3:
        return None

    if data[x].nunique() <= 1 or data[y].nunique() <= 1:
        return None

    x_values = data[x]
    y_values = data[y]

    x_mean = x_values.mean()
    y_mean = y_values.mean()

    denominator = ((x_values - x_mean) ** 2).sum()

    if denominator == 0:
        return None

    slope = ((x_values - x_mean) * (y_values - y_mean)).sum() / denominator

    intercept = y_mean - slope * x_mean

    predictions = slope * x_values + intercept

    total_variation = ((y_values - y_mean) ** 2).sum()

    residual_variation = ((y_values - predictions) ** 2).sum()

    r_squared: float

    if total_variation == 0:
        r_squared = 0

    else:
        r_squared = 1 - residual_variation / total_variation

    line_x = [x_values.min(), x_values.max()]

    line_y = [slope * line_x[0] + intercept, slope * line_x[1] + intercept]

    return {"x": line_x, "y": line_y, "r_squared": r_squared}


def prepare_grouped_scatter(
    df: pd.DataFrame, candidate: Mapping[str, Any]
) -> go.Figure:
    """
    Scatter colored by a grouping column, with an OLS trend line per group
    and a dashed overall trend line: the Simpson's paradox view.
    """

    x = candidate["x"]
    y = candidate["y"]
    group = candidate["group"]

    clean = df[[x, y, group]].copy()
    clean[x] = pd.to_numeric(clean[x], errors="coerce")
    clean[y] = pd.to_numeric(clean[y], errors="coerce")
    clean = clean.dropna()
    clean[group] = clean[group].astype(str)

    fig = go.Figure()
    groups = sorted(clean[group].unique())

    for index, level in enumerate(groups):
        part = clean[clean[group] == level]
        color = GROUP_COLORS[index % len(GROUP_COLORS)]

        fig.add_trace(
            go.Scatter(
                x=part[x],
                y=part[y],
                mode="markers",
                name=f"{level}",
                legendgroup=level,
                marker=dict(color=color, size=7, opacity=0.6, line=dict(width=0)),
                hovertemplate=(
                    f"<b>{humanize_column_name(group)}</b>: {level}<br>"
                    f"<b>{humanize_column_name(x)}</b>: %{{x}}<br>"
                    f"<b>{humanize_column_name(y)}</b>: %{{y}}<extra></extra>"
                ),
            )
        )

        trend = calculate_linear_trend(part, x, y)

        if trend is not None:
            fig.add_trace(
                go.Scatter(
                    x=trend["x"],
                    y=trend["y"],
                    mode="lines",
                    name=f"{level} trend",
                    legendgroup=level,
                    showlegend=False,
                    line=dict(color=color, width=3),
                    hoverinfo="skip",
                )
            )

    overall = calculate_linear_trend(clean, x, y)

    if overall is not None:
        fig.add_trace(
            go.Scatter(
                x=overall["x"],
                y=overall["y"],
                mode="lines",
                name="All rows (overall trend)",
                line=dict(color=TEXT_LIGHT, width=2, dash="dash"),
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        title=f"{humanize_column_name(x)} vs {humanize_column_name(y)} "
        f"by {humanize_column_name(group)}",
        xaxis_title=humanize_column_name(x),
        yaxis_title=humanize_column_name(y),
        legend_title_text=humanize_column_name(group),
    )

    apply_numeric_axis_format(fig, "x", x, clean[x])
    apply_numeric_axis_format(fig, "y", y, clean[y])

    return fig


def prepare_scatter(df: pd.DataFrame, candidate: Mapping[str, Any]) -> go.Figure:
    """
    Build a scatterplot with a linear trendline.

    Very large datasets are sampled for rendering
    performance, but the trendline is calculated
    using the complete dataset.
    """

    x = candidate["x"]
    y = candidate["y"]

    clean = df[[x, y]].copy()

    clean[x] = pd.to_numeric(clean[x], errors="coerce")

    clean[y] = pd.to_numeric(clean[y], errors="coerce")

    clean = clean.dropna()

    # -----------------------------------
    # Limit rendered points for very
    # large datasets.
    # -----------------------------------

    max_display_points = 5000

    if len(clean) > max_display_points:
        display_data = clean.sample(n=max_display_points, random_state=42)

        sampled = True

    else:
        display_data = clean

        sampled = False

    opacity = 0.45 if len(display_data) > 1000 else 0.72

    fig = px.scatter(
        display_data,
        x=x,
        y=y,
        title=visualization_title(candidate),
        labels={x: humanize_column_name(x), y: humanize_column_name(y)},
        render_mode="auto",
    )

    # -----------------------------------
    # Neon scatter points
    # -----------------------------------

    fig.update_traces(
        marker=dict(color=NEON_GREEN, size=7, opacity=opacity, line=dict(width=0)),
        selector=dict(mode="markers"),
    )

    x_hover = hover_numeric_value(x, clean[x], "x")

    y_hover = hover_numeric_value(y, clean[y], "y")

    fig.update_traces(
        hovertemplate=(
            f"<b>{humanize_column_name(x)}</b>: "
            f"{x_hover}"
            "<br>"
            f"<b>{humanize_column_name(y)}</b>: "
            f"{y_hover}"
            "<extra></extra>"
        ),
        selector=dict(mode="markers"),
    )

    # -----------------------------------
    # Linear trendline
    # -----------------------------------

    trend = calculate_linear_trend(clean, x, y)

    if trend is not None:
        fig.add_trace(
            go.Scatter(
                x=trend["x"],
                y=trend["y"],
                mode="lines",
                name=(f"Linear trend (R² = {trend['r_squared']:.2f})"),
                line=dict(color=NEON_GREEN_SOFT, width=2, dash="dash"),
                hoverinfo="skip",
            )
        )

    # -----------------------------------
    # Axis formatting
    # -----------------------------------

    apply_numeric_axis_format(fig, "x", x, clean[x])

    apply_numeric_axis_format(fig, "y", y, clean[y])

    if sampled:
        fig.add_annotation(
            x=1,
            y=1.08,
            xref="paper",
            yref="paper",
            xanchor="right",
            showarrow=False,
            font=dict(color=TEXT_MUTED),
            text=(f"Showing {max_display_points:,} of {len(clean):,} observations"),
        )

    return fig


# ==================================================
# LINE CHART
# ==================================================


def prepare_datetime_line_data(
    clean: pd.DataFrame, x: str, y: str
) -> tuple[pd.DataFrame, str | None]:
    """
    Sort and optionally aggregate datetime data
    to make dense line charts easier to read.

    Returns:
        data
        aggregation label
    """

    clean = clean.sort_values(x)

    unique_points = clean[x].nunique()

    date_span = (clean[x].max() - clean[x].min()).days

    # -----------------------------------
    # Sparse time series:
    # preserve original observations.
    # -----------------------------------

    if unique_points <= 100:
        # groupby(as_index=False)[y].mean() is a DataFrame; the stubs say Series.
        result = clean.groupby(x, as_index=False)[y].mean().sort_values(x)  # type: ignore[call-overload]

        return (result, None)

    # -----------------------------------
    # Long / dense time series:
    # monthly average.
    # -----------------------------------

    if date_span > 365 or unique_points > 180:
        result = clean.set_index(x)[y].resample("MS").mean().dropna().reset_index()

        return (result, "Monthly average")

    # -----------------------------------
    # Medium-density time series:
    # weekly average.
    # -----------------------------------

    result = clean.set_index(x)[y].resample("W").mean().dropna().reset_index()

    return (result, "Weekly average")


def prepare_line(df: pd.DataFrame, candidate: Mapping[str, Any]) -> go.Figure:
    """
    Build a temporal line chart.

    Dense datetime series are automatically
    aggregated for readability.
    """

    x = candidate["x"]
    y = candidate["y"]

    clean = df[[x, y]].copy()

    clean[y] = pd.to_numeric(clean[y], errors="coerce")

    # -----------------------------------
    # Numeric temporal values such as year.
    # -----------------------------------

    if pd.api.types.is_numeric_dtype(clean[x]) and not pd.api.types.is_bool_dtype(
        clean[x]
    ):
        clean[x] = pd.to_numeric(clean[x], errors="coerce")

        clean = clean.dropna(subset=[x, y])

        plot_data = clean.groupby(x, as_index=False)[y].mean().sort_values(x)

        aggregation_label = None
        datetime_axis = False

    # -----------------------------------
    # Datetime temporal values.
    # -----------------------------------

    else:
        clean[x] = pd.to_datetime(clean[x], errors="coerce")

        clean = clean.dropna(subset=[x, y])

        (plot_data, aggregation_label) = prepare_datetime_line_data(clean, x, y)

        datetime_axis = True

    title = visualization_title(candidate)

    if aggregation_label:
        title = f"{title}<br><sup>{aggregation_label}</sup>"

    fig = px.line(
        plot_data,
        x=x,
        y=y,
        title=title,
        labels={x: humanize_column_name(x), y: humanize_column_name(y)},
        markers=(len(plot_data) <= 60),
    )

    # -----------------------------------
    # Neon line
    # -----------------------------------

    fig.update_traces(
        line=dict(color=NEON_GREEN, width=3), marker=dict(color=NEON_GREEN, size=6)
    )

    y_hover = hover_numeric_value(y, plot_data[y], "y")

    if datetime_axis:
        if aggregation_label == "Monthly average":
            x_hover = "%{x|%b %Y}"

        else:
            x_hover = "%{x|%b %d, %Y}"

    else:
        x_hover = "%{x}"

    fig.update_traces(
        hovertemplate=(
            f"<b>{humanize_column_name(x)}</b>: "
            f"{x_hover}"
            "<br>"
            f"<b>{humanize_column_name(y)}</b>: "
            f"{y_hover}"
            "<extra></extra>"
        )
    )

    if datetime_axis:
        fig.update_xaxes(type="date")

    apply_numeric_axis_format(fig, "y", y, plot_data[y])

    fig.update_layout(hovermode="x unified")

    return fig


# ==================================================
# HISTOGRAM
# ==================================================


def histogram_bin_count(series: pd.Series) -> int:
    """
    Estimate a useful histogram bin count using
    the Freedman-Diaconis rule.

    Falls back to square-root binning when needed.
    """

    clean = pd.to_numeric(series, errors="coerce").dropna()

    n = len(clean)

    if n < 2:
        return 10

    q1 = clean.quantile(0.25)

    q3 = clean.quantile(0.75)

    iqr = q3 - q1

    data_range = clean.max() - clean.min()

    if iqr > 0 and data_range > 0:
        bin_width = 2 * iqr * (n ** (-1 / 3))

        if bin_width > 0:
            bins = math.ceil(data_range / bin_width)

        else:
            bins = int(math.sqrt(n))

    else:
        bins = int(math.sqrt(n))

    return max(10, min(bins, 60))


def prepare_histogram_display_data(
    series: pd.Series,
) -> tuple[pd.Series, dict[str, Any]]:
    """
    Prepare histogram data for visualization.

    Extremely distant tails can make an otherwise
    informative histogram collapse into one visible
    bar. When that happens, trim only the displayed
    range while preserving the full dataset for
    scoring.

    Trimming is intentionally conservative. A tail
    is only hidden when its distance from the central
    distribution is much larger than the span of the
    central 98% of observations.
    """

    clean = pd.to_numeric(series, errors="coerce").dropna()

    info: dict[str, Any] = {
        "trimmed": False,
        "trim_lower": False,
        "trim_upper": False,
        "lower_bound": None,
        "upper_bound": None,
        "excluded_below": 0,
        "excluded_above": 0,
        "excluded_total": 0,
    }

    if len(clean) < HISTOGRAM_MIN_TRIM_SAMPLE or clean.nunique() < 3:
        return (clean, info)

    lower_quantile = HISTOGRAM_TRIM_QUANTILE

    upper_quantile = 1 - HISTOGRAM_TRIM_QUANTILE

    lower_bound = clean.quantile(lower_quantile)

    upper_bound = clean.quantile(upper_quantile)

    central_span = upper_bound - lower_bound

    if not math.isfinite(float(central_span)) or central_span <= 0:
        return (clean, info)

    minimum = clean.min()
    maximum = clean.max()

    lower_tail_distance = lower_bound - minimum

    upper_tail_distance = maximum - upper_bound

    trim_lower = lower_tail_distance > (
        HISTOGRAM_EXTREME_TAIL_MULTIPLIER * central_span
    )

    trim_upper = upper_tail_distance > (
        HISTOGRAM_EXTREME_TAIL_MULTIPLIER * central_span
    )

    if not (trim_lower or trim_upper):
        return (clean, info)

    mask = pd.Series(True, index=clean.index)

    if trim_lower:
        mask &= clean >= lower_bound

    if trim_upper:
        mask &= clean <= upper_bound

    display_data = clean[mask]

    # -----------------------------------
    # Safety fallback:
    # never trim into an unusable plot.
    # -----------------------------------

    if len(display_data) < 10 or display_data.nunique() < 2:
        return (clean, info)

    excluded_below = int((clean < lower_bound).sum()) if trim_lower else 0

    excluded_above = int((clean > upper_bound).sum()) if trim_upper else 0

    info = {
        "trimmed": True,
        "trim_lower": trim_lower,
        "trim_upper": trim_upper,
        "lower_bound": (float(lower_bound) if trim_lower else None),
        "upper_bound": (float(upper_bound) if trim_upper else None),
        "excluded_below": excluded_below,
        "excluded_above": excluded_above,
        "excluded_total": (excluded_below + excluded_above),
    }

    return (display_data, info)


def format_histogram_bound(column_name: str, value: float) -> str:
    """
    Format a histogram display boundary for
    a user-facing annotation.
    """

    prefix = "$" if is_currency_column(column_name) else ""

    if abs(value) >= 100:
        formatted = f"{value:,.0f}"

    elif abs(value) >= 10:
        formatted = f"{value:,.1f}"

    else:
        formatted = f"{value:,.2f}"

    return f"{prefix}{formatted}"


def histogram_display_note(column_name: str, info: dict[str, Any]) -> str | None:
    """
    Generate a transparent explanation when
    extreme observations are omitted only from
    histogram rendering.
    """

    if not info["trimmed"]:
        return None

    excluded = info["excluded_total"]

    observation_word = "observation" if excluded == 1 else "observations"

    # -----------------------------------
    # Upper tail only
    # -----------------------------------

    if info["trim_upper"] and not info["trim_lower"]:
        upper = format_histogram_bound(column_name, info["upper_bound"])

        return (
            "Display limited to the "
            f"99th percentile (≤ {upper}); "
            f"{excluded:,} extreme "
            f"{observation_word} hidden. "
            "Scoring uses the full dataset."
        )

    # -----------------------------------
    # Lower tail only
    # -----------------------------------

    if info["trim_lower"] and not info["trim_upper"]:
        lower = format_histogram_bound(column_name, info["lower_bound"])

        return (
            "Display limited from the "
            f"1st percentile (≥ {lower}); "
            f"{excluded:,} extreme "
            f"{observation_word} hidden. "
            "Scoring uses the full dataset."
        )

    # -----------------------------------
    # Both tails
    # -----------------------------------

    lower = format_histogram_bound(column_name, info["lower_bound"])

    upper = format_histogram_bound(column_name, info["upper_bound"])

    return (
        "Display limited to the central "
        f"98% ({lower} to {upper}); "
        f"{excluded:,} extreme "
        f"{observation_word} hidden. "
        "Scoring uses the full dataset."
    )


def prepare_histogram(df: pd.DataFrame, candidate: Mapping[str, Any]) -> go.Figure:
    """
    Build a histogram with automatic
    distribution-aware binning.

    Extremely distant tails are omitted only
    from the displayed histogram when they would
    otherwise destroy the useful visual scale.

    The recommendation scorer still receives
    the complete, unmodified dataset.
    """

    x = candidate["x"]

    full_data = pd.to_numeric(df[x], errors="coerce").dropna()

    (display_data, display_info) = prepare_histogram_display_data(full_data)

    nbins = histogram_bin_count(display_data)

    plot_data = pd.DataFrame({x: display_data})

    fig = px.histogram(
        plot_data,
        x=x,
        nbins=nbins,
        title=visualization_title(candidate),
        labels={x: humanize_column_name(x), "count": "Count"},
    )

    # -----------------------------------
    # Neon histogram bars
    # -----------------------------------

    fig.update_traces(
        marker=dict(color=NEON_GREEN, line=dict(color=BG_BLACK, width=1)), opacity=0.82
    )

    x_hover = hover_numeric_value(x, display_data, "x")

    fig.update_traces(
        hovertemplate=(
            f"<b>{humanize_column_name(x)}</b>: "
            f"{x_hover}"
            "<br>"
            "<b>Count</b>: %{y:,}"
            "<extra></extra>"
        )
    )

    apply_numeric_axis_format(fig, "x", x, display_data)

    fig.update_yaxes(tickformat=",d", title="Count")

    # -----------------------------------
    # Explain display-only trimming
    # -----------------------------------

    note = histogram_display_note(x, display_info)

    if note:
        fig.add_annotation(
            x=0,
            y=1.08,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
            align="left",
            text=note,
            font=dict(color=TEXT_MUTED, size=11),
        )

    # Store whether additional top spacing
    # is needed by render_candidate().
    fig.update_layout(meta={"histogram_trimmed": display_info["trimmed"]})

    return fig


# ==================================================
# BOX PLOT
# ==================================================


def box_category_order(clean: pd.DataFrame, x: str, y: str) -> list[Any]:
    """
    Determine a sensible category order.

    Numeric/ordinal groups preserve their natural
    order. Text categories are ordered by median.
    """

    if pd.api.types.is_bool_dtype(clean[x]):
        return sorted(clean[x].dropna().unique())

    if pd.api.types.is_numeric_dtype(clean[x]):
        return sorted(clean[x].dropna().unique())

    medians = clean.groupby(x)[y].median().sort_values(ascending=False)

    return list(medians.index)


def prepare_box(df: pd.DataFrame, candidate: Mapping[str, Any]) -> go.Figure:
    """
    Build a grouped box plot with sensible
    category ordering.
    """

    x = candidate["x"]
    y = candidate["y"]

    clean = df[[x, y]].copy()

    clean[y] = pd.to_numeric(clean[y], errors="coerce")

    clean = clean.dropna(subset=[x, y])

    category_order = box_category_order(clean, x, y)

    fig = px.box(
        clean,
        x=x,
        y=y,
        points="outliers",
        title=visualization_title(candidate),
        labels={x: humanize_column_name(x), y: humanize_column_name(y)},
        category_orders={x: category_order},
    )

    # -----------------------------------
    # Neon box plot
    # -----------------------------------

    fig.update_traces(
        marker=dict(color=NEON_GREEN),
        line=dict(color=NEON_GREEN, width=2),
        fillcolor=("rgba(57,255,20,0.28)"),
    )

    apply_numeric_axis_format(fig, "y", y, clean[y])

    return fig


# ==================================================
# BAR CHART
# ==================================================


def prepare_bar(df: pd.DataFrame, candidate: Mapping[str, Any]) -> go.Figure:
    """
    Build either:

    - category count bar chart
    - category + numeric mean bar chart
    """

    x = candidate["x"]

    aggregation = candidate.get("aggregation", "count")

    # -----------------------------------
    # Count bar chart
    # -----------------------------------

    if aggregation == "count":
        counts = df[x].dropna().value_counts().rename_axis(x).reset_index(name="count")

        # Numeric/ordinal categories should
        # preserve natural ordering.
        if pd.api.types.is_numeric_dtype(counts[x]):
            counts = counts.sort_values(x)

        else:
            counts = counts.sort_values("count", ascending=False)

        fig = px.bar(
            counts,
            x=x,
            y="count",
            title=visualization_title(candidate),
            labels={x: humanize_column_name(x), "count": "Count"},
        )

        # -----------------------------------
        # Neon bars
        # -----------------------------------

        fig.update_traces(
            marker=dict(color=NEON_GREEN, line=dict(color=NEON_GREEN_SOFT, width=1)),
            opacity=0.85,
        )

        fig.update_traces(
            hovertemplate=(
                f"<b>{humanize_column_name(x)}</b>: "
                "%{x}"
                "<br>"
                "<b>Count</b>: %{y:,}"
                "<extra></extra>"
            )
        )

        fig.update_yaxes(tickformat=",d")

        return fig

    # -----------------------------------
    # Category + numeric mean bar chart
    # -----------------------------------

    y = candidate["y"]

    clean = df[[x, y]].copy()

    clean[y] = coerce_numeric_or_boolean(clean[y])

    clean = clean.dropna(subset=[x, y])

    grouped = clean.groupby(x, as_index=False)[y].mean()

    # Numeric categories such as ordinal
    # ratings should remain naturally ordered.
    if pd.api.types.is_numeric_dtype(grouped[x]):
        grouped = grouped.sort_values(x)

    else:
        grouped = grouped.sort_values(y, ascending=False)

    fig = px.bar(
        grouped,
        x=x,
        y=y,
        title=visualization_title(candidate),
        labels={x: humanize_column_name(x), y: (f"Average {humanize_column_name(y)}")},
    )

    # -----------------------------------
    # Neon bars
    # -----------------------------------

    fig.update_traces(
        marker=dict(color=NEON_GREEN, line=dict(color=NEON_GREEN_SOFT, width=1)),
        opacity=0.85,
    )

    y_hover = hover_numeric_value(y, grouped[y], "y")

    fig.update_traces(
        hovertemplate=(
            f"<b>{humanize_column_name(x)}</b>: "
            "%{x}"
            "<br>"
            f"<b>Average {humanize_column_name(y)}</b>: "
            f"{y_hover}"
            "<extra></extra>"
        )
    )

    apply_numeric_axis_format(fig, "y", y, grouped[y])

    return fig


# ==================================================
# COMMON RENDERER
# ==================================================


def render_candidate(df: pd.DataFrame, candidate: Mapping[str, Any]) -> go.Figure:
    """
    Convert a recommendation into a
    polished Plotly visualization.
    """

    chart = candidate["chart"]

    if chart == "scatter" and candidate.get("group"):
        fig = prepare_grouped_scatter(df, candidate)

    elif chart == "scatter":
        fig = prepare_scatter(df, candidate)

    elif chart == "line":
        fig = prepare_line(df, candidate)

    elif chart == "histogram":
        fig = prepare_histogram(df, candidate)

    elif chart == "box":
        fig = prepare_box(df, candidate)

    elif chart == "bar":
        fig = prepare_bar(df, candidate)

    else:
        raise ValueError(f"Unsupported chart type: {chart}")

    # -----------------------------------
    # Histograms with a trimming note
    # need slightly more room above the
    # chart.
    # -----------------------------------

    histogram_trimmed = False

    if isinstance(fig.layout.meta, dict):
        histogram_trimmed = fig.layout.meta.get("histogram_trimmed", False)

    top_margin = 105 if histogram_trimmed else 75

    # -----------------------------------
    # Shared dimensions / spacing
    # -----------------------------------

    fig.update_layout(
        margin={"l": 25, "r": 25, "t": top_margin, "b": 25},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )

    # -----------------------------------
    # Apply dark neon theme
    # -----------------------------------

    fig = apply_plot_theme(fig)

    # Keep grids subtle rather than
    # overwhelming the neon data.

    fig.update_xaxes(showgrid=False)

    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)

    return fig


# ==================================================
# CORRELATION OVERVIEW
# ==================================================

# Diverging pair for rho: red for negative, blue for positive, a neutral gray
# at 0 (the dark-surface steps of the validated reference palette).
RHO_NEGATIVE = "#eb6263"
RHO_NEUTRAL = "#383838"
RHO_POSITIVE = "#338ced"

# Write the value in each cell only while the cells are large enough.
OVERVIEW_MAX_ANNOTATED_COLUMNS = 12


def render_correlation_overview(overview: Mapping[str, Any]) -> go.Figure:
    """
    Heatmap of visift.overview.correlation_overview(): Spearman rho between
    numeric columns, clustered, diagonal blank. Hover gives rho and the
    number of rows behind it.
    """

    columns = list(overview["columns"])
    rho = overview["rho"]
    rows = overview["rows"]
    labels = [humanize_column_name(c) for c in columns]
    z = rho.to_numpy(dtype=float)

    hover = [
        [
            (
                f"{labels[i]} vs {labels[j]}<br>"
                + (
                    f"ρ = {z[i, j]:+.2f} (n = {int(rows.iat[i, j]):,})"
                    if math.isfinite(z[i, j])
                    else ("same column" if i == j else "too few shared rows")
                )
            )
            for j in range(len(columns))
        ]
        for i in range(len(columns))
    ]

    annotate = len(columns) <= OVERVIEW_MAX_ANNOTATED_COLUMNS
    text = [
        [f"{value:.2f}" if annotate and math.isfinite(value) else "" for value in row]
        for row in z
    ]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale=[[0.0, RHO_NEGATIVE], [0.5, RHO_NEUTRAL], [1.0, RHO_POSITIVE]],
            colorbar=dict(
                title=dict(text="Spearman ρ", font=dict(color=TEXT_LIGHT)),
                tickvals=[-1, -0.5, 0, 0.5, 1],
                tickfont=dict(color=TEXT_MUTED),
                outlinewidth=0,
            ),
            text=text,
            texttemplate="%{text}",
            textfont=dict(color=TEXT_LIGHT, size=11),
            customdata=hover,
            hovertemplate="%{customdata}<extra></extra>",
            xgap=2,
            ygap=2,
            hoverongaps=False,
        )
    )

    size = len(columns)
    apply_plot_theme(fig)
    fig.update_layout(
        title="Correlation overview (Spearman ρ)",
        height=int(min(820, max(380, 140 + 30 * size))),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_xaxes(showgrid=False, tickangle=-45, side="bottom")
    fig.update_yaxes(showgrid=False, autorange="reversed")

    return fig
