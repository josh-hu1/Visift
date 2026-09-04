import math

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


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
    "expense"
}


def humanize_column_name(name):
    """
    Convert a raw column name into a cleaner
    human-readable label.
    """

    return (
        str(name)
        .replace("_", " ")
        .strip()
        .title()
    )


def is_currency_column(column_name):
    """
    Infer whether a numeric column likely
    represents currency based on its name.
    """

    name = str(
        column_name
    ).lower()

    return any(
        term in name
        for term in CURRENCY_TERMS
    )


def visualization_title(candidate):
    """
    Generate a user-facing visualization title.
    """

    chart = candidate["chart"]

    x = humanize_column_name(
        candidate["x"]
    )

    y = (
        humanize_column_name(
            candidate["y"]
        )
        if "y" in candidate
        else None
    )

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

def numeric_tick_format(series):
    """
    Choose an axis tick format based on the
    magnitude of the numeric values.
    """

    clean = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

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


def hover_number_format(series):
    """
    Choose a more precise number format for
    hover tooltips.
    """

    clean = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if clean.empty:
        return ",.2f"

    integer_like = (
        (
            clean
            - clean.round()
        )
        .abs()
        .max()
        < 1e-9
    )

    if integer_like:
        return ",.0f"

    return ",.2f"


def hover_numeric_value(
    column_name,
    series,
    variable
):
    """
    Construct a Plotly hover template value
    for a numeric variable.
    """

    number_format = hover_number_format(
        series
    )

    prefix = (
        "$"
        if is_currency_column(
            column_name
        )
        else ""
    )

    return (
        f"{prefix}"
        f"%{{{variable}:{number_format}}}"
    )


def apply_numeric_axis_format(
    fig,
    axis,
    column_name,
    series
):
    """
    Apply clean number formatting to one
    Plotly axis.
    """

    tick_format = numeric_tick_format(
        series
    )

    prefix = (
        "$"
        if is_currency_column(
            column_name
        )
        else ""
    )

    settings = {
        "tickformat": tick_format,
        "tickprefix": prefix
    }

    if axis == "x":

        fig.update_xaxes(
            **settings
        )

    elif axis == "y":

        fig.update_yaxes(
            **settings
        )


# ==================================================
# SCATTERPLOT
# ==================================================

def calculate_linear_trend(
    data,
    x,
    y
):
    """
    Calculate a simple OLS linear trendline
    and R-squared without requiring statsmodels.
    """

    if len(data) < 3:
        return None

    if (
        data[x].nunique() <= 1
        or data[y].nunique() <= 1
    ):
        return None

    x_values = data[x]
    y_values = data[y]

    x_mean = x_values.mean()
    y_mean = y_values.mean()

    denominator = (
        (
            x_values - x_mean
        ) ** 2
    ).sum()

    if denominator == 0:
        return None

    slope = (
        (
            (x_values - x_mean)
            * (y_values - y_mean)
        ).sum()
        / denominator
    )

    intercept = (
        y_mean
        - slope * x_mean
    )

    predictions = (
        slope * x_values
        + intercept
    )

    total_variation = (
        (
            y_values - y_mean
        ) ** 2
    ).sum()

    residual_variation = (
        (
            y_values - predictions
        ) ** 2
    ).sum()

    if total_variation == 0:

        r_squared = 0

    else:

        r_squared = (
            1
            - residual_variation
            / total_variation
        )

    line_x = [
        x_values.min(),
        x_values.max()
    ]

    line_y = [
        slope * line_x[0]
        + intercept,
        slope * line_x[1]
        + intercept
    ]

    return {
        "x": line_x,
        "y": line_y,
        "r_squared": r_squared
    }


def prepare_scatter(
    df,
    candidate
):
    """
    Build a scatterplot with a linear trendline.

    Very large datasets are sampled for rendering
    performance, but the trendline is calculated
    using the complete dataset.
    """

    x = candidate["x"]
    y = candidate["y"]

    clean = (
        df[[x, y]]
        .copy()
    )

    clean[x] = pd.to_numeric(
        clean[x],
        errors="coerce"
    )

    clean[y] = pd.to_numeric(
        clean[y],
        errors="coerce"
    )

    clean = clean.dropna()

    # -----------------------------------
    # Limit rendered points for very
    # large datasets.
    # -----------------------------------

    max_display_points = 5000

    if len(clean) > max_display_points:

        display_data = clean.sample(
            n=max_display_points,
            random_state=42
        )

        sampled = True

    else:

        display_data = clean

        sampled = False

    opacity = (
        0.45
        if len(display_data) > 1000
        else 0.70
    )

    fig = px.scatter(
        display_data,
        x=x,
        y=y,
        title=visualization_title(
            candidate
        ),
        labels={
            x: humanize_column_name(x),
            y: humanize_column_name(y)
        },
        opacity=opacity,
        render_mode="auto"
    )

    x_hover = hover_numeric_value(
        x,
        clean[x],
        "x"
    )

    y_hover = hover_numeric_value(
        y,
        clean[y],
        "y"
    )

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

    # -----------------------------------
    # Linear trendline
    # -----------------------------------

    trend = calculate_linear_trend(
        clean,
        x,
        y
    )

    if trend is not None:

        fig.add_trace(
            go.Scatter(
                x=trend["x"],
                y=trend["y"],
                mode="lines",
                name=(
                    "Linear trend "
                    f"(R² = "
                    f"{trend['r_squared']:.2f})"
                ),
                line={
                    "dash": "dash"
                },
                hoverinfo="skip"
            )
        )

    # -----------------------------------
    # Axis formatting
    # -----------------------------------

    apply_numeric_axis_format(
        fig,
        "x",
        x,
        clean[x]
    )

    apply_numeric_axis_format(
        fig,
        "y",
        y,
        clean[y]
    )

    if sampled:

        fig.add_annotation(
            x=1,
            y=1.08,
            xref="paper",
            yref="paper",
            xanchor="right",
            showarrow=False,
            text=(
                f"Showing {max_display_points:,} "
                f"of {len(clean):,} observations"
            )
        )

    return fig


# ==================================================
# LINE CHART
# ==================================================

def prepare_datetime_line_data(
    clean,
    x,
    y
):
    """
    Sort and optionally aggregate datetime data
    to make dense line charts easier to read.

    Returns:
        data
        aggregation label
    """

    clean = clean.sort_values(
        x
    )

    unique_points = (
        clean[x].nunique()
    )

    date_span = (
        clean[x].max()
        - clean[x].min()
    ).days

    # -----------------------------------
    # Sparse time series:
    # preserve original observations.
    # -----------------------------------

    if unique_points <= 100:

        result = (
            clean
            .groupby(
                x,
                as_index=False
            )[y]
            .mean()
            .sort_values(x)
        )

        return (
            result,
            None
        )

    # -----------------------------------
    # Long / dense time series:
    # monthly average.
    # -----------------------------------

    if (
        date_span > 365
        or unique_points > 180
    ):

        result = (
            clean
            .set_index(x)[y]
            .resample("MS")
            .mean()
            .dropna()
            .reset_index()
        )

        return (
            result,
            "Monthly average"
        )

    # -----------------------------------
    # Medium-density time series:
    # weekly average.
    # -----------------------------------

    result = (
        clean
        .set_index(x)[y]
        .resample("W")
        .mean()
        .dropna()
        .reset_index()
    )

    return (
        result,
        "Weekly average"
    )


def prepare_line(
    df,
    candidate
):
    """
    Build a temporal line chart.

    Dense datetime series are automatically
    aggregated for readability.
    """

    x = candidate["x"]
    y = candidate["y"]

    clean = (
        df[[x, y]]
        .copy()
    )

    clean[y] = pd.to_numeric(
        clean[y],
        errors="coerce"
    )

    # -----------------------------------
    # Numeric temporal values such as year.
    # -----------------------------------

    if (
        pd.api.types.is_numeric_dtype(
            clean[x]
        )
        and not pd.api.types.is_bool_dtype(
            clean[x]
        )
    ):

        clean[x] = pd.to_numeric(
            clean[x],
            errors="coerce"
        )

        clean = clean.dropna(
            subset=[x, y]
        )

        plot_data = (
            clean
            .groupby(
                x,
                as_index=False
            )[y]
            .mean()
            .sort_values(x)
        )

        aggregation_label = None
        datetime_axis = False

    # -----------------------------------
    # Datetime temporal values.
    # -----------------------------------

    else:

        clean[x] = pd.to_datetime(
            clean[x],
            errors="coerce"
        )

        clean = clean.dropna(
            subset=[x, y]
        )

        (
            plot_data,
            aggregation_label
        ) = prepare_datetime_line_data(
            clean,
            x,
            y
        )

        datetime_axis = True

    title = visualization_title(
        candidate
    )

    if aggregation_label:

        title = (
            f"{title}"
            "<br>"
            f"<sup>{aggregation_label}</sup>"
        )

    fig = px.line(
        plot_data,
        x=x,
        y=y,
        title=title,
        labels={
            x: humanize_column_name(x),
            y: humanize_column_name(y)
        },
        markers=(
            len(plot_data) <= 60
        )
    )

    y_hover = hover_numeric_value(
        y,
        plot_data[y],
        "y"
    )

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

        fig.update_xaxes(
            type="date"
        )

    apply_numeric_axis_format(
        fig,
        "y",
        y,
        plot_data[y]
    )

    fig.update_layout(
        hovermode="x unified"
    )

    return fig


# ==================================================
# HISTOGRAM
# ==================================================

def histogram_bin_count(series):
    """
    Estimate a useful histogram bin count using
    the Freedman-Diaconis rule.

    Falls back to square-root binning when needed.
    """

    clean = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    n = len(clean)

    if n < 2:
        return 10

    q1 = clean.quantile(
        0.25
    )

    q3 = clean.quantile(
        0.75
    )

    iqr = q3 - q1

    data_range = (
        clean.max()
        - clean.min()
    )

    if (
        iqr > 0
        and data_range > 0
    ):

        bin_width = (
            2
            * iqr
            * (
                n ** (-1 / 3)
            )
        )

        if bin_width > 0:

            bins = math.ceil(
                data_range
                / bin_width
            )

        else:

            bins = int(
                math.sqrt(n)
            )

    else:

        bins = int(
            math.sqrt(n)
        )

    return max(
        10,
        min(
            bins,
            60
        )
    )


def prepare_histogram(
    df,
    candidate
):
    """
    Build a histogram with automatic
    distribution-aware binning.
    """

    x = candidate["x"]

    clean = pd.to_numeric(
        df[x],
        errors="coerce"
    ).dropna()

    nbins = histogram_bin_count(
        clean
    )

    plot_data = pd.DataFrame({
        x: clean
    })

    fig = px.histogram(
        plot_data,
        x=x,
        nbins=nbins,
        title=visualization_title(
            candidate
        ),
        labels={
            x: humanize_column_name(x),
            "count": "Count"
        }
    )

    x_hover = hover_numeric_value(
        x,
        clean,
        "x"
    )

    fig.update_traces(
        hovertemplate=(
            f"<b>{humanize_column_name(x)}</b>: "
            f"{x_hover}"
            "<br>"
            "<b>Count</b>: %{y:,}"
            "<extra></extra>"
        )
    )

    apply_numeric_axis_format(
        fig,
        "x",
        x,
        clean
    )

    fig.update_yaxes(
        tickformat=",d",
        title="Count"
    )

    return fig


# ==================================================
# BOX PLOT
# ==================================================

def box_category_order(
    clean,
    x,
    y
):
    """
    Determine a sensible category order.

    Numeric/ordinal groups preserve their natural
    order. Text categories are ordered by median.
    """

    if pd.api.types.is_bool_dtype(
        clean[x]
    ):

        return sorted(
            clean[x].dropna().unique()
        )

    if pd.api.types.is_numeric_dtype(
        clean[x]
    ):

        return sorted(
            clean[x].dropna().unique()
        )

    medians = (
        clean
        .groupby(x)[y]
        .median()
        .sort_values(
            ascending=False
        )
    )

    return list(
        medians.index
    )


def prepare_box(
    df,
    candidate
):
    """
    Build a grouped box plot with sensible
    category ordering.
    """

    x = candidate["x"]
    y = candidate["y"]

    clean = (
        df[[x, y]]
        .copy()
    )

    clean[y] = pd.to_numeric(
        clean[y],
        errors="coerce"
    )

    clean = clean.dropna(
        subset=[x, y]
    )

    category_order = box_category_order(
        clean,
        x,
        y
    )

    fig = px.box(
        clean,
        x=x,
        y=y,
        points="outliers",
        title=visualization_title(
            candidate
        ),
        labels={
            x: humanize_column_name(x),
            y: humanize_column_name(y)
        },
        category_orders={
            x: category_order
        }
    )

    apply_numeric_axis_format(
        fig,
        "y",
        y,
        clean[y]
    )

    return fig


# ==================================================
# BAR CHART
# ==================================================

def prepare_bar(
    df,
    candidate
):
    """
    Build either:

    - category count bar chart
    - category + numeric mean bar chart
    """

    x = candidate["x"]

    aggregation = candidate.get(
        "aggregation",
        "count"
    )

    # -----------------------------------
    # Count bar chart
    # -----------------------------------

    if aggregation == "count":

        counts = (
            df[x]
            .dropna()
            .value_counts()
            .rename_axis(x)
            .reset_index(
                name="count"
            )
        )

        # Numeric/ordinal categories should
        # preserve natural ordering.
        if pd.api.types.is_numeric_dtype(
            counts[x]
        ):

            counts = counts.sort_values(
                x
            )

        else:

            counts = counts.sort_values(
                "count",
                ascending=False
            )

        fig = px.bar(
            counts,
            x=x,
            y="count",
            title=visualization_title(
                candidate
            ),
            labels={
                x: humanize_column_name(x),
                "count": "Count"
            }
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

        fig.update_yaxes(
            tickformat=",d"
        )

        return fig

    # -----------------------------------
    # Category + numeric mean bar chart
    # -----------------------------------

    y = candidate["y"]

    clean = (
        df[[x, y]]
        .copy()
    )

    clean[y] = pd.to_numeric(
        clean[y],
        errors="coerce"
    )

    clean = clean.dropna(
        subset=[x, y]
    )

    grouped = (
        clean
        .groupby(
            x,
            as_index=False
        )[y]
        .mean()
    )

    # Numeric categories such as ordinal
    # ratings should remain naturally ordered.
    if pd.api.types.is_numeric_dtype(
        grouped[x]
    ):

        grouped = grouped.sort_values(
            x
        )

    else:

        grouped = grouped.sort_values(
            y,
            ascending=False
        )

    fig = px.bar(
        grouped,
        x=x,
        y=y,
        title=visualization_title(
            candidate
        ),
        labels={
            x: humanize_column_name(x),
            y: (
                f"Average "
                f"{humanize_column_name(y)}"
            )
        }
    )

    y_hover = hover_numeric_value(
        y,
        grouped[y],
        "y"
    )

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

    apply_numeric_axis_format(
        fig,
        "y",
        y,
        grouped[y]
    )

    return fig


# ==================================================
# COMMON RENDERER
# ==================================================

def render_candidate(
    df,
    candidate
):
    """
    Convert a recommendation into a
    polished Plotly visualization.
    """

    chart = candidate["chart"]

    if chart == "scatter":

        fig = prepare_scatter(
            df,
            candidate
        )

    elif chart == "line":

        fig = prepare_line(
            df,
            candidate
        )

    elif chart == "histogram":

        fig = prepare_histogram(
            df,
            candidate
        )

    elif chart == "box":

        fig = prepare_box(
            df,
            candidate
        )

    elif chart == "bar":

        fig = prepare_bar(
            df,
            candidate
        )

    else:

        raise ValueError(
            f"Unsupported chart type: "
            f"{chart}"
        )

    # -----------------------------------
    # Shared appearance
    # -----------------------------------

    fig.update_layout(
        margin={
            "l": 25,
            "r": 25,
            "t": 75,
            "b": 25
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1
        }
    )

    fig.update_xaxes(
        showgrid=False
    )

    fig.update_yaxes(
        zeroline=False
    )

    return fig