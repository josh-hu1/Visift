import math

import pandas as pd
import plotly.express as px


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


def visualization_title(candidate):
    """
    Generate a user-facing visualization title.
    """

    chart = candidate["chart"]

    x = humanize_column_name(
        candidate["x"]
    )

    y = (
        humanize_column_name(candidate["y"])
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


def prepare_scatter(
    df,
    candidate
):
    """
    Build a scatterplot.
    """

    x = candidate["x"]
    y = candidate["y"]

    clean = (
        df[[x, y]]
        .dropna()
    )

    fig = px.scatter(
        clean,
        x=x,
        y=y,
        title=visualization_title(
            candidate
        ),
        labels={
            x: humanize_column_name(x),
            y: humanize_column_name(y)
        },
        opacity=0.65
    )

    return fig


def prepare_line(
    df,
    candidate
):
    """
    Build a temporal line chart.

    Duplicate temporal values are aggregated
    using the mean of the numeric variable.
    """

    x = candidate["x"]
    y = candidate["y"]

    clean = (
        df[[x, y]]
        .copy()
    )

    parsed_dates = pd.to_datetime(
        clean[x],
        errors="coerce"
    )

    if parsed_dates.notna().mean() >= 0.80:

        clean[x] = parsed_dates

    clean[y] = pd.to_numeric(
        clean[y],
        errors="coerce"
    )

    clean = clean.dropna(
        subset=[x, y]
    )

    clean = (
        clean
        .groupby(
            x,
            as_index=False
        )[y]
        .mean()
        .sort_values(x)
    )

    fig = px.line(
        clean,
        x=x,
        y=y,
        title=visualization_title(
            candidate
        ),
        labels={
            x: humanize_column_name(x),
            y: humanize_column_name(y)
        }
    )

    return fig


def prepare_histogram(
    df,
    candidate
):
    """
    Build a histogram.
    """

    x = candidate["x"]

    clean = pd.to_numeric(
        df[x],
        errors="coerce"
    ).dropna()

    if len(clean) == 0:

        nbins = 10

    else:

        nbins = int(
            math.sqrt(len(clean))
        )

        nbins = max(
            10,
            min(nbins, 50)
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
            x: humanize_column_name(x)
        }
    )

    return fig


def prepare_box(
    df,
    candidate
):
    """
    Build a grouped box plot.
    """

    x = candidate["x"]
    y = candidate["y"]

    clean = (
        df[[x, y]]
        .dropna()
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
        }
    )

    return fig


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

        return fig

    # -----------------------------------
    # Mean bar chart
    # -----------------------------------

    y = candidate["y"]

    clean = (
        df[[x, y]]
        .dropna()
    )

    grouped = (
        clean
        .groupby(
            x,
            as_index=False
        )[y]
        .mean()
    )

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

    return fig


def render_candidate(
    df,
    candidate
):
    """
    Convert a recommendation into a
    Plotly visualization.
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

    fig.update_layout(
        margin=dict(
            l=30,
            r=30,
            t=65,
            b=30
        )
    )

    return fig