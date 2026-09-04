import pandas as pd
import streamlit as st

from profiler import profile_dataset
from candidate_generator import generate_candidates

from scoring.engine import (
    score_all_candidates,
    diversify_recommendations,
    chart_display_name,
    recommendation_strength,
    summary_reasons
)

from visualization.renderer import (
    render_candidate,
    visualization_title,
    humanize_column_name,
    is_currency_column
)


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Dataviz Engine",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# DISPLAY HELPERS
# ==================================================

def strength_color(strength):
    """
    Map recommendation strength to a
    Streamlit badge color.
    """

    colors = {
        "Excellent": "green",
        "Strong": "blue",
        "Moderate": "orange",
        "Weak": "gray",
        "Very weak": "red"
    }

    return colors.get(
        strength,
        "gray"
    )


def format_semantic_type(value):
    """
    Convert semantic-type identifiers into
    clean display text.
    """

    return (
        str(value)
        .replace("_", " ")
        .title()
    )


def relationship_strength(value):
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


# ==================================================
# PLAIN-ENGLISH INSIGHT GENERATION
# ==================================================

def scatter_insight(candidate):
    """
    Generate a plain-English explanation for
    a scatterplot recommendation.
    """

    x = humanize_column_name(
        candidate["x"]
    )

    y = humanize_column_name(
        candidate["y"]
    )

    statistics = candidate[
        "statistics"
    ]

    pearson = statistics[
        "pearson"
    ]

    spearman = statistics[
        "spearman"
    ]

    strongest = (
        pearson
        if abs(pearson) >= abs(spearman)
        else spearman
    )

    strength = relationship_strength(
        strongest
    )

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

    return (
        f"{x} and {y} show little evidence of "
        f"a consistent relationship."
    )


def line_insight(candidate):
    """
    Generate a plain-English explanation for
    a line-chart recommendation.

    Supports:
        - trend
        - seasonality
        - level shift
        - volatility shift
    """

    x = humanize_column_name(
        candidate["x"]
    )

    y = humanize_column_name(
        candidate["y"]
    )

    statistics = candidate[
        "statistics"
    ]

    signal = candidate[
        "components"
    ]["signal"]

    dominant_pattern = (
        statistics.get(
            "dominant_pattern",
            "trend"
        )
    )

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

    if (
        dominant_pattern
        == "seasonality"
    ):

        period_observations = (
            statistics.get(
                "seasonal_period_observations",
                0
            )
        )

        autocorrelation = (
            statistics.get(
                "seasonal_autocorrelation",
                0
            )
        )

        if (
            period_observations
            and period_observations > 0
        ):

            period_display = (
                f"{period_observations:.0f}"
            )

            return (
                f"{y} shows a {strength} repeating pattern "
                f"over {x}, recurring approximately every "
                f"{period_display} observations "
                f"(autocorrelation {autocorrelation:.2f})."
            )

        return (
            f"{y} shows a {strength} repeating seasonal "
            f"pattern over {x}."
        )

    # ==================================================
    # VOLATILITY SHIFT
    # ==================================================

    if (
        dominant_pattern
        == "volatility_shift"
    ):

        dispersion_ratio = (
            statistics.get(
                "volatility_dispersion_ratio",
                1
            )
        )

        split_fraction = (
            statistics.get(
                "volatility_split_fraction",
                0
            )
        )

        if (
            split_fraction
            and split_fraction > 0
        ):

            split_percent = (
                split_fraction
                * 100
            )

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

    if (
        dominant_pattern
        == "level_shift"
    ):

        effect_size = (
            statistics.get(
                "level_shift_effect_size",
                0
            )
        )

        split_fraction = (
            statistics.get(
                "level_shift_split_fraction",
                0
            )
        )

        if (
            split_fraction
            and split_fraction > 0
        ):

            split_percent = (
                split_fraction
                * 100
            )

            return (
                f"{y} shows a {strength} persistent change in level "
                f"over {x}, with the strongest shift occurring near "
                f"{split_percent:.0f}% of the way through the series "
                f"(standardized effect {effect_size:.2f})."
            )

        return (
            f"{y} shows a {strength} persistent level shift "
            f"over {x}."
        )

    # ==================================================
    # TREND
    # ==================================================

    direction = statistics.get(
        "direction",
        "flat"
    )

    pearson = statistics.get(
        "pearson",
        0
    )

    spearman = statistics.get(
        "spearman",
        0
    )

    strongest_correlation = max(
        abs(pearson),
        abs(spearman)
    )

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

    return (
        f"{y} shows little evidence of a consistent "
        f"directional trend over {x}."
    )

def format_group_label(value):
    """
    Format category labels cleanly.

    Float values that represent whole numbers
    are shown without a decimal.
    """

    if isinstance(value, float):

        if value.is_integer():
            return str(
                int(value)
            )

    return str(value)


def box_insight(
    df,
    candidate
):
    """
    Generate a plain-English explanation for
    a grouped box plot.
    """

    x = candidate["x"]
    y = candidate["y"]

    clean = (
        df[[x, y]]
        .dropna()
        .copy()
    )

    if clean.empty:

        return (
            "The available data is insufficient to "
            "describe the grouped distributions."
        )

    medians = (
        clean
        .groupby(x)[y]
        .median()
        .sort_values()
    )

    if len(medians) <= 1:

        return (
            f"There is only one usable "
            f"{humanize_column_name(x)} group."
        )

    lowest_group = (
        format_group_label(
            medians.index[0]
        )
    )

    highest_group = (
        format_group_label(
            medians.index[-1]
        )
    )

    lowest_value = (
        medians.iloc[0]
    )

    highest_value = (
        medians.iloc[-1]
    )

    y_name = humanize_column_name(
        y
    )

    x_name = humanize_column_name(
        x
    )

    if is_currency_column(y):

        # Escape dollar signs because Streamlit
        # Markdown otherwise interprets them as math.
        lowest_display = (
            f"\\${lowest_value:,.0f}"
        )

        highest_display = (
            f"\\${highest_value:,.0f}"
        )

    else:

        lowest_display = (
            f"{lowest_value:,.1f}"
        )

        highest_display = (
            f"{highest_value:,.1f}"
        )

    return (
        f"Median {y_name} varies across {x_name} groups, "
        f"ranging from about {lowest_display} for "
        f"{lowest_group} to {highest_display} for "
        f"{highest_group}."
    )


def histogram_insight(candidate):
    """
    Generate a plain-English explanation for
    a histogram.
    """

    x = humanize_column_name(
        candidate["x"]
    )

    statistics = candidate[
        "statistics"
    ]

    skewness = statistics[
        "skewness"
    ]

    outlier_rate = (
        statistics["outlier_rate"]
        * 100
    )

    if skewness >= 0.75:

        shape = "noticeably right-skewed"

    elif skewness <= -0.75:

        shape = "noticeably left-skewed"

    elif abs(skewness) >= 0.30:

        shape = "slightly asymmetric"

    else:

        shape = "fairly symmetric"

    if outlier_rate >= 5:

        outlier_text = (
            f"and contains a notable share of "
            f"potential outliers ({outlier_rate:.1f}%)."
        )

    elif outlier_rate > 0:

        outlier_text = (
            f"with relatively few potential outliers "
            f"({outlier_rate:.1f}%)."
        )

    else:

        outlier_text = (
            "with no potential outliers detected "
            "by the current rule."
        )

    return (
        f"The distribution of {x} is {shape} "
        f"{outlier_text}"
    )


def bar_insight(
    df,
    candidate
):
    """
    Generate a plain-English explanation for
    a bar-chart recommendation.
    """

    x = candidate["x"]

    x_name = humanize_column_name(
        x
    )

    aggregation = candidate.get(
        "aggregation",
        "count"
    )

    # -----------------------------------
    # Count bar
    # -----------------------------------

    if aggregation == "count":

        counts = (
            df[x]
            .dropna()
            .value_counts()
        )

        if counts.empty:

            return (
                f"No usable {x_name} values "
                f"were available."
            )

        largest = counts.index[0]

        smallest = counts.index[-1]

        if counts.max() == counts.min():

            return (
                f"{x_name} categories appear "
                f"evenly represented."
            )

        return (
            f"{largest} is the most common {x_name} "
            f"category, while {smallest} is the least common."
        )

    # -----------------------------------
    # Mean bar
    # -----------------------------------

    y = candidate["y"]

    y_name = humanize_column_name(
        y
    )

    clean = (
        df[[x, y]]
        .dropna()
    )

    means = (
        clean
        .groupby(x)[y]
        .mean()
        .sort_values()
    )

    if means.empty:

        return (
            f"No usable values were available "
            f"for this comparison."
        )

    lowest_group = means.index[0]
    highest_group = means.index[-1]

    return (
        f"Average {y_name} is highest for "
        f"{highest_group} and lowest for "
        f"{lowest_group}."
    )


def plain_english_insight(
    df,
    candidate
):
    """
    Generate the most useful plain-English
    takeaway for a recommendation.
    """

    chart = candidate[
        "chart"
    ]

    if chart == "scatter":

        return scatter_insight(
            candidate
        )

    if chart == "line":

        return line_insight(
            candidate
        )

    if chart == "box":

        return box_insight(
            df,
            candidate
        )

    if chart == "histogram":

        return histogram_insight(
            candidate
        )

    if chart == "bar":

        return bar_insight(
            df,
            candidate
        )

    return (
        "This visualization was selected because "
        "it combines strong chart suitability with "
        "a potentially useful pattern."
    )


# ==================================================
# DATA LOADING
# ==================================================

def load_data():
    """
    Allow the user to load the included sample
    dataset or upload a CSV.
    """

    with st.sidebar:

        st.header(
            "Dataset"
        )

        data_source = st.radio(
            "Choose a data source",
            [
                "Sample dataset",
                "Upload CSV"
            ]
        )

        if data_source == "Upload CSV":

            uploaded_file = (
                st.file_uploader(
                    "Upload CSV",
                    type=["csv"],
                    help=(
                        "Upload a CSV dataset for "
                        "automatic visualization analysis."
                    )
                )
            )

            if uploaded_file is None:

                st.info(
                    "Upload a CSV file to begin."
                )

                st.stop()

            try:

                df = pd.read_csv(
                    uploaded_file
                )

            except Exception as error:

                st.error(
                    "The CSV could not be read."
                )

                st.exception(
                    error
                )

                st.stop()

            source_name = (
                uploaded_file.name
            )

        else:

            try:

                df = pd.read_csv(
                    "synthetic_sales.csv"
                )

            except FileNotFoundError:

                st.error(
                    "synthetic_sales.csv "
                    "could not be found."
                )

                st.stop()

            source_name = (
                "synthetic_sales.csv"
            )

    return (
        df,
        source_name
    )


# ==================================================
# ENGINE
# ==================================================

@st.cache_data(
    show_spinner=False
)
def analyze_dataset(df):
    """
    Run the full recommendation pipeline.

    Results are cached so changing interface
    filters does not rerun the analysis.
    """

    profile = profile_dataset(
        df
    )

    candidates = generate_candidates(
        df,
        profile
    )

    scored = score_all_candidates(
        df,
        candidates,
        profile
    )

    recommendations = (
        diversify_recommendations(
            scored
        )
    )

    return (
        profile,
        candidates,
        recommendations
    )


# ==================================================
# HEADER
# ==================================================

def display_header():
    """
    Display the main application header.
    """

    st.title(
        "Dataviz Engine"
    )

    st.markdown(
        """
Automatically discover the visualizations most likely
to reveal useful patterns in your dataset.
"""
    )

    st.markdown(
        """
**Instead of showing every chart that can be made,
Dataviz Engine ranks the charts most likely to be informative.**
"""
    )

    st.markdown(
        ":violet-badge[Explainable scoring] "
        ":blue-badge[5 visualization types] "
        ":gray-badge[CSV analysis]"
    )


# ==================================================
# DATASET OVERVIEW
# ==================================================

def display_dataset_metrics(
    df,
    profile,
    candidates,
    recommendations
):
    """
    Display high-level dataset and engine metrics.
    """

    missing_values = int(
        df.isna().sum().sum()
    )

    top_score = (
        recommendations[0]["score"]
        if recommendations
        else 0
    )

    column1, column2, column3, column4 = (
        st.columns(4)
    )

    column1.metric(
        "Rows",
        f"{profile['rows']:,}"
    )

    column2.metric(
        "Columns",
        profile["columns"]
    )

    column3.metric(
        "Candidates analyzed",
        len(candidates)
    )

    column4.metric(
        "Top score",
        f"{top_score:.1f}"
    )

    st.caption(
        f"{missing_values:,} missing values detected "
        f"across the dataset."
    )


def build_column_profile_table(
    profile
):
    """
    Convert column profiles into a cleaner
    dataframe for display.
    """

    rows = []

    for column in (
        profile["column_profiles"]
    ):

        rows.append({
            "Column": (
                column["name"]
            ),
            "Semantic Type": (
                format_semantic_type(
                    column["semantic_type"]
                )
            ),
            "Unique Values": (
                column["unique_count"]
            ),
            "Missing": (
                column["missing_count"]
            ),
            "Missing %": (
                f"{column['missing_percent']:.1f}%"
            ),
            "Pandas Type": (
                column["pandas_dtype"]
            )
        })

    return pd.DataFrame(
        rows
    )


def build_display_preview(
    df,
    profile,
    rows=100
):
    """
    Build a display-only dataframe with
    friendlier dates, currency formatting,
    and numeric formatting.

    The original dataframe remains unchanged.
    """

    preview = (
        df.head(rows)
        .copy()
    )

    semantic_types = {
        column["name"]: (
            column["semantic_type"]
        )
        for column in profile[
            "column_profiles"
        ]
    }

    for column in preview.columns:

        semantic_type = (
            semantic_types.get(
                column
            )
        )

        original_series = (
            df[column]
        )

        # -----------------------------------
        # Datetime display
        # -----------------------------------

        if semantic_type == "datetime":

            parsed = pd.to_datetime(
                preview[column],
                errors="coerce"
            )

            preview[column] = (
                parsed
                .dt.strftime(
                    "%b %d, %Y"
                )
            )

        # -----------------------------------
        # Currency display
        # -----------------------------------

        elif (
            semantic_type
            in {
                "numeric_continuous",
                "numeric_discrete"
            }
            and is_currency_column(
                column
            )
        ):

            numeric = pd.to_numeric(
                preview[column],
                errors="coerce"
            )

            preview[column] = (
                numeric.map(
                    lambda value: (
                        f"${value:,.2f}"
                        if pd.notna(value)
                        else None
                    )
                )
            )

        # -----------------------------------
        # Integer numeric display
        # -----------------------------------

        elif pd.api.types.is_integer_dtype(
            original_series
        ):

            numeric = pd.to_numeric(
                preview[column],
                errors="coerce"
            )

            preview[column] = (
                numeric.map(
                    lambda value: (
                        f"{int(value):,}"
                        if pd.notna(value)
                        else None
                    )
                )
            )

        # -----------------------------------
        # Continuous numeric display
        # -----------------------------------

        elif (
            semantic_type
            == "numeric_continuous"
        ):

            numeric = pd.to_numeric(
                preview[column],
                errors="coerce"
            )

            preview[column] = (
                numeric.map(
                    lambda value: (
                        f"{value:,.2f}"
                        if pd.notna(value)
                        else None
                    )
                )
            )

    return preview


# ==================================================
# SCORE DISPLAY
# ==================================================

def display_score_breakdown(
    candidate
):
    """
    Display recommendation score components.
    """

    display_names = {
        "semantic_fit": "Semantic Fit",
        "readability": "Readability",
        "sample_support": "Sample Support",
        "data_quality": "Data Quality",
        "signal": "Pattern Strength",
        "visualization_quality": "Chart Suitability"
    }

    component_rows = []

    for name, value in (
        candidate["components"].items()
    ):

        component_rows.append({
            "Component": (
                display_names.get(
                    name,
                    name
                    .replace("_", " ")
                    .title()
                )
            ),
            "Score": value
        })

    st.dataframe(
        pd.DataFrame(
            component_rows
        ),
        width="stretch",
        hide_index=True
    )


# ==================================================
# RECOMMENDATION CARD
# ==================================================

def display_recommendation(
    df,
    candidate,
    rank
):
    """
    Render one polished recommendation card.
    """

    title = visualization_title(
        candidate
    )

    chart_name = chart_display_name(
        candidate["chart"]
    )

    score = candidate[
        "score"
    ]

    strength = recommendation_strength(
        score
    )

    color = strength_color(
        strength
    )

    alternatives = candidate.get(
        "alternative_charts",
        []
    )

    with st.container(
        border=True
    ):

        # -----------------------------------
        # Header
        # -----------------------------------

        header_left, header_right = (
            st.columns(
                [6, 1]
            )
        )

        with header_left:

            st.markdown(
                f"### #{rank} — {title}"
            )

            st.markdown(
                f":{color}-badge[{strength}] "
                f":gray-badge[{chart_name}]"
            )

        with header_right:

            st.metric(
                "Score",
                f"{score:.1f}"
            )

        # -----------------------------------
        # Plain-English takeaway
        # -----------------------------------

        insight = plain_english_insight(
            df,
            candidate
        )

        st.info(
            insight,
            icon="💡"
        )

        # -----------------------------------
        # Top metrics
        # -----------------------------------

        if alternatives:

            (
                metric1,
                metric2,
                metric3
            ) = st.columns(3)

        else:

            (
                metric1,
                metric2
            ) = st.columns(2)

        metric1.metric(
            "Pattern strength",
            (
                f"{candidate['components']['signal']:.1f}"
                "/100"
            )
        )

        metric2.metric(
            "Chart suitability",
            (
                f"{candidate['components']['visualization_quality']:.1f}"
                "/100"
            )
        )

        if alternatives:

            metric3.metric(
                "Alternative views",
                len(alternatives)
            )

        # -----------------------------------
        # Visualization
        # -----------------------------------

        fig = render_candidate(
            df,
            candidate
        )

        st.plotly_chart(
            fig,
            width="stretch",
            config={
                "displaylogo": False,
                "responsive": True
            }
        )

        # -----------------------------------
        # Details
        # -----------------------------------

        with st.expander(
            "Recommendation details"
        ):

            evidence_tab, scoring_tab = (
                st.tabs(
                    [
                        "Evidence",
                        "Score breakdown"
                    ]
                )
            )

            with evidence_tab:

                st.markdown(
                    "#### Key evidence"
                )

                for reason in summary_reasons(
                    candidate
                ):

                    st.write(
                        f"• {reason}"
                    )

                if alternatives:

                    st.markdown(
                        "#### Alternative views"
                    )

                    for alternative in alternatives:

                        st.write(
                            f"• "
                            f"{alternative['chart_name']} "
                            f"— "
                            f"{alternative['score']:.2f}/100"
                        )

            with scoring_tab:

                st.markdown(
                    "#### Component scores"
                )

                display_score_breakdown(
                    candidate
                )


# ==================================================
# SIDEBAR FILTERS
# ==================================================

def recommendation_controls(
    recommendations
):
    """
    Build recommendation filtering controls.
    """

    with st.sidebar:

        st.divider()

        st.header(
            "Recommendations"
        )

        available_chart_types = sorted({
            candidate["chart"]
            for candidate in recommendations
        })

        selected_chart_types = (
            st.multiselect(
                "Chart types",
                options=available_chart_types,
                default=available_chart_types,
                format_func=chart_display_name
            )
        )

        minimum_score = st.slider(
            "Minimum score",
            min_value=0,
            max_value=100,
            value=35,
            help=(
                "Hide recommendations below "
                "this score."
            )
        )

        maximum_display = min(
            15,
            len(recommendations)
        )

        if maximum_display > 0:

            top_n = st.slider(
                "Maximum recommendations",
                min_value=1,
                max_value=maximum_display,
                value=min(
                    5,
                    maximum_display
                )
            )

        else:

            top_n = 0

    return (
        selected_chart_types,
        minimum_score,
        top_n
    )


# ==================================================
# RECOMMENDATIONS TAB
# ==================================================

def display_recommendations_tab(
    df,
    candidates,
    recommendations,
    selected_chart_types,
    minimum_score,
    top_n
):
    """
    Display globally ranked recommendations.
    """

    st.subheader(
        "Recommended Visualizations"
    )

    st.write(
        f"The engine generated **{len(candidates)} valid "
        f"visualization candidates** and ranked them by "
        f"chart suitability and pattern strength."
    )

    filtered = [
        candidate
        for candidate in recommendations
        if (
            candidate["chart"]
            in selected_chart_types
            and candidate["score"]
            >= minimum_score
        )
    ]

    filtered = filtered[
        :top_n
    ]

    if not filtered:

        st.warning(
            "No recommendations match the "
            "current filters."
        )

        return

    st.caption(
        f"Showing {len(filtered)} recommendation"
        f"{'' if len(filtered) == 1 else 's'}."
    )

    for rank, candidate in enumerate(
        filtered,
        start=1
    ):

        display_recommendation(
            df,
            candidate,
            rank
        )


# ==================================================
# DATASET TAB
# ==================================================

def display_dataset_tab(
    df,
    profile,
    source_name
):
    """
    Display dataset metadata and preview.
    """

    st.subheader(
        "Dataset"
    )

    st.caption(
        f"Source: {source_name}"
    )

    st.markdown(
        "### Inferred column types"
    )

    st.write(
        "The engine combines pandas data types, "
        "column names, cardinality, and observed "
        "values to infer how each column should "
        "be interpreted."
    )

    st.dataframe(
        build_column_profile_table(
            profile
        ),
        width="stretch",
        hide_index=True
    )

    st.markdown(
        "### Data preview"
    )

    preview_rows = min(
        100,
        len(df)
    )

    st.caption(
        f"Showing the first "
        f"{preview_rows} rows."
    )

    preview = build_display_preview(
        df,
        profile,
        rows=preview_rows
    )

    st.dataframe(
        preview,
        width="stretch",
        hide_index=True
    )


# ==================================================
# HOW IT WORKS
# ==================================================

def method_card(
    number,
    title,
    description
):
    """
    Display one compact methodology card.
    """

    with st.container(
        border=True
    ):

        st.markdown(
            f"### {number}. {title}"
        )

        st.write(
            description
        )


def display_method_tab():
    """
    Explain the recommendation pipeline.
    """

    st.subheader(
        "How Dataviz Engine Works"
    )

    st.write(
        "The engine separates whether a chart "
        "**can be made well** from whether it is "
        "**likely to reveal something useful**."
    )

    row1_col1, row1_col2, row1_col3 = (
        st.columns(3)
    )

    with row1_col1:

        method_card(
            1,
            "Profile",
            (
                "Analyze types, missing values, "
                "cardinality, and distributions."
            )
        )

    with row1_col2:

        method_card(
            2,
            "Generate",
            (
                "Create semantically valid bar, "
                "scatter, line, histogram, and "
                "box-plot candidates."
            )
        )

    with row1_col3:

        method_card(
            3,
            "Evaluate",
            (
                "Measure semantic fit, readability, "
                "data quality, and sample support."
            )
        )

    row2_col1, row2_col2 = (
        st.columns(2)
    )

    with row2_col1:

        method_card(
            4,
            "Measure patterns",
            (
                "Use chart-specific statistics to "
                "estimate how much useful structure "
                "the chart may reveal."
            )
        )

    with row2_col2:

        method_card(
            5,
            "Rank & diversify",
            (
                "Combine suitability and pattern "
                "strength, rank candidates, and "
                "remove redundant views."
            )
        )

    st.markdown(
        "### Recommendation score"
    )

    st.write(
        "A high score requires both a chart that "
        "is appropriate for the data and evidence "
        "of an informative pattern. A visualization "
        "cannot rank highly simply because it is "
        "technically valid."
    )

    score1, score2, score3, score4 = (
        st.columns(4)
    )

    score1.metric(
        "Excellent",
        "80–100"
    )

    score2.metric(
        "Strong",
        "65–79"
    )

    score3.metric(
        "Moderate",
        "50–64"
    )

    score4.metric(
        "Weak",
        "35–49"
    )


# ==================================================
# MAIN APPLICATION
# ==================================================

def main():
    """
    Run the Dataviz Engine frontend.
    """

    display_header()

    df, source_name = (
        load_data()
    )

    # -----------------------------------
    # Dataset validation
    # -----------------------------------

    if df.empty:

        st.error(
            "The dataset contains no rows."
        )

        st.stop()

    if len(df.columns) == 0:

        st.error(
            "The dataset contains no columns."
        )

        st.stop()

    # -----------------------------------
    # Analysis
    # -----------------------------------

    try:

        with st.spinner(
            "Analyzing dataset..."
        ):

            (
                profile,
                candidates,
                recommendations
            ) = analyze_dataset(
                df
            )

    except Exception as error:

        st.error(
            "The visualization engine encountered "
            "an error while analyzing this dataset."
        )

        st.exception(
            error
        )

        st.stop()

    # -----------------------------------
    # Overview
    # -----------------------------------

    st.divider()

    display_dataset_metrics(
        df,
        profile,
        candidates,
        recommendations
    )

    # -----------------------------------
    # Controls
    # -----------------------------------

    (
        selected_chart_types,
        minimum_score,
        top_n
    ) = recommendation_controls(
        recommendations
    )

    # -----------------------------------
    # Navigation
    # -----------------------------------

    st.divider()

    (
        recommendations_tab,
        dataset_tab,
        method_tab
    ) = st.tabs(
        [
            "Recommendations",
            "Dataset",
            "How it works"
        ]
    )

    with recommendations_tab:

        display_recommendations_tab(
            df,
            candidates,
            recommendations,
            selected_chart_types,
            minimum_score,
            top_n
        )

    with dataset_tab:

        display_dataset_tab(
            df,
            profile,
            source_name
        )

    with method_tab:

        display_method_tab()


if __name__ == "__main__":
    main()