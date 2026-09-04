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
    visualization_title
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
# HELPERS
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
    Convert semantic type identifiers into
    cleaner display text.
    """

    return (
        str(value)
        .replace("_", " ")
        .title()
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

            uploaded_file = st.file_uploader(
                "Upload CSV",
                type=["csv"],
                help=(
                    "Upload a CSV dataset for "
                    "automatic visualization analysis."
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

    Results are cached so changing frontend
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
    Convert column profiles into a clean
    dataframe for display.
    """

    rows = []

    for column in (
        profile["column_profiles"]
    ):

        rows.append({
            "Column": column["name"],
            "Semantic Type": (
                format_semantic_type(
                    column["semantic_type"]
                )
            ),
            "Pandas Type": (
                column["pandas_dtype"]
            ),
            "Unique Values": (
                column["unique_count"]
            ),
            "Missing": (
                column["missing_count"]
            ),
            "Missing %": (
                column["missing_percent"]
            )
        })

    return pd.DataFrame(
        rows
    )


# ==================================================
# SCORE DISPLAY
# ==================================================

def display_score_breakdown(
    candidate
):
    """
    Display all recommendation score components.
    """

    component_rows = []

    for name, value in (
        candidate["components"].items()
    ):

        component_rows.append({
            "Component": (
                name
                .replace("_", " ")
                .title()
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
    Render a polished recommendation card.
    """

    title = visualization_title(
        candidate
    )

    chart_name = chart_display_name(
        candidate["chart"]
    )

    score = candidate["score"]

    strength = recommendation_strength(
        score
    )

    color = strength_color(
        strength
    )

    with st.container(
        border=True
    ):

        header_left, header_right = (
            st.columns(
                [5, 1]
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

        metric1, metric2, metric3 = (
            st.columns(3)
        )

        metric1.metric(
            "Insight signal",
            (
                f"{candidate['components']['signal']:.1f}"
                "/100"
            )
        )

        metric2.metric(
            "Visualization quality",
            (
                f"{candidate['components']['visualization_quality']:.1f}"
                "/100"
            )
        )

        metric3.metric(
            "Alternative views",
            len(
                candidate.get(
                    "alternative_charts",
                    []
                )
            )
        )

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
                    "#### Why this chart ranked highly"
                )

                for reason in summary_reasons(
                    candidate
                ):

                    st.write(
                        f"• {reason}"
                    )

                alternatives = candidate.get(
                    "alternative_charts",
                    []
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
        f"visualization quality and insight strength."
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
        f"Showing the first {preview_rows} rows."
    )

    st.dataframe(
        df.head(
            preview_rows
        ),
        width="stretch",
        hide_index=True
    )


# ==================================================
# HOW IT WORKS TAB
# ==================================================

def display_method_tab():
    """
    Explain the recommendation pipeline.
    """

    st.subheader(
        "How Dataviz Engine Works"
    )

    st.write(
        "The engine separates visualization validity "
        "from visualization usefulness."
    )

    with st.container(
        border=True
    ):

        st.markdown(
            "### 1. Profile the dataset"
        )

        st.write(
            "Columns are analyzed for missing values, "
            "cardinality, distribution statistics, and "
            "semantic meaning."
        )

    with st.container(
        border=True
    ):

        st.markdown(
            "### 2. Generate valid candidates"
        )

        st.write(
            "Semantically compatible combinations are "
            "generated for bar charts, scatterplots, "
            "line charts, histograms, and box plots."
        )

    with st.container(
        border=True
    ):

        st.markdown(
            "### 3. Measure visualization quality"
        )

        st.write(
            "Each candidate is evaluated for semantic fit, "
            "readability, data quality, and sample support."
        )

    with st.container(
        border=True
    ):

        st.markdown(
            "### 4. Measure insight strength"
        )

        st.write(
            "Chart-specific statistics evaluate whether "
            "the visualization actually reveals a useful "
            "pattern."
        )

    with st.container(
        border=True
    ):

        st.markdown(
            "### 5. Rank and diversify"
        )

        st.write(
            "The final recommendation score combines "
            "quality and signal, then removes redundant "
            "views of the same underlying relationship."
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
    # Overview metrics
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
    # Main navigation
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