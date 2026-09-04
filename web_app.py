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
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Data Visualization Recommender",
    page_icon="📊",
    layout="wide"
)


# ==================================================
# DATA LOADING
# ==================================================

def load_data():
    """
    Allow the user to use the sample dataset
    or upload a CSV.
    """

    st.sidebar.header(
        "Dataset"
    )

    data_source = st.sidebar.radio(
        "Data source",
        [
            "Sample dataset",
            "Upload CSV"
        ]
    )

    if data_source == "Upload CSV":

        uploaded_file = (
            st.sidebar.file_uploader(
                "Upload a CSV file",
                type=["csv"]
            )
        )

        if uploaded_file is None:

            st.info(
                "Upload a CSV file using the "
                "sidebar to begin."
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
                "was not found."
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
# DATASET SUMMARY
# ==================================================

def display_dataset_summary(
    df,
    profile,
    source_name
):
    """
    Display headline information about
    the active dataset.
    """

    st.subheader(
        "Dataset Overview"
    )

    st.caption(
        f"Source: {source_name}"
    )

    column1, column2, column3 = (
        st.columns(3)
    )

    column1.metric(
        "Rows",
        f"{profile['rows']:,}"
    )

    column2.metric(
        "Columns",
        profile["columns"]
    )

    missing_values = int(
        df.isna().sum().sum()
    )

    column3.metric(
        "Missing Values",
        f"{missing_values:,}"
    )


def display_column_types(
    profile
):
    """
    Display the engine's semantic type
    classification for every column.
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
                column["semantic_type"]
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

    type_df = pd.DataFrame(
        rows
    )

    st.dataframe(
        type_df,
        width="stretch",
        hide_index=True
    )


# ==================================================
# RECOMMENDATION DISPLAY
# ==================================================

def display_score_breakdown(
    candidate
):
    """
    Display detailed scoring information.
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


def display_recommendation(
    df,
    candidate,
    rank
):
    """
    Display one ranked visualization
    recommendation.
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

    st.markdown("---")

    st.subheader(
        f"#{rank} — {title}"
    )

    metric1, metric2, metric3 = (
        st.columns(3)
    )

    metric1.metric(
        "Recommendation Score",
        f"{score:.2f}/100"
    )

    metric2.metric(
        "Strength",
        strength
    )

    metric3.metric(
        "Chart Type",
        chart_name
    )

    fig = render_candidate(
        df,
        candidate
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "displaylogo": False
        }
    )

    with st.expander(
        "Why was this recommended?"
    ):

        insight_column, quality_column = (
            st.columns(2)
        )

        insight_column.metric(
            "Insight Signal",
            (
                f"{candidate['components']['signal']}"
                "/100"
            )
        )

        quality_column.metric(
            "Visualization Quality",
            (
                f"{candidate['components']['visualization_quality']}"
                "/100"
            )
        )

        st.markdown(
            "#### Key Evidence"
        )

        for reason in summary_reasons(
            candidate
        ):

            st.write(
                f"• {reason}"
            )

        st.markdown(
            "#### Score Breakdown"
        )

        display_score_breakdown(
            candidate
        )

        alternatives = candidate.get(
            "alternative_charts",
            []
        )

        if alternatives:

            st.markdown(
                "#### Alternative Views"
            )

            for alternative in alternatives:

                st.write(
                    f"• "
                    f"{alternative['chart_name']} "
                    f"— "
                    f"{alternative['score']:.2f}/100"
                )


# ==================================================
# MAIN APP
# ==================================================

def main():
    """
    Run the Streamlit visualization
    recommendation application.
    """

    st.title(
        "Data Visualization Recommender"
    )

    st.write(
        "Automatically profile a dataset, "
        "generate valid visualization candidates, "
        "and rank the charts most likely to reveal "
        "useful patterns."
    )

    df, source_name = (
        load_data()
    )

    # -----------------------------------
    # Basic validation
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
    # Run engine
    # -----------------------------------

    try:

        profile = profile_dataset(
            df
        )

        candidates = (
            generate_candidates(
                df,
                profile
            )
        )

        scored = (
            score_all_candidates(
                df,
                candidates,
                profile
            )
        )

        recommendations = (
            diversify_recommendations(
                scored
            )
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
    # Sidebar recommendation controls
    # -----------------------------------

    st.sidebar.header(
        "Recommendations"
    )

    maximum_recommendations = min(
        15,
        len(recommendations)
    )

    if maximum_recommendations > 0:

        top_n = st.sidebar.slider(
            "Number of recommendations",
            min_value=1,
            max_value=maximum_recommendations,
            value=min(
                5,
                maximum_recommendations
            )
        )

    else:

        top_n = 0

    minimum_score = (
        st.sidebar.slider(
            "Minimum score",
            min_value=0,
            max_value=100,
            value=0
        )
    )

    # -----------------------------------
    # Overview
    # -----------------------------------

    display_dataset_summary(
        df,
        profile,
        source_name
    )

    with st.expander(
        "View Inferred Column Types"
    ):

        display_column_types(
            profile
        )

    with st.expander(
        "Preview Dataset"
    ):

        st.dataframe(
            df.head(100),
            width="stretch"
        )

    # -----------------------------------
    # Recommendations
    # -----------------------------------

    st.markdown("---")

    st.header(
        "Recommended Visualizations"
    )

    st.write(
        f"The engine generated "
        f"**{len(candidates)} valid candidates** "
        f"and ranked them based on visualization "
        f"quality and insight strength."
    )

    filtered = [
        candidate
        for candidate in recommendations
        if candidate["score"]
        >= minimum_score
    ]

    filtered = (
        filtered[:top_n]
    )

    if not filtered:

        st.warning(
            "No recommendations meet "
            "the current score threshold."
        )

        return

    for rank, candidate in enumerate(
        filtered,
        start=1
    ):

        display_recommendation(
            df,
            candidate,
            rank
        )


if __name__ == "__main__":
    main()