from itertools import combinations


NUMERIC_TYPES = {
    "numeric_continuous",
    "numeric_discrete"
}

CATEGORY_TYPES = {
    "categorical",
    "ordinal",
    "boolean",
    "geographic"
}

# Category-like fields that can sensibly define groups for
# a binary-rate comparison.
#
# Boolean is intentionally excluded here. When both variables
# are boolean, there is no obvious predictor/outcome direction
# to infer automatically from semantics alone.
RATE_PREDICTOR_TYPES = {
    "categorical",
    "ordinal",
    "geographic"
}

TIME_TYPES = {
    "datetime",
    "temporal"
}


def get_semantic_type(
    profile,
    column_name
):
    """
    Get semantic type for a column from the dataset profile.
    """

    for column in profile[
        "column_profiles"
    ]:

        if (
            column["name"]
            == column_name
        ):

            return column[
                "semantic_type"
            ]

    return None


def generate_candidates(
    df,
    profile
):
    """
    Generate valid visualization candidates based on
    semantic column types.

    Candidate generation is intentionally permissive:
    it decides whether a visualization can make semantic
    sense. The scoring layer decides whether the resulting
    visualization is actually informative.
    """

    candidates = []

    columns = list(
        df.columns
    )

    # ---------------------------------
    # Single-column visualizations
    # ---------------------------------

    for column in columns:

        semantic_type = (
            get_semantic_type(
                profile,
                column
            )
        )

        # Numeric distribution
        if (
            semantic_type
            in NUMERIC_TYPES
        ):

            candidates.append({
                "chart": "histogram",
                "x": column
            })

        # Category counts
        if (
            semantic_type
            in CATEGORY_TYPES
        ):

            candidates.append({
                "chart": "bar",
                "x": column,
                "aggregation": "count"
            })

    # ---------------------------------
    # Two-column visualizations
    # ---------------------------------

    for (
        col1,
        col2
    ) in combinations(
        columns,
        2
    ):

        type1 = (
            get_semantic_type(
                profile,
                col1
            )
        )

        type2 = (
            get_semantic_type(
                profile,
                col2
            )
        )

        # -----------------------------
        # Numeric + Numeric
        # -----------------------------

        if (
            type1 in NUMERIC_TYPES
            and type2 in NUMERIC_TYPES
        ):

            candidates.append({
                "chart": "scatter",
                "x": col1,
                "y": col2
            })

        # -----------------------------
        # Time + Numeric
        # -----------------------------

        if (
            type1 in TIME_TYPES
            and type2 in NUMERIC_TYPES
        ):

            candidates.append({
                "chart": "line",
                "x": col1,
                "y": col2
            })

        elif (
            type2 in TIME_TYPES
            and type1 in NUMERIC_TYPES
        ):

            candidates.append({
                "chart": "line",
                "x": col2,
                "y": col1
            })

        # -----------------------------
        # Category + Numeric
        # -----------------------------

        if (
            type1 in CATEGORY_TYPES
            and type2 in NUMERIC_TYPES
        ):

            candidates.append({
                "chart": "bar",
                "x": col1,
                "y": col2,
                "aggregation": "mean"
            })

            candidates.append({
                "chart": "box",
                "x": col1,
                "y": col2
            })

        elif (
            type2 in CATEGORY_TYPES
            and type1 in NUMERIC_TYPES
        ):

            candidates.append({
                "chart": "bar",
                "x": col2,
                "y": col1,
                "aggregation": "mean"
            })

            candidates.append({
                "chart": "box",
                "x": col2,
                "y": col1
            })

        # -----------------------------
        # Category + Boolean Outcome
        # -----------------------------
        #
        # The mean of a boolean outcome is the positive rate:
        #
        #     mean(converted) = conversion rate
        #     mean(churned)   = churn rate
        #     mean(survived)  = survival rate
        #
        # This reuses the existing mean-bar scoring path rather
        # than introducing a separate chart/scorer family.

        if (
            type1 in RATE_PREDICTOR_TYPES
            and type2 == "boolean"
        ):

            candidates.append({
                "chart": "bar",
                "x": col1,
                "y": col2,
                "aggregation": "mean"
            })

        elif (
            type2 in RATE_PREDICTOR_TYPES
            and type1 == "boolean"
        ):

            candidates.append({
                "chart": "bar",
                "x": col2,
                "y": col1,
                "aggregation": "mean"
            })

    return [
        candidate
        for candidate in candidates
        if (
            candidate.get("chart") != "line"
            or is_line_eligible_time_axis(
                candidate.get("x"),
                get_semantic_type(
                    profile,
                    candidate.get("x")
                )
            )
        )
    ]


def is_line_eligible_time_axis(
    column_name,
    semantic_type
):
    # Datetimes and non-cyclical temporal fields remain eligible.
    # Standalone cyclical calendar positions do not.
    if semantic_type == "datetime":
        return True

    if semantic_type != "temporal":
        return False

    normalized = (
        str(column_name)
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )

    cyclical_names = {
        "month",
        "mnth",
        "month_num",
        "month_number",
        "month_of_year",
        "quarter",
        "qtr",
        "quarter_num",
        "quarter_number",
        "quarter_of_year",
        "week",
        "wk",
        "week_num",
        "week_number",
        "week_of_year",
        "day",
        "day_num",
        "day_number",
        "day_of_month",
        "dom",
        "weekday",
        "dayofweek",
        "day_of_week",
        "dow",
        "hour",
        "hr",
        "hour_of_day"
    }

    return normalized not in cyclical_names

