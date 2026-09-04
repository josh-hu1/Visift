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

TIME_TYPES = {
    "datetime",
    "temporal"
}


def get_semantic_type(profile, column_name):
    """
    Get semantic type for a column from the dataset profile.
    """

    for column in profile["column_profiles"]:
        if column["name"] == column_name:
            return column["semantic_type"]

    return None


def generate_candidates(df, profile):
    """
    Generate valid visualization candidates based on
    semantic column types.
    """

    candidates = []

    columns = list(df.columns)

    # ---------------------------------
    # Single-column visualizations
    # ---------------------------------

    for column in columns:

        semantic_type = get_semantic_type(
            profile,
            column
        )

        # Numeric distribution
        if semantic_type in NUMERIC_TYPES:
            candidates.append({
                "chart": "histogram",
                "x": column
            })

        # Category counts
        if semantic_type in CATEGORY_TYPES:
            candidates.append({
                "chart": "bar",
                "x": column,
                "aggregation": "count"
            })

    # ---------------------------------
    # Two-column visualizations
    # ---------------------------------

    for col1, col2 in combinations(columns, 2):

        type1 = get_semantic_type(profile, col1)
        type2 = get_semantic_type(profile, col2)

        # -----------------------------
        # Numeric + Numeric
        # -----------------------------

        if type1 in NUMERIC_TYPES and type2 in NUMERIC_TYPES:

            candidates.append({
                "chart": "scatter",
                "x": col1,
                "y": col2
            })

        # -----------------------------
        # Time + Numeric
        # -----------------------------

        if type1 in TIME_TYPES and type2 in NUMERIC_TYPES:

            candidates.append({
                "chart": "line",
                "x": col1,
                "y": col2
            })

        elif type2 in TIME_TYPES and type1 in NUMERIC_TYPES:

            candidates.append({
                "chart": "line",
                "x": col2,
                "y": col1
            })

        # -----------------------------
        # Category + Numeric
        # -----------------------------

        if type1 in CATEGORY_TYPES and type2 in NUMERIC_TYPES:

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

        elif type2 in CATEGORY_TYPES and type1 in NUMERIC_TYPES:

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

    return candidates