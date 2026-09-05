import re

import pandas as pd


# ==================================================
# NAME / VALUE HELPERS
# ==================================================

def normalize_column_name(series: pd.Series) -> str:
    """
    Normalize a column name for semantic-name heuristics.
    """

    name = str(series.name).strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)

    return name.strip("_")


def column_name_tokens(series: pd.Series) -> set[str]:
    """
    Split a normalized column name into tokens.
    """

    name = normalize_column_name(series)

    if not name:
        return set()

    return {
        token
        for token in name.split("_")
        if token
    }


def values_are_integer_like(series: pd.Series) -> bool:
    """
    Return True when all non-null numeric values are whole-number-like,
    even if pandas stores them as floats because of missing values.
    """

    numeric = pd.to_numeric(
        series.dropna(),
        errors="coerce"
    ).dropna()

    if numeric.empty:
        return False

    fractional = (
        numeric
        - numeric.round()
    ).abs()

    return bool(
        (fractional <= 1e-9).all()
    )


# ==================================================
# DATETIME
# ==================================================

def detect_datetime(series: pd.Series) -> bool:
    """
    Determine whether a non-numeric column is probably datetime data.
    """

    non_null = series.dropna()

    if non_null.empty:
        return False

    try:
        converted = pd.to_datetime(
            non_null,
            errors="coerce",
            format="mixed"
        )

        success_rate = converted.notna().mean()

        return success_rate >= 0.90

    except Exception:
        return False


# ==================================================
# CALENDAR LABEL FIELDS
# ==================================================

def is_calendar_label_text(
    series: pd.Series
) -> bool:
    non_null = series.dropna()

    if non_null.empty:
        return False

    column_name = normalize_column_name(
        series
    )

    normalized_values = {
        str(value).strip().lower()
        for value in non_null.unique()
    }

    month_names = {
        "jan", "january",
        "feb", "february",
        "mar", "march",
        "apr", "april",
        "may",
        "jun", "june",
        "jul", "july",
        "aug", "august",
        "sep", "sept", "september",
        "oct", "october",
        "nov", "november",
        "dec", "december"
    }

    weekday_names = {
        "mon", "monday",
        "tue", "tues", "tuesday",
        "wed", "wednesday",
        "thu", "thur", "thurs", "thursday",
        "fri", "friday",
        "sat", "saturday",
        "sun", "sunday"
    }

    if column_name in {
        "month",
        "mnth",
        "month_name"
    }:
        return bool(
            normalized_values
            and normalized_values.issubset(
                month_names
            )
        )

    if column_name in {
        "weekday",
        "dayofweek",
        "day_of_week",
        "dow"
    }:
        return bool(
            normalized_values
            and normalized_values.issubset(
                weekday_names
            )
        )

    return False


# ==================================================
# IDENTIFIERS
# ==================================================

def is_identifier(series: pd.Series) -> bool:
    """
    Determine whether a column probably represents an identifier.

    Examples:
        customer_id
        user_id
        transaction_id
        UUID
    """

    non_null = series.dropna()

    if non_null.empty:
        return False

    unique_ratio = (
        non_null.nunique()
        / len(non_null)
    )

    column_name = normalize_column_name(series)

    name_suggests_id = (
        column_name == "id"
        or column_name.endswith("_id")
        or "uuid" in column_name
        or "guid" in column_name
        or "identifier" in column_name
    )

    # IDs are usually almost completely unique.
    highly_unique = unique_ratio >= 0.95

    return (
        name_suggests_id
        and highly_unique
    )


# ==================================================
# TEMPORAL NUMERIC FIELDS
# ==================================================

def is_temporal_numeric(series: pd.Series) -> bool:
    """
    Detect numeric columns that represent time.

    Supports literal calendar values such as year=2024 as well as
    compact encodings often found in public/production datasets,
    such as yr={0,1} and mnth={1,...,12}.
    """

    non_null = series.dropna()

    if (
        non_null.empty
        or not values_are_integer_like(series)
    ):
        return False

    column_name = normalize_column_name(series)

    numeric = pd.to_numeric(
        non_null,
        errors="coerce"
    ).dropna()

    if numeric.empty:
        return False

    year_names = {
        "year",
        "yr",
        "calendar_year",
        "fiscal_year"
    }

    month_names = {
        "month",
        "mnth",
        "month_num",
        "month_number"
    }

    quarter_names = {
        "quarter",
        "qtr",
        "quarter_num",
        "quarter_number"
    }

    week_names = {
        "week",
        "wk",
        "week_num",
        "week_number"
    }

    day_names = {
        "day",
        "day_num",
        "day_number"
    }

    if column_name in year_names:
        # A strong name hint lets us support both literal years
        # and compact 0/1-style year encodings.
        return numeric.nunique() <= 300

    if column_name in month_names:
        return bool(
            numeric.between(0, 12).all()
            and numeric.nunique() <= 13
        )

    if column_name in quarter_names:
        return bool(
            numeric.between(0, 4).all()
            and numeric.nunique() <= 5
        )

    if column_name in week_names:
        return bool(
            numeric.between(0, 53).all()
            and numeric.nunique() <= 54
        )

    if column_name in day_names:
        return True

    return False


# ==================================================
# GEOGRAPHIC FIELDS
# ==================================================

def is_geographic(series: pd.Series) -> bool:
    """
    Detect columns that likely represent geographic identifiers.

    Token-based matching avoids accidental substring matches such as
    treating a column like platelet_count as latitude.
    """

    column_name = normalize_column_name(series)
    tokens = column_name_tokens(series)

    geographic_names = {
        "zip",
        "zipcode",
        "zip_code",
        "postal",
        "postal_code",
        "postalcode",
        "latitude",
        "longitude",
        "lat",
        "lon",
        "lng"
    }

    geographic_tokens = {
        "zip",
        "zipcode",
        "postal",
        "postalcode",
        "latitude",
        "longitude",
        "lat",
        "lon",
        "lng"
    }

    return (
        column_name in geographic_names
        or bool(tokens & geographic_tokens)
    )


# ==================================================
# ORDINAL FIELDS
# ==================================================

def is_ordinal(series: pd.Series) -> bool:
    """
    Detect low-cardinality ordered categories such as ratings,
    ranks, severity levels, priorities, and tiers.
    """

    non_null = series.dropna()

    if non_null.empty:
        return False

    column_name = normalize_column_name(series)
    tokens = column_name_tokens(series)

    ordinal_terms = {
        "rating",
        "rank",
        "level",
        "grade",
        "severity",
        "priority",
        "tier",
        "stage",
        "likert"
    }

    name_suggests_ordinal = bool(
        tokens
        & ordinal_terms
    )

    # Keep the useful existing score heuristic, but only when the
    # score has relatively few distinct levels.
    if (
        "score" in tokens
        and non_null.nunique() <= 20
    ):
        name_suggests_ordinal = True

    if not name_suggests_ordinal:
        return False

    return non_null.nunique() <= 20


# ==================================================
# BOOLEAN-LIKE FIELDS
# ==================================================

def is_boolean_text(series: pd.Series) -> bool:
    """
    Detect common text encodings of binary values.

    Examples:
        yes / no
        true / false
        y / n
        on / off
        "1" / "0"
    """

    non_null = series.dropna()

    if non_null.empty:
        return False

    normalized_values = {
        str(value).strip().lower()
        for value in non_null.unique()
    }

    recognized_pairs = (
        {"yes", "no"},
        {"true", "false"},
        {"y", "n"},
        {"on", "off"},
        {"1", "0"},
    )

    return any(
        normalized_values == pair
        for pair in recognized_pairs
    )


def is_boolean_numeric(series: pd.Series) -> bool:
    """
    Detect numeric 0/1 indicators.

    Exact {0,1} data are treated as boolean unless a stronger
    temporal/category-code semantic rule matched first.

    Constant 0-only or 1-only fields are considered boolean only
    when their column name strongly suggests an indicator.
    """

    non_null = series.dropna()

    if (
        non_null.empty
        or not values_are_integer_like(series)
    ):
        return False

    numeric = pd.to_numeric(
        non_null,
        errors="coerce"
    ).dropna()

    unique_values = {
        int(value)
        for value in numeric.unique()
    }

    if unique_values == {0, 1}:
        return True

    column_name = normalize_column_name(series)
    tokens = column_name_tokens(series)

    boolean_terms = {
        "flag",
        "indicator",
        "active",
        "enabled",
        "disabled",
        "holiday",
        "workingday",
        "weekend",
        "member",
        "subscribed",
        "selected",
        "verified"
    }

    name_suggests_boolean = (
        column_name.startswith("is_")
        or column_name.startswith("has_")
        or bool(tokens & boolean_terms)
    )

    return bool(
        name_suggests_boolean
        and unique_values
        and unique_values.issubset({0, 1})
    )


# ==================================================
# NUMERIC CATEGORY CODES
# ==================================================

def is_numeric_categorical_code(series: pd.Series) -> bool:
    """
    Detect low-cardinality numeric codes that represent categories
    rather than quantities.

    The rule requires BOTH:
        1. a small number of integer-like levels, and
        2. category-like evidence in the column name.

    This prevents genuine counts such as units_sold or attempts from
    being converted to categories merely because they have few values.
    """

    non_null = series.dropna()

    if non_null.empty:
        return False

    unique_count = non_null.nunique()

    if (
        unique_count < 2
        or unique_count > 30
        or not values_are_integer_like(series)
    ):
        return False

    column_name = normalize_column_name(series)
    tokens = column_name_tokens(series)

    category_terms = {
        "category",
        "class",
        "group",
        "segment",
        "status",
        "type",
        "kind",
        "mode",
        "code",
        "cluster",
        "bucket",
        "band",
        "sex",
        "gender",
        "region",
        "zone",
        "season",
        "weekday",
        "dow",
        "weather",
        "weathersit",
        "condition"
    }

    exact_category_names = {
        "season",
        "weekday",
        "dayofweek",
        "day_of_week",
        "dow",
        "weathersit",
        "weather_situation"
    }

    return (
        column_name in exact_category_names
        or bool(tokens & category_terms)
        or column_name.endswith("_code")
        or column_name.endswith("_type")
        or column_name.endswith("_class")
        or column_name.endswith("class")
        or column_name.endswith("_category")
        or column_name.endswith("_status")
    )


# ==================================================
# MAIN CLASSIFIER
# ==================================================

def classify_column(series: pd.Series) -> str:
    """
    Infer the semantic type of a pandas Series.

    Possible outputs:
        empty
        boolean
        datetime
        identifier
        temporal
        geographic
        ordinal
        numeric_discrete
        numeric_continuous
        categorical
        text
        unknown
    """

    non_null = series.dropna()

    if non_null.empty:
        return "empty"

    # --------------------
    # Boolean
    # --------------------
    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    # --------------------
    # Already parsed datetime
    # --------------------
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # --------------------
    # Identifier
    # --------------------
    if is_identifier(series):
        return "identifier"

    # --------------------
    # Numeric
    # --------------------
    if pd.api.types.is_numeric_dtype(series):

        unique_count = non_null.nunique()

        # Geographic values should not be treated as quantities.
        if is_geographic(series):
            return "geographic"

        # Numeric representations of time.
        #
        # This intentionally runs before binary detection because
        # compact time encodings such as yr={0,1} are temporal.
        if is_temporal_numeric(series):
            return "temporal"

        # Ordered categories such as 1-5 ratings.
        if is_ordinal(series):
            return "ordinal"

        # Named category codes such as season=1..4 or weekday=0..6.
        #
        # This runs before generic binary detection so a coded binary
        # class can remain categorical when its name clearly says so.
        if is_numeric_categorical_code(series):
            return "categorical"

        # Numeric yes/no indicators such as holiday or workingday.
        if is_boolean_numeric(series):
            return "boolean"

        # Preserve genuine low-cardinality counts when there is no
        # stronger semantic evidence of a category.
        if (
            pd.api.types.is_integer_dtype(series)
            or values_are_integer_like(series)
        ):
            if unique_count <= 20:
                return "numeric_discrete"

        return "numeric_continuous"

    # --------------------
    # Strings / object columns
    # --------------------
    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
    ):

        if is_boolean_text(series):
            return "boolean"

        if is_calendar_label_text(series):
            return "categorical"

        if detect_datetime(series):
            return "datetime"

        unique_count = non_null.nunique()
        row_count = len(non_null)

        unique_ratio = (
            unique_count
            / row_count
        )

        # Highly unique strings are more likely to be
        # free-form text than categorical data.
        if unique_ratio >= 0.80:
            return "text"

        # Few distinct repeated values -> categorical.
        if unique_count <= 20:
            return "categorical"

        # Larger sets can still be categorical if
        # values repeat frequently.
        if (
            unique_count <= 100
            and unique_ratio < 0.50
        ):
            return "categorical"

        return "text"

    return "unknown"
