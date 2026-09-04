import pandas as pd


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

    unique_count = non_null.nunique()
    row_count = len(non_null)

    unique_ratio = unique_count / row_count

    column_name = str(series.name).lower()

    # Strong hint from column name
    identifier_names = [
        "id",
        "_id",
        "uuid",
        "guid",
        "identifier"
    ]

    name_suggests_id = (
        column_name == "id"
        or column_name.endswith("_id")
        or any(term in column_name for term in ["uuid", "guid"])
    )

    # IDs are usually almost completely unique
    highly_unique = unique_ratio >= 0.95

    return name_suggests_id and highly_unique

def is_temporal_numeric(series: pd.Series) -> bool:
    """
    Detect numeric columns that represent time, such as years.
    """

    non_null = series.dropna()

    if non_null.empty:
        return False

    column_name = str(series.name).lower()

    temporal_names = {
        "year",
        "month",
        "quarter",
        "week",
        "day"
    }

    if column_name not in temporal_names:
        return False

    # Special handling for years
    if column_name == "year":
        return non_null.between(1800, 2200).all()

    return True

def is_geographic(series: pd.Series) -> bool:
    """
    Detect columns that likely represent geographic identifiers.
    """

    column_name = str(series.name).lower()

    geographic_terms = [
        "zip",
        "zipcode",
        "zip_code",
        "postal",
        "postal_code",
        "latitude",
        "longitude",
        "lat",
        "lon",
        "lng"
    ]

    return any(term in column_name for term in geographic_terms)

def is_ordinal(series: pd.Series) -> bool:
    """
    Detect ordered categorical variables such as ratings.
    """

    non_null = series.dropna()

    if non_null.empty:
        return False

    column_name = str(series.name).lower()

    ordinal_terms = [
        "rating",
        "score",
        "rank",
        "level",
        "grade"
    ]

    name_suggests_ordinal = any(
        term in column_name
        for term in ordinal_terms
    )

    if not name_suggests_ordinal:
        return False

    # Ordinal variables usually have relatively few levels
    return non_null.nunique() <= 20

def classify_column(series: pd.Series) -> str:
    """
    Infer the semantic type of a pandas Series.

    Possible outputs:
        empty
        boolean
        datetime
        identifier
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
        if is_temporal_numeric(series):
            return "temporal"

        # Ordered categories such as 1-5 ratings.
        if is_ordinal(series):
            return "ordinal"

        if pd.api.types.is_integer_dtype(series):
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

        if detect_datetime(series):
            return "datetime"

        unique_count = non_null.nunique()
        row_count = len(non_null)

        unique_ratio = unique_count / row_count

        # Highly unique strings are more likely to be
        # free-form text than categorical data.
        if unique_ratio >= 0.80:
            return "text"

        # Few distinct repeated values -> categorical
        if unique_count <= 20:
            return "categorical"

        # Larger sets can still be categorical if
        # values repeat frequently.
        if unique_count <= 100 and unique_ratio < 0.50:
            return "categorical"

        return "text"

    return "unknown"