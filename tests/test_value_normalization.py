import pandas as pd

from visift.value_normalization import coerce_numeric_or_boolean, is_boolean_like_series


def test_yes_no_text_is_boolean_like_and_coerces_to_0_1():
    series = pd.Series(["yes", "no", "Yes", "NO", None])
    assert is_boolean_like_series(series)
    assert coerce_numeric_or_boolean(series).dropna().tolist() == [1, 0, 1, 0]


def test_numeric_strings_coerce_to_numbers():
    series = pd.Series(["1.5", "2", "x"])
    coerced = coerce_numeric_or_boolean(series)
    assert coerced.iloc[:2].tolist() == [1.5, 2.0]
    assert pd.isna(coerced.iloc[2])
