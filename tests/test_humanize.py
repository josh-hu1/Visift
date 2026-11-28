"""Column labels: a trailing unit token becomes a parenthesized suffix."""

import pytest

from visift.visualization.renderer import humanize_column_name


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("bill_depth_mm", "Bill Depth (mm)"),
        ("body_mass_g", "Body Mass (g)"),
        ("flipper_length_mm", "Flipper Length (mm)"),
        ("weight_kg", "Weight (kg)"),
        ("weight_lbs", "Weight (lbs)"),
        ("latency_ms", "Latency (ms)"),
        ("growth_pct", "Growth (%)"),
        ("price_usd", "Price (USD)"),
        ("energy_kwh", "Energy (kWh)"),
        ("Duration MS", "Duration (ms)"),
    ],
)
def test_trailing_unit_becomes_a_suffix(name, label):
    assert humanize_column_name(name) == label


@pytest.mark.parametrize(
    ("name", "label"),
    [
        ("mm", "Mm"),  # a unit alone is not a suffix
        ("g", "G"),
        ("mm_count", "Mm Count"),  # only the last token
        ("height_h", "Height H"),  # ambiguous single letters stay
        ("temp_min", "Temp Min"),  # "min" usually means minimum
        ("species", "Species"),
        ("sepal length", "Sepal Length"),
    ],
)
def test_other_names_are_title_cased(name, label):
    assert humanize_column_name(name) == label
