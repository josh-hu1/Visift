import numpy as np
import pandas as pd
import plotly.graph_objects as go

import visift
from visift.datasets import load_sample
from visift.overview import MAX_COLUMNS, correlation_overview
from visift.profiler import profile_dataset
from visift.visualization.renderer import render_correlation_overview


def _overview(df):
    return correlation_overview(df, profile_dataset(df))


def test_needs_three_numeric_columns(rng):
    df = pd.DataFrame(
        {"a": rng.normal(size=100), "b": rng.normal(size=100), "c": ["x", "y"] * 50}
    )
    assert _overview(df) is None


def test_penguins_matrix_is_spearman_with_a_blank_diagonal():
    df = load_sample("penguins")
    overview = _overview(df)

    assert sorted(overview["columns"]) == sorted(
        ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
    )
    rho = overview["rho"]
    assert np.isnan(np.diag(rho.to_numpy())).all()
    np.testing.assert_allclose(rho.to_numpy(), rho.to_numpy().T, equal_nan=True)

    pair = df[["flipper_length_mm", "body_mass_g"]].dropna()
    expected = pair["flipper_length_mm"].corr(pair["body_mass_g"], method="spearman")
    assert rho.loc["flipper_length_mm", "body_mass_g"] == np.float64(expected)
    assert overview["rows"].loc["flipper_length_mm", "body_mass_g"] == len(pair)


def test_columns_that_move_together_sit_together(rng):
    n = 300
    a, c = rng.normal(size=n), rng.normal(size=n)
    df = pd.DataFrame(
        {
            "a": a,
            "c": c,
            "noise": rng.normal(size=n),
            "a_twin": a + rng.normal(0, 0.3, n),
            "c_twin": -c + rng.normal(0, 0.3, n),
        }
    )
    order = _overview(df)["columns"]

    assert abs(order.index("a") - order.index("a_twin")) == 1
    assert abs(order.index("c") - order.index("c_twin")) == 1


def test_thin_pairs_are_blank(rng):
    n = 200
    df = pd.DataFrame(
        {
            "a": rng.normal(size=n),
            "b": rng.normal(size=n),
            "sparse": np.where(np.arange(n) < 8, rng.normal(size=n), np.nan),
        }
    )
    overview = _overview(df)
    assert np.isnan(overview["rho"].loc["a", "sparse"])
    assert overview["rows"].loc["a", "sparse"] == 8

    fig = render_correlation_overview(overview)
    hover = np.array(fig.data[0].customdata)
    assert any("too few shared rows" in cell for cell in hover.ravel())


def test_wide_tables_keep_the_most_correlated_columns(rng):
    n = 200
    base = rng.normal(size=n)
    data = {f"x{i:02d}": rng.normal(size=n) for i in range(MAX_COLUMNS + 5)}
    data["x03"] = base
    data["x29"] = base + rng.normal(0, 0.1, n)
    overview = _overview(pd.DataFrame(data))

    assert len(overview["columns"]) == MAX_COLUMNS
    assert overview["numeric_columns"] == MAX_COLUMNS + 5
    assert {"x03", "x29"} <= set(overview["columns"])


def test_figure_labels_values_and_colors():
    fig = render_correlation_overview(_overview(load_sample("penguins")))
    (heatmap,) = fig.data

    assert isinstance(heatmap, go.Heatmap)
    assert "Body Mass (g)" in heatmap.x
    assert (heatmap.zmin, heatmap.zmax) == (-1, 1)
    # Diverging: negative and positive poles, neutral midpoint.
    assert [stop[0] for stop in heatmap.colorscale] == [0.0, 0.5, 1.0]
    # Small matrices show their values; the diagonal stays empty.
    assert heatmap.text[0][0] == ""
    assert heatmap.text[0][1] != ""


def test_large_matrices_have_no_cell_text(rng):
    df = pd.DataFrame({f"x{i}": rng.normal(size=100) for i in range(15)})
    fig = render_correlation_overview(_overview(df))
    assert all(cell == "" for row in fig.data[0].text for cell in row)


def test_notebook_api():
    recs = visift.recommend(load_sample("penguins"), top=3)
    assert isinstance(recs.correlation_overview(), go.Figure)

    titanic = load_sample("titanic")[["survived", "sex", "age", "fare"]]
    assert visift.recommend(titanic, top=1).correlation_overview() is None
