"""Exported code reproduces each chart's data exactly; the HTML report is complete."""

import re

import numpy as np
import pandas as pd
import pytest

from tests.conftest import REPO_ROOT
from visift.export import chart_code, html_report
from visift.pipeline import analyze
from visift.visualization.renderer import render_candidate

SAMPLES = {
    "src/visift/samples/penguins.csv": "species",
    "src/visift/samples/titanic.csv": "survived",
    "src/visift/samples/bike_sharing_day.csv": "cnt",
    "src/visift/samples/bank_marketing_sample.csv": "y",
    "synthetic_sales.csv": None,
}

TRACE_FIELDS = ("x", "y")


def _charts_for(path, target):
    df = pd.read_csv(REPO_ROOT / path)
    analysis = analyze(df, target=target)
    charts = list(analysis["recommendations"][:10])
    charts += [
        {"chart": "scatter", "x": p["x"], "y": p["y"], "group": p["group"]}
        for p in analysis["paradoxes"]
    ]
    return df, charts


def _same(a, b):
    if a is None or b is None:
        return a is None and b is None
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    if np.issubdtype(a.dtype, np.datetime64) or np.issubdtype(b.dtype, np.datetime64):
        return bool((pd.to_datetime(a) == pd.to_datetime(b)).all())
    try:
        return bool(np.allclose(a.astype(float), b.astype(float), equal_nan=True))
    except (TypeError, ValueError):
        return bool((a.astype(str) == b.astype(str)).all())


def _assert_same_data(expected, actual, label):
    assert len(actual.data) == len(expected.data), label
    for want, got in zip(expected.data, actual.data, strict=True):
        assert want.type == got.type, label
        for field in TRACE_FIELDS:
            assert _same(getattr(want, field, None), getattr(got, field, None)), (
                label,
                field,
            )
        if want.type == "histogram":
            assert want.nbinsx == got.nbinsx, label


CASES = [
    pytest.param(path, mode, id=f"{path.split('/')[-1]}-{mode}")
    for path in SAMPLES
    for mode in ("no-target", "target")
    if not (mode == "target" and SAMPLES[path] is None)
]

CHECKED: list[str] = []


@pytest.mark.slow
@pytest.mark.parametrize(("path", "mode"), CASES)
def test_snippets_reproduce_chart_data(tmp_path, path, mode):
    target = SAMPLES[path] if mode == "target" else None
    df, charts = _charts_for(path, target)
    csv = tmp_path / "data.csv"
    df.to_csv(csv, index=False)
    reloaded = pd.read_csv(csv)

    for chart in charts:
        code = chart_code(chart, str(csv), reloaded)
        namespace: dict = {}
        exec(compile(code, "<snippet>", "exec"), namespace)  # noqa: S102
        label = f"{path} {mode} {chart['chart']} {chart['x']} {chart.get('y')}"
        _assert_same_data(render_candidate(reloaded, chart), namespace["fig"], label)
        CHECKED.append(label)

    assert charts


def test_snippets_are_short_and_self_contained():
    df, charts = _charts_for("src/visift/samples/penguins.csv", "species")
    for chart in charts:
        code = chart_code(chart, "penguins.csv", df)
        assert "visift" not in code.replace("Visift", "")
        assert code.startswith("# ")
        assert len(code.splitlines()) <= 30


def test_html_report_contents():
    df = pd.read_csv(REPO_ROOT / "src/visift/samples/penguins.csv")
    analysis = analyze(df, target="species")
    recommendations = analysis["recommendations"][:5]
    page = html_report(
        df,
        source_name="Palmer Penguins",
        stats_line="344 rows",
        target="species",
        paradoxes=analysis["paradoxes"],
        recommendations=recommendations,
        notes=analysis["notes"],
        insight=lambda frame, chart: "takeaway",
    )
    charts = len(recommendations) + len(analysis["paradoxes"])
    assert page.count('class="plotly-graph-div"') == charts
    # plotly.js is embedded (no external <script src>), so the page works offline;
    # "cdn.plot.ly" appears only inside the library as a topojson default.
    assert re.findall(r"<script[^>]*\bsrc=", page) == []
    assert page.count("cdn.plot.ly") <= 1
    assert page.count("Plotly.newPlot") == charts
    assert "Worth a closer look" in page and "Outcome in focus" in page
    assert "github.com/josh-hu1/Visift" in page
    assert page.count("pattern factor (0.35 + 0.65") == len(recommendations)
