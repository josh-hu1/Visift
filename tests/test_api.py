"""The notebook API: visift.recommend / visift.report."""

import sys

import plotly.graph_objects as go
import pytest

import visift
from visift.datasets import SAMPLES, load_sample


@pytest.fixture(scope="module")
def penguins():
    return load_sample("penguins")


@pytest.fixture(scope="module")
def recs(penguins):
    return visift.recommend(penguins, target="species", top=5)


def test_public_names_and_no_streamlit():
    assert {"recommend", "report", "analyze", "__version__"} <= set(dir(visift))
    assert "streamlit" not in sys.modules or "web_app" in sys.modules


def test_recommend_returns_ranked_typed_recommendations(recs):
    assert isinstance(recs, list) and len(recs) == 5
    assert [r.rank for r in recs] == [1, 2, 3, 4, 5]
    first = recs[0]
    assert isinstance(first, visift.Recommendation)
    assert first.driver and "species" in (first.x, first.y)
    assert first.band in {"Excellent", "Strong", "Moderate", "Weak", "Very weak"}
    assert first.takeaway.endswith(".")
    assert first.breakdown["score"] == pytest.approx(first.score, abs=0.01)


def test_matches_the_pipeline_order(penguins, recs):
    ranked = visift.analyze(penguins, target="species")["recommendations"][:5]
    assert [(r.chart, r.x, r.y) for r in recs] == [
        (r["chart"], r["x"], r.get("y")) for r in ranked
    ]


def test_figure_code_and_html(recs):
    first = recs[0]
    assert isinstance(first.figure(), go.Figure)
    code = first.code()
    assert "import pandas as pd" in code and "plotly" in code
    assert "<b>#1 " in first._repr_html_()
    assert "drivers of <b>species</b> first" in recs._repr_html_()


def test_paradoxes_notes_and_frame(recs):
    assert any(p["group"] == "species" for p in recs.paradoxes)
    frame = recs.to_frame()
    assert list(frame["rank"]) == [1, 2, 3, 4, 5] and "takeaway" in frame


def test_unknown_target_raises(penguins):
    with pytest.raises(KeyError, match="not a column"):
        visift.recommend(penguins, target="nope")


def test_report_writes_an_offline_page(penguins, tmp_path):
    path = visift.report(penguins, tmp_path / "r.html", target="species", top=3)
    page = path.read_text()
    assert "Simpson" in page and "species" in page


def test_every_bundled_sample_loads():
    for name, sample in SAMPLES.items():
        df = load_sample(name)
        assert len(df) > 100 and sample.outcome in df.columns
    with pytest.raises(KeyError):
        load_sample("nope")
