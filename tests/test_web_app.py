"""Streamlit app behavior (headless AppTest): sample gallery and outcome picker."""

from pathlib import Path

import pandas as pd
import pytest

# The app needs the [app] extra (streamlit); skip cleanly without it.
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

import web_app  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "web_app.py")

pytestmark = pytest.mark.slow


def _run(app=None):
    app = app or AppTest.from_file(APP, default_timeout=240)
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    return app


def _selectbox(app, label):
    return next(s for s in app.selectbox if s.label == label)


def test_every_sample_file_exists_and_has_its_outcome():
    for label, (path, outcome, hook, credit) in web_app.SAMPLES.items():
        df = pd.read_csv(web_app.APP_DIR / path)
        assert len(df) > 100, label
        assert hook and credit, label
        if outcome is not None:
            assert outcome in df.columns, label


def test_default_is_penguins_with_species_preselected():
    app = _run()
    assert _selectbox(app, "Choose a dataset").value == "Palmer Penguins"
    assert _selectbox(app, "Focus on an outcome (optional)").value == "species"
    headers = [s.value for s in app.subheader]
    assert "Worth a closer look" in headers
    assert "Drivers of Species" in headers


@pytest.mark.parametrize(
    ("sample", "outcome"),
    [
        ("Titanic", "survived"),
        ("Bike Sharing (daily)", "cnt"),
        ("Synthetic sales", None),
    ],
)
def test_switching_samples_resets_the_outcome(sample, outcome):
    app = _run()
    _selectbox(app, "Choose a dataset").set_value(sample)
    app = _run(app)
    assert _selectbox(app, "Focus on an outcome (optional)").value == outcome
    assert any(sample in c.value for c in app.caption)


def test_outcome_can_be_cleared():
    app = _run()
    # The first option is "None" (no outcome).
    _selectbox(app, "Focus on an outcome (optional)").select_index(0)
    app = _run(app)
    headers = [s.value for s in app.subheader]
    assert "Recommended Visualizations" in headers
    assert not any(h.startswith("Drivers of") for h in headers)


def test_upload_option_waits_for_a_file():
    app = _run()
    _selectbox(app, "Choose a dataset").set_value(web_app.UPLOAD_OPTION)
    app = _run(app)
    assert any("Upload a CSV file to begin." in i.value for i in app.info)
    assert not app.subheader  # nothing analyzed until a file arrives
