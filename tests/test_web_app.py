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
    assert any("Upload a CSV or Excel file to begin." in i.value for i in app.info)
    assert not app.subheader  # nothing analyzed until a file arrives


# --------------------------------------------------
# My board (pinned charts)
# --------------------------------------------------


def _pin_buttons(app):
    return [
        b
        for b in app.button
        if (b.key or "").startswith(("pin::card::", "pin::paradox::"))
    ]


def _board(app):
    return app.session_state["board"]["pins"] if "board" in app.session_state else []


def _click(app, button):
    button.click()
    return _run(app)


def test_pin_and_unpin():
    app = _run()
    first = _pin_buttons(app)[0]
    key = first.key
    app = _click(app, first)
    assert len(_board(app)) == 1
    assert any("1 of 4 pins" in c.value for c in app.caption)
    assert app.button(key=key).label == "Unpin from board"

    app = _click(app, app.button(key=key))
    assert _board(app) == []


def test_at_most_four_pins():
    app = _run()
    for _ in range(5):
        unpinned = [b for b in _pin_buttons(app) if b.label.startswith("📌")]
        app = _click(app, unpinned[0])
    assert len(_board(app)) == 4
    assert any("up to 4 charts" in t.value for t in app.toast)


def test_pins_survive_an_outcome_change():
    app = _run()
    app = _click(app, _pin_buttons(app)[0])
    pinned = [p["chart"].get("x") for p in _board(app)]
    app.selectbox(key="target::Palmer Penguins").select_index(0)
    app = _run(app)
    assert [p["chart"].get("x") for p in _board(app)] == pinned


def test_pins_reset_on_dataset_switch():
    app = _run()
    app = _click(app, _pin_buttons(app)[0])
    assert len(_board(app)) == 1
    _selectbox(app, "Choose a dataset").set_value("Titanic")
    app = _run(app)
    assert _board(app) == []
    assert any("Nothing pinned yet" in i.value for i in app.info)


def test_board_empty_state():
    app = _run()
    assert any("Nothing pinned yet" in i.value for i in app.info)


def test_tab_labels_do_not_change_when_pinning():
    # A changed tab label makes Streamlit rebuild the tabs and jump back to
    # the first one, so unpinning on the board would leave the board.
    app = _run()
    before = [tab.label for tab in app.tabs]
    app = _click(app, _pin_buttons(app)[0])
    assert [tab.label for tab in app.tabs] == before
    assert "My board" in before


# --------------------------------------------------
# Feedback thumbs
# --------------------------------------------------


def test_thumbs_vote_writes_one_row(tmp_path, monkeypatch):
    from visift.feedback import SQLiteFeedbackStore

    database = tmp_path / "votes.sqlite"
    monkeypatch.setenv("VISIFT_FEEDBACK_DB", str(database))

    app = _run()
    thumbs = [w for w in app.get("feedback")]
    assert thumbs, "every recommendation card has thumbs"
    thumbs[0].set_value(1)
    app = _run(app)

    (event,) = SQLiteFeedbackStore(database).events()
    assert event["vote"] == 1 and event["dataset"] == "Palmer Penguins"
    assert event["target"] == "species" and event["rank"] == 1
    assert any("Thanks, this helps" in c.value for c in app.caption)


def test_feedback_off_shows_no_thumbs(monkeypatch):
    # Disabled mode hides the thumbs entirely (rather than showing inert ones).
    monkeypatch.setenv("VISIFT_FEEDBACK", "off")
    app = _run()
    assert len(app.get("feedback")) == 0


def test_paradox_card_shows_its_title_once():
    import json

    app = _run()
    heading = "### Bill Length (mm) vs Bill Depth (mm) by Species"
    assert any(m.value == heading for m in app.markdown)
    specs = [json.loads(chart.proto.spec) for chart in app.get("plotly_chart")]
    titles = [str(s["layout"].get("title", {}).get("text", "")) for s in specs]
    assert not any(heading[4:] in t for t in titles), "figure repeats the heading"
    # The paradox scatter is still drawn (one trace per species), just untitled.
    assert any(
        s["layout"].get("legend", {}).get("title", {}).get("text") == "Species"
        and s["data"][0]["type"] == "scatter"
        for s in specs
    )
    assert any("Average Bill Length (mm)" in t for t in titles)


def test_dataset_tab_shows_the_correlation_overview():
    import json

    app = _run()
    assert any(m.value == "### Correlation overview" for m in app.markdown)
    specs = [json.loads(chart.proto.spec) for chart in app.get("plotly_chart")]
    assert any(s["data"][0]["type"] == "heatmap" for s in specs)
