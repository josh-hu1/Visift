import json
import logging

import pytest

from visift.feedback import (
    NullFeedbackStore,
    SQLiteFeedbackStore,
    default_store,
    hash_column,
    make_event,
    summarize,
)

CANDIDATE = {
    "chart": "scatter",
    "x": "bill_length_mm",
    "y": "bill_depth_mm",
    "score": 72.3,
    "components": {
        "semantic_fit": 100.0,
        "readability": 98.0,
        "data_quality": 99.4,
        "sample_support": 100.0,
        "signal": 56.1,
        "visualization_quality": 99.5,
    },
}


def _event(vote=1, reason=None, candidate=CANDIDATE, **overrides):
    arguments = {
        "session_id": "s1",
        "dataset": "Palmer Penguins",
        "uploaded": False,
        "target": "species",
        "candidate": candidate,
        "rank": 3,
        "vote": vote,
        "reason": reason,
    }
    arguments.update(overrides)
    return make_event(**arguments)


@pytest.fixture
def store(tmp_path):
    return SQLiteFeedbackStore(tmp_path / "feedback.sqlite")


def test_record_and_read_back(store):
    store.record(_event())
    (event,) = store.events()
    assert event["chart"] == "scatter" and event["x"] == "bill_length_mm"
    assert event["vote"] == 1 and event["rank"] == 3 and event["score"] == 72.3
    assert event["signal"] == 56.1 and event["target"] == "species"


def test_changing_a_vote_replaces_the_row(store):
    store.record(_event(vote=1))
    store.record(_event(vote=-1, reason="Obvious / trivial"))
    (event,) = store.events()
    assert event["vote"] == -1 and event["reason"] == "Obvious / trivial"


def test_other_sessions_and_outcomes_are_separate_rows(store):
    store.record(_event())
    store.record(_event(session_id="s2"))
    store.record(_event(target=None))
    assert len(store.events()) == 3


def test_clearing_deletes_the_row(store):
    store.record(_event())
    store.clear(_event())
    assert store.events() == []


def test_summary_by_chart_type(store):
    store.record(_event(vote=1))
    store.record(_event(vote=1, session_id="s2"))
    store.record(_event(vote=-1, session_id="s3"))
    box = {**CANDIDATE, "chart": "box", "x": "species", "y": "body_mass_g"}
    store.record(_event(vote=-1, candidate=box))
    summary = store.summary()
    assert summary["scatter"] == {"up": 2, "down": 1, "approval": pytest.approx(2 / 3)}
    assert summary["box"] == {"up": 0, "down": 1, "approval": 0.0}
    assert summarize([]) == {}


def test_uploads_hash_column_names_and_hide_the_file_name(store):
    event = _event(uploaded=True, dataset="secret_customers.csv", target="churned")
    store.record(event)
    (saved,) = store.events()
    assert saved["dataset"] == "upload"
    assert saved["x"] == hash_column("bill_length_mm") and saved["x"].startswith("col:")
    assert saved["target"] == hash_column("churned")
    stored = json.dumps(saved)
    for secret in ("secret_customers", "churned", "bill_length_mm"):
        assert secret not in stored


def test_disabled_store_records_nothing(monkeypatch, tmp_path):
    monkeypatch.setenv("VISIFT_FEEDBACK", "off")
    monkeypatch.setenv("VISIFT_FEEDBACK_DB", str(tmp_path / "x.sqlite"))
    store = default_store()
    assert isinstance(store, NullFeedbackStore)
    store.record(_event())
    assert store.events() == [] and store.summary() == {}
    assert not (tmp_path / "x.sqlite").exists()


def test_write_failure_is_logged_not_raised(tmp_path, caplog):
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("file in the way")
    broken = SQLiteFeedbackStore(blocker / "feedback.sqlite")
    with caplog.at_level(logging.ERROR):
        broken.record(_event())
        broken.clear(_event())
        assert broken.events() == []
    assert "Could not record feedback" in caplog.text
