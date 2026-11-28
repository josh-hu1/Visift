"""The visift command line (visift.cli.main called in-process)."""

import json

import numpy as np
import pandas as pd
import pytest

import visift
from visift.cli import main


def _csv() -> str:
    # Real but imperfect relationships (nothing near-duplicate or determined,
    # which Visift would rightly move to the dataset notes).
    rng = np.random.default_rng(7)
    n = 200
    region = rng.choice(["North", "South", "East"], n)
    units = rng.integers(1, 100, n)
    revenue = units * 9.5 + rng.normal(0, 150, n)
    churned = (rng.random(n) < np.where(region == "North", 0.6, 0.15)).astype(int)
    frame = pd.DataFrame(
        {
            "region": region,
            "units": units,
            "revenue": revenue.round(2),
            "churned": churned,
        }
    )
    return frame.to_csv(index=False)


@pytest.fixture
def table(tmp_path):
    path = tmp_path / "sales.csv"
    path.write_text(_csv())
    return path


def test_version(capsys):
    assert main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == f"visift {visift.__version__}"


def test_analyze_prints_a_ranked_table(table, capsys):
    assert main(["analyze", str(table), "--top", "3"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("sales.csv: 200 rows, 4 columns")
    assert "  1 " in out and "  3 " in out and "  4 " not in out
    assert "Excellent" in out or "Strong" in out


def test_target_marks_drivers_first(table, capsys):
    assert main(["analyze", str(table), "--target", "churned", "--top", "4"]) == 0
    out = capsys.readouterr().out
    assert "Outcome: churned" in out
    assert "  1*" in out
    # Takeaways are plain text: currency without a Markdown escape.
    assert "$5" in out and "\\$" not in out


def test_unknown_target_is_a_usage_error(table, capsys):
    assert main(["analyze", str(table), "--target", "nope"]) == 2
    assert "no column named 'nope'" in capsys.readouterr().err


def test_html_and_json_outputs(table, tmp_path, capsys):
    html = tmp_path / "report.html"
    data = tmp_path / "recs.json"
    code = main(
        ["analyze", str(table), "--top", "3", "--html", str(html), "--json", str(data)]
    )
    assert code == 0
    page = html.read_text()
    assert page.startswith("<!DOCTYPE html>") or page.lower().startswith(
        "<!doctype html"
    )
    assert "sales.csv" in page
    payload = json.loads(data.read_text())
    assert payload["source"] == "sales.csv" and payload["target"] is None
    (first, *_rest) = payload["recommendations"]
    assert len(payload["recommendations"]) == 3
    assert {"rank", "score", "band", "takeaway", "evidence", "breakdown"} <= set(first)
    assert first["breakdown"]["score"] == pytest.approx(first["score"], abs=0.01)


def test_json_to_stdout_is_pure_json(table, capsys):
    assert main(["analyze", str(table), "--top", "2", "--json", "-"]) == 0
    assert len(json.loads(capsys.readouterr().out)["recommendations"]) == 2


def test_unreadable_file_exits_1_without_traceback(tmp_path, capsys):
    bad = tmp_path / "picture.png"
    bad.write_bytes(b"\x89PNG not a table")
    assert main(["analyze", str(bad)]) == 1
    err = capsys.readouterr().err
    assert "Unsupported file type" in err and "hint:" in err
    assert "Traceback" not in err


def test_missing_file_exits_1(tmp_path, capsys):
    assert main(["analyze", str(tmp_path / "nope.csv")]) == 1
    assert "Could not read" in capsys.readouterr().err


def test_usage_errors_exit_2(capsys):
    assert main([]) == 2
    assert main(["analyze"]) == 2
    assert main(["analyze", "x.csv", "--top", "0"]) == 2


@pytest.mark.slow
def test_bundled_sample_uses_its_outcome(capsys):
    assert main(["analyze", "--sample", "penguins", "--top", "3"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("penguins.csv: 344 rows")
    assert "Outcome: species" in out
    assert "Simpson's paradox" in out
