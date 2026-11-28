"""
Thumbs-up / thumbs-down feedback on recommendations.

Each vote is stored as a FeedbackEvent: what was rated (dataset, selected
outcome, chart identity), where it ranked, its score and score components,
the vote and an optional reason. That is enough to evaluate or train a
ranking model later (learning to rank), without storing any data.

Privacy:
- no data values are ever stored;
- for uploaded files the dataset is recorded as "upload" (never the file
  name) and column names are replaced by a token salted per session
  ("col:3fa2b1c9"), so they can't be read back: the salt is random per app
  session and never stored, so a dictionary of common names ("age",
  "income") can't reverse a token, and tokens can't be linked across
  sessions. Within one session a column always gets the same token. The
  built-in samples keep their real column names;
- the session id is a random uuid4 per app session, not an identity.

Storage: SQLite (standard library) at $VISIFT_FEEDBACK_DB, default
.visift/feedback.sqlite. VISIFT_FEEDBACK=off disables it. Each call opens
its own short-lived connection in WAL mode with a short timeout (closed
after every call), so concurrent sessions are safe. A vote is an upsert keyed on (session_id,
dataset, target, chart identity): changing it replaces the row, clearing it
deletes the row. A failure to write is logged and never breaks the app.
"""

import hashlib
import hmac
import logging
import os
import sqlite3
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Protocol, TypedDict

logger = logging.getLogger(__name__)

DEFAULT_DB = Path(".visift/feedback.sqlite")

REASONS = ("Obvious / trivial", "Wrong chart type", "Misleading", "Not interesting")

UPLOAD_DATASET = "upload"

COMPONENTS = ("semantic_fit", "readability", "data_quality", "sample_support", "signal")


class FeedbackEvent(TypedDict):
    timestamp: str
    session_id: str
    dataset: str
    target: str | None
    chart: str
    x: str
    y: str | None
    aggregation: str | None
    group: str | None
    rank: int
    score: float
    semantic_fit: float
    readability: float
    data_quality: float
    sample_support: float
    signal: float
    vote: Literal[1, -1]
    reason: str | None


class ChartSummary(TypedDict):
    up: int
    down: int
    approval: float


class FeedbackStore(Protocol):
    def record(self, event: FeedbackEvent) -> None: ...

    def clear(self, event: FeedbackEvent) -> None: ...

    def events(self) -> list[FeedbackEvent]: ...

    def summary(self) -> dict[str, ChartSummary]: ...


# --------------------------------------------------
# Building events
# --------------------------------------------------


def hash_column(name: str, salt: str) -> str:
    """
    A short stand-in for an uploaded column name: HMAC-SHA256 keyed by a
    per-session salt. The same name and salt always give the same token;
    without the salt (never stored) it can't be guessed or linked.
    """

    digest = hmac.new(salt.encode("utf-8"), str(name).encode("utf-8"), hashlib.sha256)
    return "col:" + digest.hexdigest()[:8]


def make_event(
    *,
    session_id: str,
    salt: str,
    dataset: str,
    uploaded: bool,
    target: str | None,
    candidate: Mapping[str, Any],
    rank: int,
    vote: Literal[1, -1],
    reason: str | None = None,
    now: datetime | None = None,
) -> FeedbackEvent:
    """
    Build an event for a vote. For uploads, the dataset becomes "upload" and
    every column name (x, y, group, target) is replaced by a token salted with
    `salt` (random per session, never stored).
    """

    def column(value: Any) -> str | None:
        if value is None or value == "":
            return None
        return hash_column(value, salt) if uploaded else str(value)

    components = candidate.get("components", {})

    return {
        "timestamp": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
        "session_id": session_id,
        "dataset": UPLOAD_DATASET if uploaded else dataset,
        "target": column(target),
        "chart": str(candidate["chart"]),
        "x": column(candidate["x"]) or "",
        "y": column(candidate.get("y")),
        "aggregation": candidate.get("aggregation"),
        "group": column(candidate.get("group")),
        "rank": int(rank),
        "score": float(candidate["score"]),
        **{name: float(components.get(name, 0.0)) for name in COMPONENTS},  # type: ignore[typeddict-item]
        "vote": vote,
        "reason": reason,
    }


# --------------------------------------------------
# Stores
# --------------------------------------------------

_COLUMNS = tuple(FeedbackEvent.__annotations__)


def _q(name: str) -> str:
    """Quote an identifier ('group' is an SQL keyword)."""

    return f'"{name}"'


_KEY = (
    "session_id",
    "dataset",
    "target_key",
    "chart",
    "x",
    "y_key",
    "aggregation_key",
    "group_key",
)

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS feedback (
    {", ".join(f"{_q(c)} {'INTEGER' if c in ('rank', 'vote') else 'REAL' if c in ('score', *COMPONENTS) else 'TEXT'}" for c in _COLUMNS)},
    target_key TEXT NOT NULL,
    y_key TEXT NOT NULL,
    aggregation_key TEXT NOT NULL,
    group_key TEXT NOT NULL,
    PRIMARY KEY ({", ".join(map(_q, _KEY))})
)
"""


def _key_values(event: FeedbackEvent) -> tuple[str, ...]:
    # SQLite treats NULLs as distinct in a primary key, so key on '' instead.
    return (
        event["session_id"],
        event["dataset"],
        event["target"] or "",
        event["chart"],
        event["x"],
        event["y"] or "",
        event["aggregation"] or "",
        event["group"] or "",
    )


class SQLiteFeedbackStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """A connection that commits on success and is always closed."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=2.0)) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(_SCHEMA)
            with connection:  # commit or roll back
                yield connection

    def record(self, event: FeedbackEvent) -> None:
        try:
            values = [event[c] for c in _COLUMNS]  # type: ignore[literal-required]
            key = _key_values(event)
            columns = (*_COLUMNS, "target_key", "y_key", "aggregation_key", "group_key")
            placeholders = ", ".join("?" for _ in columns)
            updates = ", ".join(f"{_q(c)}=excluded.{_q(c)}" for c in _COLUMNS)
            with self._connect() as connection:
                connection.execute(
                    f"INSERT INTO feedback ({', '.join(map(_q, columns))}) "
                    f"VALUES ({placeholders}) "
                    f"ON CONFLICT ({', '.join(map(_q, _KEY))}) DO UPDATE SET {updates}",
                    (*values, *key[2:3], *key[5:]),
                )
        except Exception:  # never break the app over feedback
            logger.exception("Could not record feedback in %s", self.path)

    def clear(self, event: FeedbackEvent) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    "DELETE FROM feedback WHERE "
                    + " AND ".join(f"{_q(k)}=?" for k in _KEY),
                    _key_values(event),
                )
        except Exception:
            logger.exception("Could not clear feedback in %s", self.path)

    def events(self) -> list[FeedbackEvent]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    f"SELECT {', '.join(map(_q, _COLUMNS))} FROM feedback "
                    "ORDER BY timestamp"
                ).fetchall()
        except Exception:
            logger.exception("Could not read feedback from %s", self.path)
            return []
        return [dict(zip(_COLUMNS, row, strict=True)) for row in rows]  # type: ignore[misc]

    def summary(self) -> dict[str, ChartSummary]:
        return summarize(self.events())


class NullFeedbackStore:
    """Used when feedback is turned off: records nothing."""

    def record(self, event: FeedbackEvent) -> None:
        return None

    def clear(self, event: FeedbackEvent) -> None:
        return None

    def events(self) -> list[FeedbackEvent]:
        return []

    def summary(self) -> dict[str, ChartSummary]:
        return {}


def summarize(events: list[FeedbackEvent]) -> dict[str, ChartSummary]:
    """Vote counts and approval rate (share of 👍) by chart type."""

    result: dict[str, ChartSummary] = {}
    for event in events:
        entry = result.setdefault(event["chart"], {"up": 0, "down": 0, "approval": 0.0})
        entry["up" if event["vote"] == 1 else "down"] += 1
    for entry in result.values():
        entry["approval"] = entry["up"] / (entry["up"] + entry["down"])
    return result


def feedback_enabled() -> bool:
    return os.environ.get("VISIFT_FEEDBACK", "on").strip().lower() not in {
        "off",
        "0",
        "false",
        "no",
    }


def default_store() -> FeedbackStore:
    if not feedback_enabled():
        return NullFeedbackStore()
    return SQLiteFeedbackStore(os.environ.get("VISIFT_FEEDBACK_DB", str(DEFAULT_DB)))
