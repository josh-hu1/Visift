"""
Summarize collected thumbs feedback.

    python -m evaluation.feedback_report [--db PATH]

Prints approval (share of 👍) by chart type and by score band, and how often
each 👎 reason was given. Phase 4 (learning to rank) builds on these events.
"""

import argparse
import os
from collections import Counter

from visift.feedback import DEFAULT_DB, SQLiteFeedbackStore, summarize

SCORE_BANDS = ((0, 50, "<50"), (50, 70, "50–70"), (70, 101, "≥70"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db", default=os.environ.get("VISIFT_FEEDBACK_DB", str(DEFAULT_DB))
    )
    events = SQLiteFeedbackStore(parser.parse_args().db).events()

    if not events:
        print("No feedback recorded yet.")
        return

    print(
        f"{len(events)} votes from {len({e['session_id'] for e in events})} sessions\n"
    )

    print("By chart type")
    for chart, entry in sorted(summarize(events).items()):
        print(
            f"  {chart:<10} 👍 {entry['up']:>3}  👎 {entry['down']:>3}  approval {entry['approval']:.0%}"
        )

    print("\nBy score band")
    for low, high, label in SCORE_BANDS:
        band = [e for e in events if low <= e["score"] < high]
        if band:
            up = sum(e["vote"] == 1 for e in band)
            print(f"  {label:<6} {len(band):>3} votes  approval {up / len(band):.0%}")

    reasons = Counter(e["reason"] or "(no reason)" for e in events if e["vote"] == -1)
    print("\n👎 reasons")
    for reason, count in reasons.most_common() or [("(no 👎 votes)", 0)]:
        print(f"  {reason:<20} {count}")


if __name__ == "__main__":
    main()
