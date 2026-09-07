from pathlib import Path
import numpy as np
import pandas as pd

from candidate_generator import generate_candidates
from semantic_types import classify_column

RESULTS_DIR = Path("evaluation/results/time_semantics_calibration_v1")

def make_profile(df):
    return {
        "column_profiles": [
            {"name": c, "semantic_type": classify_column(df[c])}
            for c in df.columns
        ]
    }

def has_line(candidates, x, y="metric"):
    return any(
        c.get("chart") == "line"
        and c.get("x") == x
        and c.get("y") == y
        for c in candidates
    )

def scenario_df(name, values, repeats=20):
    values = list(values)
    x = np.tile(values, repeats)
    n = len(x)
    return pd.DataFrame({
        name: x,
        "metric": np.linspace(10.0, 100.0, n) + np.sin(np.arange(n) / 5.0)
    })

def build_scenarios():
    dates = pd.date_range("2024-01-01", periods=120, freq="D")
    return [
        ("parsed_datetime","true_chronology","date",
         pd.DataFrame({"date": dates, "metric": np.linspace(0,1,len(dates))}), True,
         "Already-parsed datetime."),
        ("date_string_iso","true_chronology","date",
         pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "metric": np.linspace(0,1,len(dates))}), True,
         "ISO date strings."),
        ("calendar_year","true_chronology","year",
         scenario_df("year", range(2010,2025), 15), True,
         "Year-like ordered chronology."),
        ("month_number","cyclical_calendar","month",
         scenario_df("month", range(1,13), 40), False,
         "Month-of-year repeats every year."),
        ("month_names","cyclical_calendar","month",
         scenario_df("month", ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"], 40), False,
         "Text month labels."),
        ("day_of_month","cyclical_calendar","day",
         scenario_df("day", range(1,32), 20), False,
         "Day-of-month is cyclical, not standalone chronology."),
        ("weekday_number","cyclical_calendar","weekday",
         scenario_df("weekday", range(0,7), 80), False,
         "Day-of-week repeats every week."),
        ("weekday_names","cyclical_calendar","weekday",
         scenario_df("weekday", ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"], 80), False,
         "Text weekday labels."),
        ("hour_of_day","cyclical_calendar","hour",
         scenario_df("hour", range(0,24), 30), False,
         "Hour-of-day repeats every day."),
        ("quarter_number","cyclical_calendar","quarter",
         scenario_df("quarter", range(1,5), 100), False,
         "Quarter-of-year repeats every year."),
        ("week_of_year","cyclical_calendar","week",
         scenario_df("week", range(1,53), 10), False,
         "Week-of-year repeats every year."),
        ("generic_sequence","ambiguous_numeric","sequence",
         scenario_df("sequence", range(1,21), 20), False,
         "Ordered integers without time semantics."),
        ("period_code","ambiguous_numeric","period",
         scenario_df("period", range(1,13), 30), False,
         "Ambiguous period code.")
    ]

def run_calibration():
    rows = []
    for scenario, family, x, df, desired_line, notes in build_scenarios():
        profile = make_profile(df)
        inferred = next(
            col["semantic_type"]
            for col in profile["column_profiles"]
            if col["name"] == x
        )
        candidates = generate_candidates(df, profile)
        line_generated = has_line(candidates, x)
        rows.append({
            "scenario": scenario,
            "family": family,
            "column": x,
            "inferred_type": inferred,
            "line_generated": line_generated,
            "desired_line": desired_line,
            "matches_desired_behavior": line_generated == desired_line,
            "unique_values": int(df[x].nunique(dropna=True)),
            "rows": len(df),
            "notes": notes
        })

    results = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_DIR / "time_semantics_scenarios.csv", index=False)

    summary = (
        results.groupby("family", as_index=False)
        .agg(
            scenarios=("scenario","size"),
            desired_matches=("matches_desired_behavior","sum"),
            line_candidates=("line_generated","sum")
        )
    )
    summary["desired_match_pct"] = 100 * summary["desired_matches"] / summary["scenarios"]
    summary.to_csv(RESULTS_DIR / "family_summary.csv", index=False)

    print("\n" + "="*108)
    print("VISIFT TIME-SEMANTICS CALIBRATION")
    print("="*108)
    print("\nSCENARIO RESULTS\n" + "-"*108)
    print(results[
        ["scenario","family","column","inferred_type","line_generated","desired_line","matches_desired_behavior"]
    ].to_string(index=False))

    print("\nFAMILY SUMMARY\n" + "-"*108)
    p = summary.copy()
    p["desired_match_pct"] = p["desired_match_pct"].map(lambda v: f"{v:.1f}%")
    print(p.to_string(index=False))

    print("\nMISMATCHES REQUIRING REVIEW\n" + "-"*108)
    mismatches = results[~results["matches_desired_behavior"]]
    if mismatches.empty:
        print("None.")
    else:
        print(mismatches[
            ["scenario","column","inferred_type","line_generated","desired_line","notes"]
        ].to_string(index=False))

    print("\nDECISION CRITERIA\n" + "-"*108)
    print("A future fix should preserve real datetimes and validated years,")
    print("while preventing standalone month/day/weekday/hour/quarter/week fields")
    print("from automatically receiving ordinary chronological line charts.")
    print("The frozen line scorer should remain unchanged unless a separate bug is found.")
    print("\n" + "="*108)
    print(f"Results saved to {RESULTS_DIR}/")
    print("="*108 + "\n")

if __name__ == "__main__":
    run_calibration()
