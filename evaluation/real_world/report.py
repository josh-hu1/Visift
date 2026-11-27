"""
Real-world comparison table: baseline vs Visift before Phase 2 vs Visift now.

    python -m evaluation.real_world.benchmark --ranker all   # runs everything
    python -m evaluation.real_world.report                   # table only

Reads the saved dataset_summary.csv files and writes
evaluation/results/real_world_comparison.csv plus Markdown tables.
"""

from pathlib import Path

import pandas as pd

from evaluation.real_world.benchmark import RESULTS_DIRS

METRICS = {
    "recall_at_3": "R@3",
    "recall_at_5": "R@5",
    "recall_at_10": "R@10",
    "mean_reciprocal_expected_rank": "MRR",
}

DATASET_LABELS = {
    "palmer_penguins": "Palmer Penguins",
    "uci_bike_sharing_day": "Bike Sharing",
    "titanic3": "Titanic",
    "uci_bank_marketing": "Bank Marketing",
}

# (column label, ranker, target mode). The Phase 1 engine had no target
# mode, so its plain ranking is the 'before' number in both modes.
SYSTEMS = (
    ("Baseline", "baseline", False),
    ("Visift before Phase 2", "legacy", False),
    ("Visift now", "visift", False),
    ("Baseline (target)", "baseline", True),
    ("Visift before Phase 2 (no target mode)", "legacy", False),
    ("Visift now (target)", "visift", True),
)

OUTPUT = Path("evaluation/results/real_world_comparison.csv")


def load() -> pd.DataFrame:
    rows = []

    for label, ranker, target_mode in SYSTEMS:
        summary = pd.read_csv(
            RESULTS_DIRS[(ranker, target_mode)] / "dataset_summary.csv"
        )
        mode = (
            "target"
            if label.endswith("(target)") or "target mode" in label
            else "no-target"
        )

        for _, row in summary.iterrows():
            rows.append(
                {
                    "mode": mode,
                    "system": label.replace(" (target)", "").replace(
                        " (no target mode)", ""
                    ),
                    "dataset": row["dataset"],
                    "expected_insights": int(row["expected_insights"]),
                    **{metric: float(row[metric]) for metric in METRICS},
                }
            )

        weights = summary["expected_insights"]
        rows.append(
            {
                "mode": mode,
                "system": label.replace(" (target)", "").replace(
                    " (no target mode)", ""
                ),
                "dataset": "all",
                "expected_insights": int(weights.sum()),
                **{
                    metric: float((summary[metric] * weights).sum() / weights.sum())
                    for metric in METRICS
                },
            }
        )

    return pd.DataFrame(rows)


def markdown(table: pd.DataFrame, mode: str) -> str:
    part = table[table["mode"] == mode]
    systems = list(dict.fromkeys(part["system"]))
    header = "| Dataset | " + " | ".join(
        f"{system} {short}" for system in systems for short in METRICS.values()
    )
    lines = [header + " |", "|---|" + "---:|" * (len(systems) * len(METRICS))]

    for dataset in [*DATASET_LABELS, "all"]:
        cells = []
        for system in systems:
            row = part[(part["system"] == system) & (part["dataset"] == dataset)].iloc[
                0
            ]
            for metric in METRICS:
                value = row[metric]
                cells.append(
                    f"{value:.3f}" if metric.startswith("mean") else f"{value:.0%}"
                )
        name = DATASET_LABELS.get(dataset, "**All 20 insights**")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    return "\n".join(lines)


def main() -> None:
    table = load()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT, index=False)

    for mode in ("no-target", "target"):
        print(f"\n{mode.upper()} MODE\n")
        print(markdown(table, mode))

    print(f"\nSaved {OUTPUT}")


if __name__ == "__main__":
    main()
