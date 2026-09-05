# Visift Real-World Evaluation

This directory evaluates Visift on public, real-world datasets.

It is intentionally separate from the synthetic calibration benchmark:

- **Synthetic calibration** has planted ground truth and can measure false positives directly.
- **Real-world evaluation** has incomplete human annotations, so it measures whether pre-registered useful insights are surfaced near the top of the ranking.

An unannotated real-world recommendation is **not** automatically a false positive. High-ranking unmatched recommendations are written to a review queue for human inspection.

## Current datasets

### Palmer Penguins

Domain: ecology

Source:
https://github.com/allisonhorst/palmerpenguins

License: CC0

Citation:
Gorman KB, Williams TD, Fraser WR (2014). *Ecological Sexual Dimorphism and Environmental Variability within a Community of Antarctic Penguins (Genus Pygoscelis).* PLOS ONE 9(3): e90081.

### UCI Bike Sharing — daily data

Domain: transportation / demand

Source:
https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset

License: CC BY 4.0

Citation:
Fanaee-T, H. (2013). *Bike Sharing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894.

The benchmark uses UCI `day.csv`, but removes:

- `instant` — record identifier.
- `casual` and `registered` — `cnt` is defined as their sum, so retaining them would create trivial part-whole relationships that dominate exploratory ranking.

All other values are left unchanged, including integer-coded categorical fields. This intentionally tests Visift's semantic inference on realistic CSV input.

## Run

From the repository root:

```bash
python -m evaluation.real_world.download_datasets
python -m evaluation.real_world.benchmark
```

Results are written to:

```text
evaluation/results/real_world_v1/
```

The benchmark tracks:

- Recall@3
- Recall@5
- Recall@10
- mean reciprocal expected-insight rank
- individual expected-insight ranks
- pre-registered semantic-type accuracy
- inferred semantic column types
- top unmatched recommendations for manual review
- per-dataset runtime
- a direct cross-dataset comparison table

## Methodology rules

Expected insights and semantic types should be defined in `expectations.py` **before** inspecting a new dataset's Visift ranking.

Do not add an expectation merely because Visift happened to rank it highly. This keeps evaluation from becoming circular.

Real-world expectations are intentionally incomplete. An unmatched recommendation is a review item, not an automatic false positive.
