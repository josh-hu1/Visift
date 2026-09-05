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

### Titanic3

Domain: passenger survival

Source:
https://hbiostat.org/data/

Vanderbilt University Department of Biostatistics grants permission to use the datasets provided on its data page and requests attribution to the original source plus acknowledgement that the data were obtained from hbiostat.org/data courtesy of Vanderbilt Biostatistics.

The `titanic3` dataset contains 1309 passenger observations. It was compiled and interpreted by Thomas Cason from Titanic passenger-list sources described on the Vanderbilt/Harrell documentation pages.

For the Visift benchmark, we retain only:

- `pclass`
- `survived`
- `sex`
- `age`
- `sibsp`
- `parch`
- `fare`
- `embarked`

We omit name, ticket, cabin, home destination, boat, and body. Those fields are high-cardinality identity/text features or are closely tied to the observed survival outcome and would make the exploratory ranking less representative.

Titanic is deliberately useful for testing the distinction between:

- numeric-coded categories (`pclass`)
- binary indicators (`survived`)
- real discrete counts (`sibsp`, `parch`)
- continuous measurements (`age`, `fare`)
- ordinary string categories (`sex`, `embarked`)

The pre-registered survival-by-sex and survival-by-class expectations also test whether Visift can represent useful relationships between categorical/binary variables. If those are not generated, that is treated as a product capability gap rather than silently removed from the benchmark.


### UCI Bank Marketing

Domain: business / direct marketing

Source:
https://archive.ics.uci.edu/dataset/222/bank+marketing

License: CC BY 4.0

Citation:
Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306.

The dataset contains 45,211 observations from direct-marketing phone campaigns by a Portuguese banking institution. The binary target `y` records whether a client subscribed to a term deposit.

For the Visift benchmark, we use UCI `bank-full.csv` and remove only:

- `duration` — last contact duration. It is only known after the call and the dataset documentation warns that it strongly reveals the output target.

The retained benchmark columns are:

- `age`
- `job`
- `marital`
- `education`
- `default`
- `balance`
- `housing`
- `loan`
- `contact`
- `day`
- `month`
- `campaign`
- `pdays`
- `previous`
- `poutcome`
- `y`

Bank Marketing broadens the suite into a business dataset and tests:

- yes/no text values being recognized as booleans
- category-to-binary subscription-rate recommendations
- higher-cardinality categorical variables such as job
- numeric financial measures such as balance
- count-like integer fields such as campaign and previous
- calendar fields represented as day numbers and month abbreviations
- prior-campaign outcome as a potentially informative categorical relationship

The pre-registered expected insights are defined before inspecting Visift's ranking.

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

Expected insights may optionally specify:

- required x-variable
- required y-variable
- required aggregation

Those constraints prevent a reversed visualization from being counted as the intended insight when orientation changes the meaning.

## Methodology rules

Expected insights and semantic types should be defined in `expectations.py` **before** inspecting a new dataset's Visift ranking.

Do not add an expectation merely because Visift happened to rank it highly. This keeps evaluation from becoming circular.

Real-world expectations are intentionally incomplete. An unmatched recommendation is a review item, not an automatic false positive.
