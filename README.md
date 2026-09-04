# Dataviz Engine

An explainable visualization recommendation engine that automatically profiles a dataset, generates valid visualization candidates, and ranks the charts most likely to reveal useful patterns.

**Live Demo:**  
https://dataviz-engine-hvlp2ydhtkitr6kpmpczzw.streamlit.app/

![Dataviz Engine preview](assets/app-preview.png)

---

## Overview

Most visualization tools help users create charts after they already know what they want to visualize.

Dataviz Engine approaches the problem differently:

> Instead of showing every chart that can be made, it ranks the charts most likely to be informative.

Given a CSV dataset, the engine:

1. Profiles the dataset
2. Infers semantic column types
3. Generates semantically valid visualization candidates
4. Evaluates each candidate for chart suitability
5. Measures chart-specific statistical patterns
6. Calibrates relationship strength for reliability
7. Produces an explainable recommendation score
8. Removes redundant visualizations
9. Renders the highest-ranked recommendations interactively

The result is a ranked set of visualizations with statistical evidence, plain-English insights, and detailed score explanations.

---

## Key Features

- Automatic CSV analysis
- Semantic column type inference
- Visualization candidate generation
- Explainable 0–100 recommendation scoring
- Cross-chart global ranking
- Redundant-chart detection
- Nonlinear relationship detection
- Temporal pattern detection
- Statistical reliability calibration
- Plain-English insight summaries
- Interactive Plotly visualizations
- Streamlit web interface
- Dark neon analytics dashboard theme
- Dataset preview and profiling
- User-controlled chart type and score filters

### Supported Visualizations

Dataviz Engine currently supports:

- Scatterplots
- Line charts
- Bar charts
- Box plots
- Histograms

---

## How It Works

### 1. Dataset Profiling

Each column is analyzed for characteristics including:

- pandas data type
- missing values
- cardinality
- numeric statistics
- common values
- column naming patterns

The engine then assigns a semantic type such as:

- `numeric_continuous`
- `numeric_discrete`
- `categorical`
- `ordinal`
- `boolean`
- `datetime`
- `temporal`
- `geographic`
- `identifier`
- `text`

Semantic inference helps distinguish columns that may share the same pandas dtype but represent very different concepts.

For example:

```text
customer_id  → identifier
date         → datetime
region       → categorical
revenue      → numeric_continuous
rating       → ordinal
subscribed   → boolean
```

---

### 2. Candidate Generation

Once semantic types are known, the engine generates visualizations that are technically and semantically valid.

Examples include:

```text
numeric + numeric
→ scatterplot

datetime + numeric
→ line chart

categorical + numeric
→ bar chart
→ box plot

numeric
→ histogram

categorical
→ count bar chart
```

Candidate generation is intentionally permissive.

It answers:

> Can this visualization reasonably represent these variables?

The scoring system then answers the more important question:

> Is this visualization actually worth recommending?

---

## Recommendation Scoring

Each candidate receives a score between **0 and 100**.

The engine separates two concepts:

### Chart Suitability

How appropriate and reliable is the visualization itself?

Factors include:

- semantic fit
- readability
- data completeness
- sample support

### Pattern Strength

How much potentially useful structure does the visualization reveal?

Pattern strength is calculated differently depending on the chart type.

The final recommendation score combines chart suitability with pattern strength so that a chart cannot rank highly simply because it is technically valid.

Conceptually:

```text
Recommendation Score
        =
Chart Suitability
        ×
Pattern Strength Adjustment
```

This prevents clean but uninformative charts from dominating the recommendation list.

---

## Chart-Specific Pattern Detection

Different visualization types require different definitions of an informative pattern.

### Scatterplots

Scatterplots currently evaluate:

- Pearson correlation
- Spearman correlation
- p-value-based correlation reliability
- mutual information
- permutation-calibrated nonlinear dependence
- paired sample size
- missing data
- semantic suitability

The engine uses:

- **Pearson correlation** for linear relationships
- **Spearman correlation** for monotonic relationships
- **Mutual information** for nonlinear relationships such as quadratic, sinusoidal, and threshold effects

Mutual information is calibrated against shuffled-data null distributions before contributing to the final signal.

Correlation-based signals are also adjusted using statistical reliability so chance relationships in small samples are penalized without automatically suppressing genuinely strong effects.

A linear trendline and R² are rendered in the frontend for scatterplots.

---

### Line Charts

Line charts evaluate multiple forms of temporal structure:

- long-term trend
- seasonality
- level shifts
- volatility shifts
- observation support
- distinct time points
- data completeness
- semantic compatibility

Trend detection uses Pearson and Spearman correlation against time.

Seasonality detection combines frequency-domain analysis with autocorrelation.

Level-shift detection searches for persistent changes in the temporal level.

Volatility-shift detection looks for large changes in robust dispersion across temporal regimes.

The strongest supported temporal pattern determines the line-chart signal rather than adding overlapping signals together.

Dense datetime series are automatically aggregated for readability.

---

### Bar Charts

Grouped bar charts consider:

- number of categories
- observations per category
- missing values
- semantic compatibility
- eta-squared group separation

Count bar charts also evaluate category-frequency imbalance.

---

### Box Plots

Box plots evaluate:

- number of groups
- observations per group
- group separation
- median separation
- within-group outliers
- semantic compatibility

---

### Histograms

Histograms evaluate distribution characteristics including:

- skewness
- potential outliers
- tail asymmetry
- sample size
- number of unique values

Histogram bin counts are automatically estimated using the Freedman-Diaconis rule when possible.

---

## Explainable Recommendations

Each recommendation includes:

- overall score
- recommendation strength
- chart type
- pattern strength
- chart suitability
- plain-English interpretation
- statistical evidence
- detailed component scores
- alternative visualization forms

For example:

```text
Units vs Revenue
Scatterplot

Recommendation Score: 82.5 / 100
Pattern Strength: 73.5 / 100
Chart Suitability: 99.6 / 100

Higher revenue tends to be associated with higher
units sold, with a strong positive relationship.

Pearson correlation: 0.735
Spearman correlation: 0.728
R²: 0.54
```

Temporal recommendations can also explain patterns such as:

```text
Revenue shows a weak upward trend over time,
with a temporal correlation of approximately 0.20.
```

or, when present:

```text
Metric shows a strong repeating seasonal pattern.

Metric shows a strong change in variability across
temporal regimes.

Metric shows a persistent level shift over time.
```

---

## Redundancy Handling

Multiple visualization types may represent the same underlying relationship.

For example:

```text
Revenue by Rating
```

could reasonably be represented as either:

- a box plot
- a mean bar chart

Instead of filling the recommendation list with both, Dataviz Engine keeps the higher-scoring visualization as the primary recommendation and stores the other as an alternative view.

This helps the final recommendation list emphasize distinct insights rather than duplicate variable combinations.

---

## Synthetic Benchmarking

Dataviz Engine includes a controlled synthetic benchmark framework used to evaluate ranking behavior and false positives.

The current hardened benchmark contains:

- **125 synthetic datasets**
- **95 positive datasets** with known planted patterns
- **30 null/adversarial datasets**
- **1,850 visualization candidates**

The benchmark includes:

- strong and weak linear relationships
- nonlinear quadratic relationships
- noisy nonlinear relationships
- sinusoidal relationships
- threshold effects
- categorical group effects
- category imbalance
- temporal trends
- seasonality
- change points
- trend + seasonality
- volatility shifts
- skewed distributions
- outlier-heavy distributions
- small sample sizes
- 40% missingness
- independent skewed variables
- independent variables with outliers
- random categorical groups
- temporal white noise
- small-sample random correlations

### Current Hardened Benchmark Results

| Metric | Result |
|---|---:|
| Positive datasets | 95 |
| Null datasets | 30 |
| Visualization candidates | 1,850 |
| Top-1 retrieval | 100.0% |
| Top-3 retrieval | 100.0% |
| Top-5 retrieval | 100.0% |
| Mean reciprocal rank | 1.000 |
| Highest evaluated null score | 43.25 |
| Null candidates scoring ≥ 50 | 0.0% |
| Null candidates scoring ≥ 65 | 0.0% |

These results come from a **controlled synthetic benchmark**, not from arbitrary real-world datasets.

A precise interpretation is:

> On the current 125-dataset synthetic benchmark, Dataviz Engine ranked the planted visualization first in all 95 positive scenarios, while no evaluated null candidate scored 50 or higher.

The benchmark is used primarily for regression testing and model-development comparisons rather than as a claim of universal real-world accuracy.

---

## Why the Benchmark Matters

The benchmark has directly influenced the engine's development.

### Nonlinear Relationship Detection

The original engine performed well on linear and monotonic relationships but struggled with quadratic relationships because Pearson and Spearman correlation can both be close to zero for strong U-shaped patterns.

Adding permutation-calibrated mutual information improved nonlinear retrieval while preserving low scores on independent noise.

### Temporal Pattern Detection

The original line scorer primarily measured monotonic trend strength.

Harder benchmarks exposed failures on:

- pure seasonality
- volatility regime shifts

The temporal scorer was expanded to detect:

- trend
- seasonality
- level shifts
- volatility shifts

### Small-Sample Reliability

Adversarial tests showed that random correlations with only 30 observations could occasionally receive moderate scores.

Correlation signals are now adjusted using p-value-based reliability, reducing small-sample false positives while preserving genuinely strong small-sample relationships.

---

## Architecture

```text
                    CSV Dataset
                         │
                         ▼
                ┌─────────────────┐
                │ Data Profiling  │
                └────────┬────────┘
                         │
                         ▼
             ┌────────────────────────┐
             │ Semantic Type Inference│
             └───────────┬────────────┘
                         │
                         ▼
             ┌───────────────────────┐
             │ Candidate Generation  │
             └───────────┬───────────┘
                         │
                         ▼
        ┌─────────────────────────────────┐
        │     Chart-Specific Scoring      │
        │                                 │
        │  Bar       Scatter      Line    │
        │  Box       Histogram            │
        └────────────────┬────────────────┘
                         │
                         ▼
              ┌────────────────────┐
              │ Global Ranking     │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │ Diversification    │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │ Plotly Rendering   │
              └─────────┬──────────┘
                        │
                        ▼
              ┌────────────────────┐
              │ Streamlit UI       │
              └────────────────────┘
```

---

## Project Structure

```text
Dataviz-Engine/
│
├── app.py
├── web_app.py
├── profiler.py
├── semantic_types.py
├── candidate_generator.py
├── generate_test_data.py
├── requirements.txt
├── synthetic_sales.csv
│
├── scoring/
│   ├── __init__.py
│   ├── engine.py
│   ├── bar.py
│   ├── scatter.py
│   ├── line.py
│   ├── histogram.py
│   └── box.py
│
├── visualization/
│   ├── __init__.py
│   └── renderer.py
│
├── evaluation/
│   ├── __init__.py
│   ├── datasets.py
│   ├── metrics.py
│   ├── benchmark.py
│   └── results/
│
├── assets/
│   └── app-preview.png
│
└── .streamlit/
    └── config.toml
```

---

## Tech Stack

### Data Analysis

- Python
- pandas
- NumPy
- SciPy
- scikit-learn

### Visualization

- Plotly

### Frontend

- Streamlit

### Testing / Evaluation

- Synthetic data generation
- Controlled benchmark scenarios
- Top-K retrieval metrics
- Mean reciprocal rank
- False-positive evaluation

### Version Control / Deployment

- Git
- GitHub
- Streamlit Community Cloud

---

## Dark Analytics Interface

The frontend uses a custom dark theme designed around:

- black backgrounds
- dark panels
- neon-green data marks
- neon-green selected states
- muted pale-green secondary text
- high-contrast interactive Plotly charts

The visual styling is intentionally designed so that bright green is primarily used for **data and important accents** rather than large interface surfaces.

---

## Running Locally

Clone the repository:

```bash
git clone https://github.com/josh-hu1/Dataviz-Engine.git
cd Dataviz-Engine
```

Create a virtual environment if desired.

For Conda:

```bash
conda create -n dataviz python=3.12
conda activate dataviz
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
streamlit run web_app.py
```

Then open the local Streamlit address shown in the terminal.

---

## Running the Benchmark

Run the synthetic benchmark from the repository root:

```bash
python -m evaluation.benchmark
```

The benchmark evaluates positive relationship retrieval and false-positive behavior across the full scenario suite.

Results are written to the `evaluation/results/` directory.

---

## Example Workflow

1. Open Dataviz Engine
2. Select the sample dataset or upload a CSV
3. The dataset is automatically profiled
4. Semantic types are inferred
5. Valid visualization candidates are generated
6. Every candidate is scored
7. Statistical reliability is incorporated
8. Redundant recommendations are consolidated
9. The highest-ranked charts are rendered interactively
10. Recommendation evidence can be inspected through the interface

No chart selection is required beforehand.

---

## Design Principles

### Candidate Generation and Recommendation Are Separate Problems

A visualization can be technically valid without being useful.

Dataviz Engine therefore keeps candidate generation broad while allowing the ranking system to penalize weak candidates.

### Effect Size Is Not the Same as Reliability

An apparent relationship based on a very small sample should not automatically receive a high recommendation score.

The engine incorporates sample support, permutation calibration, and p-value-based reliability where appropriate.

### Scores Should Be Explainable

The engine exposes component scores and statistical evidence rather than producing an unexplained ranking.

### Different Charts Require Different Definitions of Insight

A meaningful scatterplot is not evaluated the same way as a meaningful histogram or line chart.

Each visualization type therefore has its own chart-specific signal calculation.

### Benchmark Before Tuning

Scoring changes are evaluated against controlled benchmark scenarios before additional tuning.

This helps prevent improvements on one dataset from silently creating regressions elsewhere.

---

## Current Limitations

The project is still under active development.

Potential improvements include:

- score calibration across weak, moderate, and strong effects
- broader real-world dataset testing
- irregular time-series handling
- additional semantic-type robustness
- currency and percentage string parsing
- high-cardinality category handling
- geographic visualization support
- confidence intervals
- additional statistical tests
- visualization accessibility checks
- additional chart types
- Excel support
- database support
- natural-language dataset querying
- automated dashboard generation
- learned ranking models based on user feedback

---

## Current Development Roadmap

### Completed

- Core profiling and semantic inference
- Candidate generation
- Explainable global ranking
- Interactive Streamlit frontend
- Synthetic benchmark framework
- Nonlinear scatter detection
- Temporal seasonality detection
- Temporal level-shift detection
- Temporal volatility detection
- Small-sample correlation reliability calibration
- Dark neon dashboard redesign

### Next

- Graduated score calibration
- Harder adversarial benchmark scenarios
- Real-world messy-data robustness
- Additional semantic inference improvements

### Longer Term

- More chart types
- Natural-language querying
- Automated dashboard generation
- Learned visualization ranking from user feedback

---

## Future Direction

A longer-term version of Dataviz Engine could combine statistical analysis with machine learning to learn which visualizations users actually find useful.

The recommendation engine could eventually incorporate:

```text
Dataset structure
        +
Statistical patterns
        +
Visualization best practices
        +
User interaction data
        ↓
Personalized visualization ranking
```

This would allow the system to move beyond rule-based recommendation toward learned visualization discovery.

---

## Author

**Josh Hu**  
Data Science — University of California, San Diego

GitHub:  
https://github.com/josh-hu1
