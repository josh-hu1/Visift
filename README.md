# Dataviz Engine

An explainable visualization recommendation engine that automatically profiles a dataset, generates valid chart candidates, and ranks the visualizations most likely to reveal useful patterns.

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
6. Produces an explainable recommendation score
7. Removes redundant visualizations
8. Renders the highest-ranked recommendations interactively

The result is a ranked set of visualizations with both statistical evidence and a plain-English explanation of why each chart may be useful.

---

## Features

- Automatic CSV analysis
- Semantic column type inference
- Visualization candidate generation
- Explainable 0–100 recommendation scoring
- Cross-chart ranking
- Redundant-chart detection
- Plain-English insight summaries
- Interactive Plotly visualizations
- Streamlit web interface
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

Semantic inference helps distinguish columns that may share the same underlying pandas dtype but have very different meanings.

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

This is calculated differently depending on the visualization type.

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

## Chart-Specific Signals

Different visualization types require different definitions of an informative pattern.

### Scatterplots

Scatterplots currently consider:

- Pearson correlation
- Spearman correlation
- paired sample size
- missing data
- variable suitability

A linear regression trendline and R² are also rendered in the frontend.

### Line Charts

Line charts evaluate:

- temporal + numeric semantic compatibility
- number of observations
- distinct time points
- missing values
- Pearson correlation with time
- Spearman correlation with time
- trend direction

Dense time series are automatically aggregated for visualization readability.

### Bar Charts

Grouped bar charts consider:

- number of categories
- observations per category
- missing values
- semantic compatibility
- eta-squared group separation

Count bar charts also evaluate category-frequency imbalance.

### Box Plots

Box plots evaluate:

- number of groups
- observations per group
- group separation
- median separation
- within-group outliers
- semantic compatibility

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

## Synthetic Validation

A synthetic dataset with 500 observations was created to validate the recommendation engine.

Known relationships were intentionally planted into the data.

Examples:

```text
ad_spend → revenue
strong relationship

revenue → units
strong relationship

date → revenue
upward temporal trend

region → revenue
moderate grouped effect

department → profit
little meaningful relationship
```

The engine generated **50 visualization candidates** across five chart types.

The highest-ranked scatterplots were:

| Rank | Visualization | Score | Pearson |
|---|---|---:|---:|
| 1 | Revenue vs Units | 82.45 | 0.735 |
| 2 | Ad Spend vs Revenue | 81.36 | 0.718 |
| 3 | Ad Spend vs Units | 70.67 | 0.549 |
| 4 | Revenue vs Profit | 50.97 | 0.249 |

This matched the relationships intentionally built into the synthetic data.

Weak or unrelated combinations received substantially lower scores.

For example, a technically valid category-versus-numeric visualization with almost no group separation was prevented from receiving a high recommendation score solely because it had clean data and large sample sizes.

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

### Visualization

- Plotly

### Frontend

- Streamlit

### Version Control / Deployment

- Git
- GitHub
- Streamlit Community Cloud

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

## Example Workflow

1. Open Dataviz Engine
2. Select the sample dataset or upload a CSV
3. The dataset is automatically profiled
4. Semantic types are inferred
5. Valid visualization candidates are generated
6. Every candidate is scored
7. Redundant recommendations are consolidated
8. The highest-ranked charts are rendered interactively
9. Recommendation evidence can be inspected through the interface

No chart selection is required beforehand.

---

## Design Principles

### Candidate Generation and Recommendation Are Separate Problems

A visualization can be technically valid without being useful.

Dataviz Engine therefore keeps candidate generation broad while allowing the ranking system to penalize weak candidates.

### Effect Size Is Not the Same as Confidence

An apparent relationship based on a very small sample should not automatically receive a high recommendation score.

Several scorers therefore adjust statistical signals using sample support.

### Scores Should Be Explainable

The engine exposes its component scores and statistical evidence rather than producing an unexplained ranking.

### Different Charts Require Different Definitions of Insight

A meaningful scatterplot is not evaluated the same way as a meaningful histogram.

Each visualization type therefore has its own chart-specific signal calculation.

---

## Current Limitations

The project is currently an early recommendation engine and has several areas for future improvement.

Potential improvements include:

- nonlinear relationship detection
- mutual information
- clustering and anomaly detection
- seasonality detection
- change-point detection
- more sophisticated time-series aggregation
- confidence intervals
- statistical significance testing
- visualization accessibility checks
- additional chart types
- Excel support
- database support
- natural-language dataset querying
- automated dashboard generation
- learned ranking models based on user feedback

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
