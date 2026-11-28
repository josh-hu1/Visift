"""
Visift: rank which charts are worth looking at first in a new dataset.

    import visift
    recs = visift.recommend(df, target="churned")
    recs[0].figure()
    visift.report(df, "report.html", target="churned")

Command line: visift analyze data.csv --target churned --html report.html
"""

__version__ = "0.1.0"

from visift.api import Recommendation, Recommendations, recommend, report  # noqa: E402
from visift.datasets import load_sample  # noqa: E402
from visift.pipeline import analyze  # noqa: E402

__all__ = [
    "Recommendation",
    "Recommendations",
    "__version__",
    "analyze",
    "load_sample",
    "recommend",
    "report",
]
