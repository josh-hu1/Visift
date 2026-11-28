"""
The bundled sample datasets (sources and licenses in samples/README.md).

    from visift.datasets import load_sample
    df = load_sample("penguins")
"""

from importlib.resources import as_file, files
from typing import NamedTuple

import pandas as pd


class Sample(NamedTuple):
    file: str
    outcome: str | None
    description: str


SAMPLES: dict[str, Sample] = {
    "penguins": Sample(
        "penguins.csv",
        "species",
        "Palmer Penguins: a relationship that flips within species",
    ),
    "titanic": Sample("titanic.csv", "survived", "Titanic passengers: who survived?"),
    "bike": Sample(
        "bike_sharing_day.csv", "cnt", "Bike Sharing (daily): what drives rentals?"
    ),
    "bank": Sample(
        "bank_marketing_sample.csv",
        "y",
        "Bank Marketing (5,000 rows): who subscribes to a term deposit?",
    ),
}


def load_sample(name: str) -> pd.DataFrame:
    """Load a bundled sample by short name (see SAMPLES)."""

    if name not in SAMPLES:
        raise KeyError(f"unknown sample {name!r}; choose from {', '.join(SAMPLES)}")

    with as_file(files("visift") / "samples" / SAMPLES[name].file) as path:
        return pd.read_csv(path)
