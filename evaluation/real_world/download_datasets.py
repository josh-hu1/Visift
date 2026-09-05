from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
import zipfile

import pandas as pd


# ==================================================
# CONFIGURATION
# ==================================================

DATASET_DIR = (
    Path(__file__).resolve().parent
    / "datasets"
)

PENGUINS_URL = (
    "https://raw.githubusercontent.com/"
    "allisonhorst/palmerpenguins/main/"
    "inst/extdata/penguins.csv"
)

BIKE_SHARING_ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/275/"
    "bike%2Bsharing%2Bdataset.zip"
)

TITANIC3_URL = (
    "https://hbiostat.org/data/repo/titanic3.csv"
)


# ==================================================
# DOWNLOAD HELPERS
# ==================================================

def download_bytes(
    url
):
    """
    Download raw bytes with a simple user-agent header.
    """

    request = Request(
        url,
        headers={
            "User-Agent": (
                "Visift real-world benchmark"
            )
        }
    )

    with urlopen(
        request,
        timeout=60
    ) as response:

        return response.read()


# ==================================================
# PALMER PENGUINS
# ==================================================

def download_penguins():
    """
    Download the official Palmer Penguins CSV.
    """

    content = download_bytes(
        PENGUINS_URL
    )

    destination = (
        DATASET_DIR
        / "penguins.csv"
    )

    destination.write_bytes(
        content
    )

    return destination


def validate_penguins(
    path
):
    """
    Perform lightweight integrity checks on Palmer Penguins.
    """

    df = pd.read_csv(
        path
    )

    expected_columns = {
        "species",
        "island",
        "bill_length_mm",
        "bill_depth_mm",
        "flipper_length_mm",
        "body_mass_g",
        "sex",
        "year"
    }

    missing_columns = (
        expected_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Penguins dataset is missing expected columns: "
            f"{sorted(missing_columns)}"
        )

    if len(df) != 344:

        raise ValueError(
            "Expected 344 Palmer Penguins rows, "
            f"found {len(df)}."
        )

    return df


# ==================================================
# UCI BIKE SHARING
# ==================================================

def download_bike_sharing():
    """
    Download UCI Bike Sharing and create the benchmark-ready
    daily dataset.

    We intentionally start from the official day.csv file.

    Excluded columns:
        instant
            Record identifier.

        casual
        registered
            cnt is defined as casual + registered, so keeping
            both components would create trivial part-whole
            relationships that dominate an exploratory ranking.

    All other values are left unchanged, including UCI's
    integer-coded categorical fields. That lets the benchmark
    test Visift's semantic inference on realistic CSV input.
    """

    archive_bytes = download_bytes(
        BIKE_SHARING_ZIP_URL
    )

    with zipfile.ZipFile(
        BytesIO(
            archive_bytes
        )
    ) as archive:

        candidates = [
            name
            for name in archive.namelist()
            if (
                name == "day.csv"
                or name.endswith(
                    "/day.csv"
                )
            )
        ]

        if not candidates:

            raise FileNotFoundError(
                "Could not find day.csv inside the "
                "UCI Bike Sharing archive."
            )

        day_member = candidates[0]

        raw_bytes = archive.read(
            day_member
        )

    raw = pd.read_csv(
        BytesIO(
            raw_bytes
        )
    )

    expected_raw_columns = {
        "instant",
        "dteday",
        "season",
        "yr",
        "mnth",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "casual",
        "registered",
        "cnt"
    }

    missing_columns = (
        expected_raw_columns
        - set(raw.columns)
    )

    if missing_columns:

        raise ValueError(
            "Bike Sharing day.csv is missing expected "
            f"columns: {sorted(missing_columns)}"
        )

    if len(raw) != 731:

        raise ValueError(
            "Expected 731 daily Bike Sharing rows, "
            f"found {len(raw)}."
        )

    benchmark_df = (
        raw.drop(
            columns=[
                "instant",
                "casual",
                "registered"
            ]
        )
        .copy()
    )

    destination = (
        DATASET_DIR
        / "bike_sharing_day.csv"
    )

    benchmark_df.to_csv(
        destination,
        index=False
    )

    return (
        destination,
        raw,
        benchmark_df
    )


def validate_bike_sharing(
    path
):
    """
    Validate the benchmark-ready Bike Sharing daily CSV.
    """

    df = pd.read_csv(
        path
    )

    expected_columns = {
        "dteday",
        "season",
        "yr",
        "mnth",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
        "temp",
        "atemp",
        "hum",
        "windspeed",
        "cnt"
    }

    missing_columns = (
        expected_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Benchmark-ready Bike Sharing CSV is missing "
            f"columns: {sorted(missing_columns)}"
        )

    if len(df) != 731:

        raise ValueError(
            "Expected 731 benchmark Bike Sharing rows, "
            f"found {len(df)}."
        )

    return df


# ==================================================
# TITANIC3
# ==================================================

def download_titanic():
    """
    Download Vanderbilt's titanic3 CSV and create a focused
    benchmark view.

    The source dataset has 14 variables. We retain:
        pclass
        survived
        sex
        age
        sibsp
        parch
        fare
        embarked

    We omit high-cardinality identity/text columns and fields
    such as boat/body that are strongly tied to the observed
    outcome and would act like leakage in an exploratory ranking.
    """

    content = download_bytes(
        TITANIC3_URL
    )

    raw = pd.read_csv(
        BytesIO(
            content
        )
    )

    expected_raw_columns = {
        "pclass",
        "survived",
        "name",
        "sex",
        "age",
        "sibsp",
        "parch",
        "ticket",
        "fare",
        "cabin",
        "embarked",
        "boat",
        "body",
        "home.dest"
    }

    missing_columns = (
        expected_raw_columns
        - set(raw.columns)
    )

    if missing_columns:

        raise ValueError(
            "Titanic3 CSV is missing expected columns: "
            f"{sorted(missing_columns)}"
        )

    if len(raw) != 1309:

        raise ValueError(
            "Expected 1309 Titanic3 rows, "
            f"found {len(raw)}."
        )

    retained_columns = [
        "pclass",
        "survived",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked"
    ]

    benchmark_df = (
        raw[
            retained_columns
        ]
        .copy()
    )

    destination = (
        DATASET_DIR
        / "titanic3_benchmark.csv"
    )

    benchmark_df.to_csv(
        destination,
        index=False
    )

    return (
        destination,
        raw,
        benchmark_df
    )


def validate_titanic(
    path
):
    """
    Validate the benchmark-ready Titanic3 CSV.
    """

    df = pd.read_csv(
        path
    )

    expected_columns = {
        "pclass",
        "survived",
        "sex",
        "age",
        "sibsp",
        "parch",
        "fare",
        "embarked"
    }

    missing_columns = (
        expected_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "Benchmark-ready Titanic3 CSV is missing "
            f"columns: {sorted(missing_columns)}"
        )

    if len(df) != 1309:

        raise ValueError(
            "Expected 1309 benchmark Titanic3 rows, "
            f"found {len(df)}."
        )

    return df


# ==================================================
# MAIN
# ==================================================

def main():
    """
    Download every registered real-world benchmark dataset.
    """

    DATASET_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print(
        "VISIFT REAL-WORLD DATASET DOWNLOAD"
    )
    print(
        "=" * 64
    )

    penguins_path = (
        download_penguins()
    )

    penguins = validate_penguins(
        penguins_path
    )

    print(
        f"Downloaded: {penguins_path}"
    )

    print(
        f"Validated Palmer Penguins: "
        f"{len(penguins)} rows, "
        f"{len(penguins.columns)} columns"
    )

    (
        bike_path,
        bike_raw,
        bike_benchmark
    ) = download_bike_sharing()

    bike_validated = (
        validate_bike_sharing(
            bike_path
        )
    )

    print(
        f"Downloaded/prepared: {bike_path}"
    )

    print(
        f"Validated UCI Bike Sharing daily data: "
        f"{len(bike_validated)} rows, "
        f"{len(bike_validated.columns)} benchmark columns "
        f"({len(bike_raw.columns)} raw columns)"
    )

    (
        titanic_path,
        titanic_raw,
        titanic_benchmark
    ) = download_titanic()

    titanic_validated = (
        validate_titanic(
            titanic_path
        )
    )

    print(
        f"Downloaded/prepared: {titanic_path}"
    )

    print(
        f"Validated Titanic3: "
        f"{len(titanic_validated)} rows, "
        f"{len(titanic_validated.columns)} benchmark columns "
        f"({len(titanic_raw.columns)} raw columns)"
    )

    print()
    print(
        "Dataset download complete."
    )
    print()


if __name__ == "__main__":
    main()
