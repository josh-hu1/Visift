from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ==================================================
# DATA STRUCTURES
# ==================================================

@dataclass(frozen=True)
class ExpectedInsight:
    """
    One human-curated insight that Visift should ideally
    surface somewhere near the top of the recommendation list.

    Real-world evaluation is deliberately insight-centric
    rather than exact-chart-centric. Several chart forms can
    communicate the same underlying relationship.

    required_x / required_y are optional orientation constraints.
    They are useful when reversing x and y would change the
    meaning of the expected insight, such as a mean bar showing
    survival rate by passenger class.
    """

    name: str
    variables: tuple[str, ...]
    acceptable_charts: tuple[str, ...]
    rationale: str
    aggregation: Optional[str] = None
    required_x: Optional[str] = None
    required_y: Optional[str] = None


@dataclass(frozen=True)
class ExpectedSemanticType:
    """
    Human-reviewed semantic expectation for one column.

    Multiple acceptable types are allowed when the distinction
    is genuinely debatable within Visift's taxonomy.
    """

    column: str
    acceptable_types: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class RealWorldDataset:
    """
    Metadata and pre-registered expectations for one real
    dataset used by the Visift real-world benchmark.
    """

    name: str
    filename: str
    domain: str
    source: str
    license_name: str
    citation: str
    description: str
    expected_insights: tuple[ExpectedInsight, ...]
    expected_semantics: tuple[ExpectedSemanticType, ...]
    preprocessing_notes: str = ""


# ==================================================
# DATASET DIRECTORY
# ==================================================

DATASET_DIR = (
    Path(__file__).resolve().parent
    / "datasets"
)


# ==================================================
# PALMER PENGUINS
# ==================================================

PENGUINS = RealWorldDataset(
    name="palmer_penguins",
    filename="penguins.csv",
    domain="ecology",
    source=(
        "https://github.com/"
        "allisonhorst/palmerpenguins"
    ),
    license_name="CC0",
    citation=(
        "Gorman KB, Williams TD, Fraser WR (2014). "
        "Ecological Sexual Dimorphism and Environmental "
        "Variability within a Community of Antarctic "
        "Penguins (Genus Pygoscelis). PLOS ONE 9(3): "
        "e90081."
    ),
    description=(
        "Morphometric measurements for Adelie, Chinstrap, "
        "and Gentoo penguins observed in the Palmer "
        "Archipelago."
    ),
    expected_insights=(
        ExpectedInsight(
            name="body_mass_vs_flipper_length",
            variables=(
                "body_mass_g",
                "flipper_length_mm"
            ),
            acceptable_charts=(
                "scatter",
            ),
            rationale=(
                "Body mass and flipper length have a strong "
                "morphometric relationship and are a common "
                "exploratory comparison in the dataset."
            )
        ),
        ExpectedInsight(
            name="species_vs_body_mass",
            variables=(
                "species",
                "body_mass_g"
            ),
            acceptable_charts=(
                "box",
                "bar"
            ),
            rationale=(
                "Body mass differs substantially across the "
                "three penguin species."
            )
        ),
        ExpectedInsight(
            name="species_vs_flipper_length",
            variables=(
                "species",
                "flipper_length_mm"
            ),
            acceptable_charts=(
                "box",
                "bar"
            ),
            rationale=(
                "Flipper length differs substantially across "
                "penguin species."
            )
        ),
        ExpectedInsight(
            name="species_vs_bill_length",
            variables=(
                "species",
                "bill_length_mm"
            ),
            acceptable_charts=(
                "box",
                "bar"
            ),
            rationale=(
                "Bill length is one of the morphological "
                "measurements that distinguishes species."
            )
        ),
    ),
    expected_semantics=(
        ExpectedSemanticType(
            column="species",
            acceptable_types=("categorical",),
            rationale="Species is a nominal category."
        ),
        ExpectedSemanticType(
            column="island",
            acceptable_types=("categorical",),
            rationale="Island is a nominal location category."
        ),
        ExpectedSemanticType(
            column="bill_length_mm",
            acceptable_types=("numeric_continuous",),
            rationale="Bill length is a continuous measurement."
        ),
        ExpectedSemanticType(
            column="bill_depth_mm",
            acceptable_types=("numeric_continuous",),
            rationale="Bill depth is a continuous measurement."
        ),
        ExpectedSemanticType(
            column="flipper_length_mm",
            acceptable_types=("numeric_continuous",),
            rationale="Flipper length is a measured quantity."
        ),
        ExpectedSemanticType(
            column="body_mass_g",
            acceptable_types=("numeric_continuous",),
            rationale="Body mass is a measured quantity."
        ),
        ExpectedSemanticType(
            column="sex",
            acceptable_types=("categorical",),
            rationale="Sex is represented as a nominal category."
        ),
        ExpectedSemanticType(
            column="year",
            acceptable_types=("temporal",),
            rationale="Year is an ordered temporal field."
        ),
    )
)


# ==================================================
# UCI BIKE SHARING
# ==================================================

BIKE_SHARING = RealWorldDataset(
    name="uci_bike_sharing_day",
    filename="bike_sharing_day.csv",
    domain="transportation",
    source=(
        "https://archive.ics.uci.edu/"
        "dataset/275/bike+sharing+dataset"
    ),
    license_name="CC BY 4.0",
    citation=(
        "Fanaee-T, H. (2013). Bike Sharing [Dataset]. "
        "UCI Machine Learning Repository. "
        "https://doi.org/10.24432/C5W894."
    ),
    description=(
        "Daily Capital Bikeshare rental demand from 2011 "
        "through 2012 with weather, calendar, and seasonal "
        "features."
    ),
    expected_insights=(
        ExpectedInsight(
            name="date_vs_total_rentals",
            variables=(
                "dteday",
                "cnt"
            ),
            acceptable_charts=(
                "line",
            ),
            rationale=(
                "Daily rental demand changes materially over "
                "the 731-date observation window, making a "
                "time-series view useful."
            )
        ),
        ExpectedInsight(
            name="temperature_vs_total_rentals",
            variables=(
                "temp",
                "cnt"
            ),
            acceptable_charts=(
                "scatter",
            ),
            rationale=(
                "Rental demand is meaningfully associated "
                "with temperature."
            )
        ),
        ExpectedInsight(
            name="humidity_vs_total_rentals",
            variables=(
                "hum",
                "cnt"
            ),
            acceptable_charts=(
                "scatter",
            ),
            rationale=(
                "Humidity is a weather variable with a "
                "meaningful relationship to rental demand."
            )
        ),
        ExpectedInsight(
            name="season_vs_total_rentals",
            variables=(
                "season",
                "cnt"
            ),
            acceptable_charts=(
                "box",
                "bar"
            ),
            rationale=(
                "UCI defines season as categorical, and daily "
                "rental demand differs across seasons."
            )
        ),
        ExpectedInsight(
            name="weather_situation_vs_total_rentals",
            variables=(
                "weathersit",
                "cnt"
            ),
            acceptable_charts=(
                "box",
                "bar"
            ),
            rationale=(
                "UCI defines weather situation as categorical, "
                "and rental demand differs across weather "
                "conditions."
            )
        ),
    ),
    expected_semantics=(
        ExpectedSemanticType(
            column="dteday",
            acceptable_types=("datetime",),
            rationale="UCI defines dteday as a date."
        ),
        ExpectedSemanticType(
            column="season",
            acceptable_types=("categorical", "ordinal"),
            rationale=(
                "Season is category-coded; either categorical "
                "or ordinal is acceptable for visualization "
                "generation."
            )
        ),
        ExpectedSemanticType(
            column="holiday",
            acceptable_types=("boolean",),
            rationale="Holiday is a binary indicator."
        ),
        ExpectedSemanticType(
            column="weekday",
            acceptable_types=("categorical",),
            rationale="Weekday codes identify nominal day categories."
        ),
        ExpectedSemanticType(
            column="workingday",
            acceptable_types=("boolean",),
            rationale="Working day is a binary indicator."
        ),
        ExpectedSemanticType(
            column="weathersit",
            acceptable_types=("categorical", "ordinal"),
            rationale=(
                "Weather situation is category-coded and also "
                "has an interpretable severity ordering."
            )
        ),
        ExpectedSemanticType(
            column="temp",
            acceptable_types=("numeric_continuous",),
            rationale="Temperature is a continuous measurement."
        ),
        ExpectedSemanticType(
            column="atemp",
            acceptable_types=("numeric_continuous",),
            rationale="Feeling temperature is continuous."
        ),
        ExpectedSemanticType(
            column="hum",
            acceptable_types=("numeric_continuous",),
            rationale="Normalized humidity is continuous."
        ),
        ExpectedSemanticType(
            column="windspeed",
            acceptable_types=("numeric_continuous",),
            rationale="Normalized wind speed is continuous."
        ),
        ExpectedSemanticType(
            column="cnt",
            acceptable_types=(
                "numeric_continuous",
                "numeric_discrete"
            ),
            rationale=(
                "Total rentals is a numeric count; either "
                "numeric treatment is acceptable."
            )
        ),
    ),
    preprocessing_notes=(
        "The benchmark uses UCI day.csv but removes instant "
        "(record identifier) and casual/registered because "
        "cnt is defined as their sum. Removing those columns "
        "prevents trivial identifier and target-component "
        "relationships from dominating the exploratory ranking. "
        "All remaining values, including integer-coded "
        "categories, are otherwise left unchanged."
    )
)


# ==================================================
# TITANIC3
# ==================================================

TITANIC = RealWorldDataset(
    name="titanic3",
    filename="titanic3_benchmark.csv",
    domain="passenger survival",
    source="https://hbiostat.org/data/",
    license_name=(
        "Use permitted by Vanderbilt University "
        "Department of Biostatistics"
    ),
    citation=(
        "Titanic3 data obtained from hbiostat.org/data "
        "courtesy of the Vanderbilt University Department "
        "of Biostatistics. The dataset was compiled and "
        "interpreted by Thomas Cason from Titanic passenger "
        "list sources described by Frank Harrell."
    ),
    description=(
        "Passenger-level Titanic data with survival status, "
        "passenger class, sex, age, family counts, fare, and "
        "embarkation location."
    ),
    expected_insights=(
        ExpectedInsight(
            name="survival_rate_by_sex",
            variables=(
                "sex",
                "survived"
            ),
            acceptable_charts=(
                "bar",
            ),
            aggregation="mean",
            required_x="sex",
            required_y="survived",
            rationale=(
                "Survival status differs substantially by sex. "
                "The intended visualization is survival rate "
                "by sex rather than the reverse orientation."
            )
        ),
        ExpectedInsight(
            name="survival_rate_by_passenger_class",
            variables=(
                "pclass",
                "survived"
            ),
            acceptable_charts=(
                "bar",
            ),
            aggregation="mean",
            required_x="pclass",
            required_y="survived",
            rationale=(
                "Passenger class is a core survival-related "
                "factor. The intended view is survival rate "
                "by passenger class."
            )
        ),
        ExpectedInsight(
            name="age_by_survival_status",
            variables=(
                "survived",
                "age"
            ),
            acceptable_charts=(
                "box",
            ),
            required_x="survived",
            required_y="age",
            rationale=(
                "Age distributions are useful to compare "
                "between survivors and non-survivors."
            )
        ),
        ExpectedInsight(
            name="fare_by_passenger_class",
            variables=(
                "pclass",
                "fare"
            ),
            acceptable_charts=(
                "box",
                "bar"
            ),
            required_x="pclass",
            required_y="fare",
            rationale=(
                "Passenger fares differ strongly across the "
                "three passenger classes."
            )
        ),
        ExpectedInsight(
            name="fare_distribution",
            variables=(
                "fare",
            ),
            acceptable_charts=(
                "histogram",
            ),
            required_x="fare",
            rationale=(
                "Passenger fare has a distinctly uneven "
                "distribution that is useful to inspect."
            )
        ),
    ),
    expected_semantics=(
        ExpectedSemanticType(
            column="pclass",
            acceptable_types=("categorical", "ordinal"),
            rationale=(
                "Passenger class is a three-level ordered "
                "category encoded numerically."
            )
        ),
        ExpectedSemanticType(
            column="survived",
            acceptable_types=("boolean",),
            rationale="Survival status is a binary 0/1 outcome."
        ),
        ExpectedSemanticType(
            column="sex",
            acceptable_types=("categorical",),
            rationale="Sex is a nominal category."
        ),
        ExpectedSemanticType(
            column="age",
            acceptable_types=("numeric_continuous",),
            rationale=(
                "Age is measured in years and includes "
                "fractional ages for some infants."
            )
        ),
        ExpectedSemanticType(
            column="sibsp",
            acceptable_types=("numeric_discrete",),
            rationale=(
                "SibSp is a genuine count of siblings/spouses "
                "aboard, not a category code."
            )
        ),
        ExpectedSemanticType(
            column="parch",
            acceptable_types=("numeric_discrete",),
            rationale=(
                "Parch is a genuine count of parents/children "
                "aboard, not a category code."
            )
        ),
        ExpectedSemanticType(
            column="fare",
            acceptable_types=("numeric_continuous",),
            rationale="Fare is a continuous monetary quantity."
        ),
        ExpectedSemanticType(
            column="embarked",
            acceptable_types=("categorical",),
            rationale="Embarkation port is a nominal category."
        ),
    ),
    preprocessing_notes=(
        "The source titanic3 dataset has 14 variables. The "
        "benchmark keeps only pclass, survived, sex, age, "
        "sibsp, parch, fare, and embarked. Name, ticket, cabin, "
        "home destination, boat, and body are omitted because "
        "they are high-cardinality identifiers/text fields or "
        "post-outcome/leakage-like fields that would distort an "
        "exploratory ranking. Source values in the retained "
        "columns are otherwise left unchanged."
    )
)


# ==================================================
# SUITE
# ==================================================

REAL_WORLD_DATASETS = (
    PENGUINS,
    BIKE_SHARING,
    TITANIC,
)


def build_real_world_suite():
    """
    Return the currently registered real-world datasets.

    Keep expectations pre-registered here before inspecting
    Visift's benchmark output. This reduces the temptation to
    redefine the target after seeing the ranking.
    """

    return list(
        REAL_WORLD_DATASETS
    )
