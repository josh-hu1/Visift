import numpy as np
import pandas as pd


def generate_dataset(
    n_rows=500,
    seed=42,
    output_file="synthetic_sales.csv"
):
    """
    Generate a synthetic sales dataset with intentionally
    planted relationships for testing visualization ranking.
    """

    rng = np.random.default_rng(seed)

    # -----------------------------------
    # Basic dimensions
    # -----------------------------------

    customer_id = [
        f"CUST_{i:05d}"
        for i in range(1, n_rows + 1)
    ]

    dates = pd.date_range(
        start="2024-01-01",
        end="2026-12-31",
        periods=n_rows
    )

    regions = rng.choice(
        ["West", "East", "South", "North"],
        size=n_rows,
        p=[0.30, 0.25, 0.25, 0.20]
    )

    departments = rng.choice(
        ["Electronics", "Home", "Clothing", "Sports"],
        size=n_rows
    )

    # -----------------------------------
    # Advertising spend
    # -----------------------------------

    ad_spend = rng.normal(
        loc=5000,
        scale=1500,
        size=n_rows
    )

    ad_spend = np.clip(
        ad_spend,
        500,
        None
    )

    # -----------------------------------
    # Time trend
    # -----------------------------------

    time_index = np.arange(n_rows)

    time_trend = time_index * 15

    # -----------------------------------
    # Region effect
    # -----------------------------------

    region_effect_map = {
        "West": 5000,
        "East": 2500,
        "South": -1000,
        "North": 0
    }

    region_effect = np.array([
        region_effect_map[region]
        for region in regions
    ])

    # -----------------------------------
    # Revenue
    #
    # Strong relationship with:
    # - ad_spend
    # - time
    #
    # Moderate relationship with:
    # - region
    # -----------------------------------

    revenue_noise = rng.normal(
        loc=0,
        scale=5000,
        size=n_rows
    )

    revenue = (
        15000
        + (4.0 * ad_spend)
        + time_trend
        + region_effect
        + revenue_noise
    )

    # -----------------------------------
    # Units
    #
    # Moderately related to revenue
    # -----------------------------------

    units = (
        revenue / 1000
        + rng.normal(
            loc=0,
            scale=8,
            size=n_rows
        )
    )

    units = np.maximum(
        np.round(units),
        1
    ).astype(int)

    # -----------------------------------
    # Rating
    #
    # Weak/moderate positive relationship
    # with revenue.
    # -----------------------------------

    revenue_standardized = (
        revenue - revenue.mean()
    ) / revenue.std()

    rating_latent = (
        3
        + 0.4 * revenue_standardized
        + rng.normal(
            loc=0,
            scale=0.9,
            size=n_rows
        )
    )

    rating = np.clip(
        np.round(rating_latent),
        1,
        5
    ).astype(int)

    # -----------------------------------
    # Profit
    #
    # Intentionally noisy.
    # Department should have almost no
    # meaningful relationship with profit.
    # -----------------------------------

    profit = (
        revenue * 0.18
        + rng.normal(
            loc=0,
            scale=7000,
            size=n_rows
        )
    )

    # -----------------------------------
    # Boolean field
    # -----------------------------------

    subscribed = rng.choice(
        [True, False],
        size=n_rows,
        p=[0.60, 0.40]
    )

    # -----------------------------------
    # Add some missing values
    # -----------------------------------

    revenue_missing = rng.choice(
        n_rows,
        size=int(n_rows * 0.02),
        replace=False
    )

    rating_missing = rng.choice(
        n_rows,
        size=int(n_rows * 0.03),
        replace=False
    )

    revenue = revenue.astype(float)
    rating = rating.astype(float)

    revenue[revenue_missing] = np.nan
    rating[rating_missing] = np.nan

    # -----------------------------------
    # Build DataFrame
    # -----------------------------------

    df = pd.DataFrame({
        "customer_id": customer_id,
        "date": dates,
        "region": regions,
        "department": departments,
        "ad_spend": ad_spend.round(2),
        "revenue": revenue.round(2),
        "profit": profit.round(2),
        "units": units,
        "rating": rating,
        "subscribed": subscribed
    })

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"Generated {output_file} "
        f"with {len(df)} rows."
    )

    return df


if __name__ == "__main__":
    generate_dataset()