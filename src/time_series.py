import pandas as pd


def prepare_time_series(df):
    """
    Prepare disease data for time-series analysis.

    Expected columns:
        country_code
        year
        cases
        disease
    """

    required_columns = [
        "country_code",
        "year",
        "cases",
        "disease",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    result = df.copy()

    result["year"] = pd.to_numeric(
        result["year"],
        errors="coerce",
    )

    result["cases"] = pd.to_numeric(
        result["cases"],
        errors="coerce",
    )

    result = result.dropna(
        subset=[
            "country_code",
            "year",
            "cases",
        ]
    )

    result["year"] = result["year"].astype(int)

    result = result.sort_values(
        ["country_code", "year"]
    ).reset_index(drop=True)

    result["cases_change"] = (
        result.groupby("country_code")["cases"]
        .diff()
        .fillna(0)
    )

    result["cases_pct_change"] = (
        result.groupby("country_code")["cases"]
        .pct_change()
        .replace(
            [float("inf"), float("-inf")],
            pd.NA,
        )
        .fillna(0)
    )

    result["rolling_mean_3"] = (
        result.groupby("country_code")["cases"]
        .transform(
            lambda series: series.rolling(
                window=3,
                min_periods=1,
            ).mean()
        )
    )

    result["rolling_std_3"] = (
        result.groupby("country_code")["cases"]
        .transform(
            lambda series: series.rolling(
                window=3,
                min_periods=1,
            ).std()
        )
        .fillna(0)
    )

    return result


def calculate_trend(df):
    """
    Calculate the overall trend for each country.

    Returns the original dataframe with:
        trend_slope
        trend_direction
    """

    result = df.copy()

    slopes = {}

    for country, group in result.groupby("country_code"):
        group = group.sort_values("year")

        if len(group) < 2:
            slopes[country] = 0.0
            continue

        x = group["year"].astype(float)
        y = group["cases"].astype(float)

        slope = (
            ((x - x.mean()) * (y - y.mean())).sum()
            / ((x - x.mean()) ** 2).sum()
        )

        slopes[country] = float(slope)

    result["trend_slope"] = (
        result["country_code"]
        .map(slopes)
    )

    result["trend_direction"] = "STABLE"

    result.loc[
        result["trend_slope"] > 0,
        "trend_direction",
    ] = "INCREASING"

    result.loc[
        result["trend_slope"] < 0,
        "trend_direction",
    ] = "DECREASING"

    return result


def get_country_time_series(df, country_code):
    """
    Return the time series for one country.
    """

    result = df[
        df["country_code"].str.upper()
        == country_code.upper()
    ].copy()

    return result.sort_values("year").reset_index(
        drop=True
    )


def get_latest_country_status(df):
    """
    Return the latest available observation
    for each country.
    """

    result = (
        df.sort_values("year")
        .groupby("country_code")
        .tail(1)
        .sort_values(
            "cases",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return result