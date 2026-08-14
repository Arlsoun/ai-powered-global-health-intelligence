import pandas as pd


def clean_disease_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Clean WHO disease surveillance data.

    Required columns:
    SpatialDim
    TimeDim
    NumericValue
    disease
    """

    required_columns = [
        "SpatialDim",
        "TimeDim",
        "NumericValue",
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

    cleaned = df[
        required_columns
    ].copy()

    cleaned = cleaned.rename(
        columns={
            "SpatialDim": "country_code",
            "TimeDim": "year",
            "NumericValue": "cases",
        }
    )

    cleaned["year"] = pd.to_numeric(
        cleaned["year"],
        errors="coerce",
    )

    cleaned["cases"] = pd.to_numeric(
        cleaned["cases"],
        errors="coerce",
    )

    cleaned["country_code"] = (
        cleaned["country_code"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    cleaned["disease"] = (
        cleaned["disease"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    cleaned = cleaned.dropna(
        subset=[
            "country_code",
            "year",
            "cases",
        ]
    )

    cleaned = cleaned[
        cleaned["cases"] >= 0
    ]

    cleaned["year"] = cleaned["year"].astype(int)

    cleaned["cases"] = cleaned["cases"].astype(float)

    cleaned = (
        cleaned
        .sort_values(
            [
                "disease",
                "country_code",
                "year",
            ]
        )
        .drop_duplicates(
            subset=[
                "disease",
                "country_code",
                "year",
            ],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return cleaned


def create_time_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create time-series features for outbreak analysis.
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

    group_columns = [
        "disease",
        "country_code",
    ]

    result["cases_change"] = (
        result.groupby(group_columns)["cases"]
        .diff()
    )

    result["cases_pct_change"] = (
        result.groupby(group_columns)["cases"]
        .pct_change()
        .replace(
            [float("inf"), -float("inf")],
            pd.NA,
        )
    )

    result["rolling_mean_3"] = (
        result.groupby(group_columns)["cases"]
        .transform(
            lambda series: series
            .rolling(3, min_periods=1)
            .mean()
        )
    )

    result["rolling_std_3"] = (
        result.groupby(group_columns)["cases"]
        .transform(
            lambda series: series
            .rolling(3, min_periods=2)
            .std()
        )
    )

    result["cases_change"] = (
        result["cases_change"]
        .fillna(0)
    )

    result["cases_pct_change"] = (
        result["cases_pct_change"]
        .fillna(0)
    )

    result["rolling_std_3"] = (
        result["rolling_std_3"]
        .fillna(0)
    )

    return result


def prepare_disease_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Complete preprocessing pipeline.
    """

    cleaned = clean_disease_data(df)

    prepared = create_time_features(
        cleaned
    )

    return prepared