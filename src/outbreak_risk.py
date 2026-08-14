import pandas as pd


def calculate_outbreak_risk(df):
    """
    Calculate an initial outbreak-risk score.

    The score combines:
    1. Recent percentage change
    2. Current cases compared with rolling average
    3. Anomaly detection
    4. Long-term trend direction
    """

    result = df.copy()

    required_columns = [
        "cases",
        "cases_pct_change",
        "rolling_mean_3",
        "is_anomaly",
        "trend_direction",
    ]

    missing = [
        column
        for column in required_columns
        if column not in result.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    result["cases_pct_change"] = pd.to_numeric(
        result["cases_pct_change"],
        errors="coerce",
    ).fillna(0)

    result["rolling_mean_3"] = pd.to_numeric(
        result["rolling_mean_3"],
        errors="coerce",
    ).fillna(result["cases"])

    result["cases"] = pd.to_numeric(
        result["cases"],
        errors="coerce",
    ).fillna(0)

    # Recent increase score
    result["increase_score"] = (
        result["cases_pct_change"]
        .clip(lower=0, upper=2)
        / 2
    )

    # Current cases compared with rolling average
    result["relative_cases"] = (
        result["cases"]
        / result["rolling_mean_3"].replace(0, pd.NA)
    )

    result["relative_cases"] = (
        result["relative_cases"]
        .fillna(1)
    )

    result["relative_cases_score"] = (
        (result["relative_cases"] - 1)
        .clip(lower=0, upper=1)
    )

    # Anomaly score
    result["anomaly_component"] = (
        result["is_anomaly"]
        .astype(int)
    )

    # Trend component
    result["trend_component"] = (
        result["trend_direction"]
        .map(
            {
                "INCREASING": 1.0,
                "STABLE": 0.5,
                "DECREASING": 0.0,
            }
        )
        .fillna(0.5)
    )

    # Combined outbreak score
    result["outbreak_score"] = (
        0.35 * result["increase_score"]
        + 0.25 * result["relative_cases_score"]
        + 0.25 * result["anomaly_component"]
        + 0.15 * result["trend_component"]
    )

    result["outbreak_probability"] = (
        result["outbreak_score"] * 100
    ).clip(
        lower=0,
        upper=100,
    )

    # Risk level
    result["outbreak_risk_level"] = "LOW"

    result.loc[
        result["outbreak_probability"] >= 35,
        "outbreak_risk_level",
    ] = "MEDIUM"

    result.loc[
        result["outbreak_probability"] >= 60,
        "outbreak_risk_level",
    ] = "HIGH"

    return result


def get_high_risk_outbreaks(df):
    """
    Return observations with HIGH outbreak risk.
    """

    return (
        df[
            df["outbreak_risk_level"] == "HIGH"
        ]
        .sort_values(
            "outbreak_probability",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def get_country_ranking(df):
    """
    Rank countries according to their latest
    outbreak probability.
    """

    latest = (
        df.sort_values("year")
        .groupby("country_code")
        .tail(1)
        .copy()
    )

    ranking = (
        latest[
            [
                "country_code",
                "year",
                "cases",
                "outbreak_probability",
                "outbreak_risk_level",
                "trend_direction",
            ]
        ]
        .sort_values(
            "outbreak_probability",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    ranking.insert(
        0,
        "rank",
        range(1, len(ranking) + 1),
    )

    return ranking