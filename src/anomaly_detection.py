import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_anomalies(
    df: pd.DataFrame,
    contamination: float = 0.05,
) -> pd.DataFrame:
    """
    Detect unusual disease-case observations by country.

    A separate Isolation Forest model is fitted for each
    country and disease combination.
    """

    required_columns = [
        "country_code",
        "year",
        "cases",
        "cases_change",
        "cases_pct_change",
        "rolling_mean_3",
        "rolling_std_3",
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

    result["anomaly"] = 0
    result["anomaly_score"] = 0.0

    feature_columns = [
        "cases",
        "cases_change",
        "cases_pct_change",
        "rolling_mean_3",
        "rolling_std_3",
    ]

    for (
        disease,
        country,
    ), group in result.groupby(
        ["disease", "country_code"],
        sort=False,
    ):
        indices = group.index

        features = (
            group[feature_columns]
            .replace(
                [float("inf"), -float("inf")],
                pd.NA,
            )
            .fillna(0)
        )

        if len(group) < 5:
            continue

        model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=200,
        )

        predictions = model.fit_predict(
            features
        )

        scores = model.decision_function(
            features
        )

        result.loc[
            indices,
            "anomaly"
        ] = predictions

        result.loc[
            indices,
            "anomaly_score"
        ] = scores

    result["is_anomaly"] = (
        result["anomaly"] == -1
    )

    return result


def get_anomalies(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Return only observations identified as anomalous."""

    if "is_anomaly" not in df.columns:
        raise ValueError(
            "Run detect_anomalies() first."
        )

    return (
        df[df["is_anomaly"]]
        .sort_values(
            [
                "disease",
                "year",
                "anomaly_score",
            ]
        )
        .reset_index(drop=True)
    )