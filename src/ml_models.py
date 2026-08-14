import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


FEATURE_COLUMNS = [
    "cases",
    "previous_cases",
    "cases_change",
    "cases_pct_change",
    "change_2y",
    "change_3y",
    "rolling_mean_3",
    "rolling_std_3",
    "historical_mean",
    "cases_vs_historical_mean",
    "trend_slope",
    "year",
]


def prepare_ml_data(df):
    data = df.copy()

    data = data.sort_values(
        ["year", "country_code"]
    ).reset_index(drop=True)

    data["previous_cases"] = (
        data.groupby("country_code")["cases"].shift(1)
    )

    data["change_2y"] = (
        data.groupby("country_code")["cases"].diff(2)
    )

    data["change_3y"] = (
        data.groupby("country_code")["cases"].diff(3)
    )

    data["historical_mean"] = (
        data.groupby("country_code")["cases"]
        .transform(
            lambda x: x.shift(1).expanding().mean()
        )
    )

    data["cases_vs_historical_mean"] = (
        data["cases"] - data["historical_mean"]
    )

    data["trend_slope"] = (
        data.groupby("country_code")["cases"]
        .transform(
            lambda x: x.shift(1)
            .rolling(3)
            .apply(
                lambda y: (
                    (y.iloc[-1] - y.iloc[0]) / 2
                    if len(y) == 3
                    else 0
                ),
                raw=False,
            )
        )
    )

    data["target"] = (
        data["outbreak_risk_level"]
        .eq("HIGH")
        .astype(int)
    )

    data = data.dropna(
        subset=[
            "previous_cases",
            "change_2y",
            "change_3y",
            "historical_mean",
            "trend_slope",
        ]
    )

    return data


def build_models():
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }


def evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
):
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    else:
        probabilities = model.decision_function(X_test)

    if y_test.nunique() == 2:
        roc_auc = roc_auc_score(
            y_test,
            probabilities,
        )
    else:
        roc_auc = 0.5

    return {
        "accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "roc_auc": roc_auc,
    }


def train_and_compare_models(df):
    data = prepare_ml_data(df)

    if data.empty:
        raise ValueError(
            "No valid data available for ML training."
        )

    data = data.sort_values(
        "year"
    ).reset_index(drop=True)

    X = data[FEATURE_COLUMNS]
    y = data["target"]

    if y.nunique() < 2:
        raise ValueError(
            "Not enough risk classes for ML training."
        )

    unique_years = sorted(
        data["year"].unique()
    )

    if len(unique_years) >= 5:

        test_year_count = max(
            2,
            int(len(unique_years) * 0.2),
        )

        test_years = unique_years[
            -test_year_count:
        ]

        train_data = data[
            ~data["year"].isin(test_years)
        ]

        test_data = data[
            data["year"].isin(test_years)
        ]

        X_train = train_data[
            FEATURE_COLUMNS
        ]

        y_train = train_data[
            "target"
        ]

        X_test = test_data[
            FEATURE_COLUMNS
        ]

        y_test = test_data[
            "target"
        ]

    else:

        try:

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=0.25,
                    random_state=42,
                    stratify=y,
                )
            )

        except ValueError:

            X_train, X_test, y_train, y_test = (
                train_test_split(
                    X,
                    y,
                    test_size=0.25,
                    random_state=42,
                )
            )

    if y_train.nunique() < 2:
        raise ValueError(
            "Training data contains only one risk class."
        )

    models = build_models()

    results = []
    trained_models = {}

    for name, model in models.items():

        metrics = evaluate_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        results.append(
            {
                "model": name,
                **metrics,
            }
        )

        trained_models[name] = model

    comparison = pd.DataFrame(
        results
    )

    comparison = comparison.sort_values(
        [
            "f1_score",
            "recall",
            "roc_auc",
        ],
        ascending=False,
    ).reset_index(drop=True)

    return (
        comparison,
        trained_models,
        X_test,
        y_test,
    )


def get_best_model(
    comparison,
    models,
):
    best_name = comparison.iloc[0][
        "model"
    ]

    return (
        best_name,
        models[best_name],
    )