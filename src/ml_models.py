import numpy as np
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
from sklearn.utils.class_weight import compute_sample_weight


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


GROUP_COLUMNS = [
    "disease",
    "country_code",
]


REQUIRED_COLUMNS = [
    "disease",
    "country_code",
    "year",
    "cases",
    "cases_change",
    "cases_pct_change",
    "rolling_mean_3",
    "rolling_std_3",
    "outbreak_risk_level",
]


def prepare_ml_data(df):
    """
    Prepare time-series data for future outbreak-risk prediction.

    Target:
        0 = LOW future outbreak risk
        1 = MEDIUM or HIGH future outbreak risk

    The model uses information from year t
    to predict the risk level in year t + 1.
    """

    data = df.copy()

    # ---------------------------------------------------------
    # VALIDATE REQUIRED COLUMNS
    # ---------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # ---------------------------------------------------------
    # SORT TIME SERIES
    # ---------------------------------------------------------

    data = data.sort_values(
        GROUP_COLUMNS + ["year"]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # ENSURE NUMERIC DATA
    # ---------------------------------------------------------

    numeric_columns = [
        "year",
        "cases",
        "cases_change",
        "cases_pct_change",
        "rolling_mean_3",
        "rolling_std_3",
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )

    # ---------------------------------------------------------
    # PREVIOUS YEAR CASES
    # ---------------------------------------------------------

    data["previous_cases"] = (
        data.groupby(GROUP_COLUMNS)["cases"]
        .shift(1)
    )

    # ---------------------------------------------------------
    # TWO-YEAR CHANGE
    # ---------------------------------------------------------

    data["change_2y"] = (
        data.groupby(GROUP_COLUMNS)["cases"]
        .diff(2)
    )

    # ---------------------------------------------------------
    # THREE-YEAR CHANGE
    # ---------------------------------------------------------

    data["change_3y"] = (
        data.groupby(GROUP_COLUMNS)["cases"]
        .diff(3)
    )

    # ---------------------------------------------------------
    # HISTORICAL MEAN
    # ---------------------------------------------------------
    # Only previous observations are used.
    # The current year's cases are excluded.

    data["historical_mean"] = (
        data.groupby(GROUP_COLUMNS)["cases"]
        .transform(
            lambda x:
            x.shift(1)
            .expanding()
            .mean()
        )
    )

    # ---------------------------------------------------------
    # CASES VS HISTORICAL MEAN
    # ---------------------------------------------------------

    data["cases_vs_historical_mean"] = (
        data["cases"]
        - data["historical_mean"]
    )

    # ---------------------------------------------------------
    # HISTORICAL TREND
    # ---------------------------------------------------------
    # Uses the previous three observations only.

    def calculate_trend(series):
        shifted = series.shift(1)

        return shifted.rolling(
            window=3,
            min_periods=3,
        ).apply(
            lambda values:
            (
                values[-1] - values[0]
            ) / 2,
            raw=True,
        )

    data["trend_slope"] = (
        data.groupby(GROUP_COLUMNS)["cases"]
        .transform(calculate_trend)
    )

    # ---------------------------------------------------------
    # FUTURE RISK LEVEL
    # ---------------------------------------------------------
    # Current year information predicts next year risk.

    data["future_risk_level"] = (
        data.groupby(GROUP_COLUMNS)[
            "outbreak_risk_level"
        ]
        .shift(-1)
    )

    # ---------------------------------------------------------
    # TARGET
    # ---------------------------------------------------------
    #
    # 0 = LOW
    # 1 = MEDIUM or HIGH
    #
    # future_risk_level is NOT included in FEATURE_COLUMNS.
    #

    data["target"] = (
        data["future_risk_level"]
        .isin(
            [
                "MEDIUM",
                "HIGH",
            ]
        )
        .astype(int)
    )

    # ---------------------------------------------------------
    # CLEAN INFINITE VALUES
    # ---------------------------------------------------------

    data = data.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    # ---------------------------------------------------------
    # REMOVE INVALID ROWS
    # ---------------------------------------------------------

    data = data.dropna(
        subset=[
            "previous_cases",
            "change_2y",
            "change_3y",
            "historical_mean",
            "trend_slope",
            "future_risk_level",
        ]
    )

    # ---------------------------------------------------------
    # VALIDATE FEATURE COLUMNS
    # ---------------------------------------------------------

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in data.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing ML feature columns: "
            f"{missing_features}"
        )

    data = data.reset_index(
        drop=True
    )

    return data


def build_models():
    """
    Create the three ML models used by the dashboard.
    """

    return {
        "Logistic Regression": Pipeline(
            [
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=3000,
                        random_state=42,
                    ),
                ),
            ]
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=3,
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
    """
    Train and evaluate one model.

    Gradient Boosting does not support class_weight,
    so balanced sample weights are supplied manually.
    """

    # ---------------------------------------------------------
    # GRADIENT BOOSTING CLASS BALANCING
    # ---------------------------------------------------------

    fit_parameters = {}

    if isinstance(
        model,
        GradientBoostingClassifier,
    ):
        sample_weights = compute_sample_weight(
            class_weight="balanced",
            y=y_train,
        )

        fit_parameters["sample_weight"] = sample_weights

    elif hasattr(
        model,
        "steps",
    ):
        final_model = model.steps[-1][1]

        if isinstance(
            final_model,
            GradientBoostingClassifier,
        ):
            sample_weights = compute_sample_weight(
                class_weight="balanced",
                y=y_train,
            )

            fit_parameters[
                "model__sample_weight"
            ] = sample_weights

    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------

    model.fit(
        X_train,
        y_train,
        **fit_parameters,
    )

    # ---------------------------------------------------------
    # PREDICTIONS
    # ---------------------------------------------------------

    predictions = model.predict(
        X_test
    )

    # ---------------------------------------------------------
    # PROBABILITIES
    # ---------------------------------------------------------

    if hasattr(
        model,
        "predict_proba",
    ):
        probabilities = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

    elif hasattr(
        model,
        "decision_function",
    ):
        probabilities = (
            model.decision_function(
                X_test
            )
        )

    else:
        probabilities = predictions.astype(
            float
        )

    # ---------------------------------------------------------
    # ROC-AUC
    # ---------------------------------------------------------

    if y_test.nunique() >= 2:

        roc_auc = roc_auc_score(
            y_test,
            probabilities,
        )

    else:

        roc_auc = 0.5

    # ---------------------------------------------------------
    # METRICS
    # ---------------------------------------------------------

    return {
        "accuracy": round(
            accuracy_score(
                y_test,
                predictions,
            ),
            4,
        ),

        "precision": round(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            4,
        ),

        "recall": round(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            4,
        ),

        "f1_score": round(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            ),
            4,
        ),

        "roc_auc": round(
            roc_auc,
            4,
        ),
    }


def find_temporal_split(
    data,
):
    """
    Find a temporal train/test split.

    The newest possible years are preferred for testing.

    The function tries to ensure both training and testing
    contain LOW and MEDIUM/HIGH observations.
    """

    unique_years = sorted(
        data["year"]
        .dropna()
        .unique()
    )

    if len(unique_years) < 3:
        return None

    # ---------------------------------------------------------
    # TRY DIFFERENT TEST WINDOWS
    # ---------------------------------------------------------

    possible_test_sizes = range(
        1,
        min(
            len(unique_years) - 1,
            6,
        ) + 1,
    )

    for test_size in possible_test_sizes:

        test_years = unique_years[
            -test_size:
        ]

        train_years = unique_years[
            :-test_size
        ]

        if not train_years:
            continue

        train_data = data[
            data["year"].isin(
                train_years
            )
        ]

        test_data = data[
            data["year"].isin(
                test_years
            )
        ]

        if train_data.empty:
            continue

        if test_data.empty:
            continue

        train_classes = (
            train_data["target"]
            .nunique()
        )

        test_classes = (
            test_data["target"]
            .nunique()
        )

        if (
            train_classes >= 2
            and test_classes >= 2
        ):
            return (
                train_data,
                test_data,
            )

    return None


def train_and_compare_models(
    df,
):
    """
    Prepare data, split temporally,
    train all models, and compare performance.
    """

    # ---------------------------------------------------------
    # PREPARE DATA
    # ---------------------------------------------------------

    data = prepare_ml_data(
        df
    )

    if data.empty:
        raise ValueError(
            "No valid data available "
            "for ML training."
        )

    # ---------------------------------------------------------
    # TARGET VALIDATION
    # ---------------------------------------------------------

    target_classes = sorted(
        data["target"]
        .unique()
    )

    if len(target_classes) < 2:
        raise ValueError(
            "Not enough future risk classes "
            "for ML training. The dataset "
            "needs both LOW and MEDIUM/HIGH "
            "future-risk observations."
        )

    # ---------------------------------------------------------
    # TEMPORAL SPLIT
    # ---------------------------------------------------------

    temporal_split = find_temporal_split(
        data
    )

    if temporal_split is not None:

        train_data, test_data = (
            temporal_split
        )

    else:

        # -----------------------------------------------------
        # FALLBACK
        # -----------------------------------------------------
        # If no temporal window contains both
        # classes, use a stratified random split.

        X = data[
            FEATURE_COLUMNS
        ]

        y = data[
            "target"
        ]

        try:

            (
                X_train,
                X_test,
                y_train,
                y_test,
            ) = train_test_split(
                X,
                y,
                test_size=0.25,
                random_state=42,
                stratify=y,
            )

        except ValueError:

            (
                X_train,
                X_test,
                y_train,
                y_test,
            ) = train_test_split(
                X,
                y,
                test_size=0.25,
                random_state=42,
            )

        train_data = None
        test_data = None

    # ---------------------------------------------------------
    # BUILD TRAIN/TEST MATRICES
    # ---------------------------------------------------------

    if (
        train_data is not None
        and test_data is not None
    ):

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

    # ---------------------------------------------------------
    # FINAL VALIDATION
    # ---------------------------------------------------------

    if y_train.nunique() < 2:

        raise ValueError(
            "Training data contains only one "
            "future risk class. More historical "
            "data is required."
        )

    if X_train.empty:

        raise ValueError(
            "Training dataset is empty."
        )

    if X_test.empty:

        raise ValueError(
            "Test dataset is empty."
        )

    # ---------------------------------------------------------
    # BUILD MODELS
    # ---------------------------------------------------------

    models = build_models()

    results = []

    trained_models = {}

    # ---------------------------------------------------------
    # TRAIN EACH MODEL
    # ---------------------------------------------------------

    for name, model in models.items():

        metrics = evaluate_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )

        results.append(
            {
                "model": name,
                **metrics,
            }
        )

        trained_models[
            name
        ] = model

    # ---------------------------------------------------------
    # MODEL COMPARISON
    # ---------------------------------------------------------

    comparison = pd.DataFrame(
        results
    )

    comparison = comparison.sort_values(
        by=[
            "f1_score",
            "recall",
            "roc_auc",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )

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
    """
    Return the highest-ranked model.
    """

    if comparison.empty:
        raise ValueError(
            "Model comparison is empty."
        )

    best_name = comparison.iloc[0][
        "model"
    ]

    if best_name not in models:
        raise ValueError(
            f"Trained model not found: "
            f"{best_name}"
        )

    return (
        best_name,
        models[best_name],
    )