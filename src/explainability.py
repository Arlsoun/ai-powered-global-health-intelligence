import pandas as pd


FEATURE_LABELS = {
    "cases": "Current cases",
    "previous_cases": "Previous-year cases",
    "cases_change": "Recent case change",
    "cases_pct_change": "Recent percentage change",
    "change_2y": "Two-year case change",
    "change_3y": "Three-year case change",
    "rolling_mean_3": "Three-year average",
    "rolling_std_3": "Recent volatility",
    "historical_mean": "Historical average",
    "cases_vs_historical_mean": "Cases compared with historical average",
    "trend_slope": "Recent trend",
    "year": "Year",
}


def get_feature_importance(model):
    """
    Extract feature importance or coefficients from a trained model.
    """

    # Pipeline containing Logistic Regression
    if hasattr(model, "named_steps"):
        estimator = model.named_steps.get("model")

        if estimator is not None and hasattr(
            estimator,
            "coef_",
        ):
            return estimator.coef_[0]

    # Tree-based models
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_

    # Direct linear model
    if hasattr(model, "coef_"):
        return model.coef_[0]

    raise ValueError(
        "The supplied model does not provide "
        "feature importance information."
    )


def explain_prediction(
    model,
    row,
    feature_columns,
):
    """
    Explain one model prediction by ranking
    the contribution of each feature.
    """

    values = []

    importance = get_feature_importance(model)

    if len(importance) != len(feature_columns):
        raise ValueError(
            "The number of model coefficients does not "
            "match the number of feature columns."
        )

    for feature, importance_value in zip(
        feature_columns,
        importance,
    ):
        if feature not in row.index:
            continue

        value = row[feature]

        if pd.isna(value):
            continue

        values.append(
            {
                "feature": feature,
                "label": FEATURE_LABELS.get(
                    feature,
                    feature.replace("_", " ").title(),
                ),
                "value": float(value),
                "importance": float(
                    importance_value
                ),
                "absolute_importance": abs(
                    float(importance_value)
                ),
            }
        )

    explanation = pd.DataFrame(values)

    if explanation.empty:
        return explanation

    explanation = explanation.sort_values(
        "absolute_importance",
        ascending=False,
    ).reset_index(drop=True)

    return explanation


def get_top_factors(
    model,
    row,
    feature_columns,
    top_n=5,
):
    """
    Return the most influential features.
    """

    explanation = explain_prediction(
        model,
        row,
        feature_columns,
    )

    if explanation.empty:
        return explanation

    return explanation.head(top_n)


def describe_factor(
    feature,
    value,
    row=None,
):
    """
    Convert a feature value into a simple
    human-readable explanation.
    """

    if feature == "cases_pct_change":
        percentage = value * 100

        if percentage > 0:
            return (
                f"Cases increased by "
                f"{percentage:.1f}% recently."
            )

        return (
            f"Cases decreased by "
            f"{abs(percentage):.1f}% recently."
        )

    if feature == "cases_vs_historical_mean":

        if value > 1:
            return (
                f"Current cases are "
                f"{value:.2f} times the historical average."
            )

        return (
            f"Current cases are "
            f"{value:.2f} times the historical average."
        )

    if feature == "trend_slope":

        if value > 0:
            return "The recent case trend is increasing."

        if value < 0:
            return "The recent case trend is decreasing."

        return "The recent case trend is stable."

    if feature == "cases":
        return (
            f"Current reported cases: "
            f"{value:,.0f}."
        )

    if feature == "previous_cases":
        return (
            f"Previous-year cases: "
            f"{value:,.0f}."
        )

    if feature == "rolling_mean_3":
        return (
            f"Three-year rolling average: "
            f"{value:,.0f}."
        )

    if feature == "rolling_std_3":
        return (
            f"Recent case volatility: "
            f"{value:,.0f}."
        )

    if feature == "cases_change":
        if value > 0:
            return (
                f"Cases increased by "
                f"{value:,.0f} from the previous year."
            )

        return (
            f"Cases changed by "
            f"{value:,.0f} from the previous year."
        )

    if feature == "change_2y":
        return (
            f"Two-year case change: "
            f"{value:,.0f}."
        )

    if feature == "change_3y":
        return (
            f"Three-year case change: "
            f"{value:,.0f}."
        )

    if feature == "historical_mean":
        return (
            f"Historical average: "
            f"{value:,.0f}."
        )

    return (
        f"{FEATURE_LABELS.get(feature, feature)}: "
        f"{value:,.2f}."
    )


def build_explanation(
    model,
    row,
    feature_columns,
    top_n=5,
):
    """
    Create a human-readable explanation
    for one prediction.
    """

    top_factors = get_top_factors(
        model,
        row,
        feature_columns,
        top_n=top_n,
    )

    explanations = []

    for _, factor in top_factors.iterrows():

        text = describe_factor(
            factor["feature"],
            factor["value"],
            row,
        )

        direction = (
            "increases"
            if factor["importance"] > 0
            else "decreases"
        )

        explanations.append(
            {
                "feature": factor["feature"],
                "label": factor["label"],
                "value": factor["value"],
                "importance": factor["importance"],
                "direction": direction,
                "explanation": text,
            }
        )

    return pd.DataFrame(explanations)


def explain_dataset(
    model,
    df,
    feature_columns,
    top_n=5,
):
    """
    Generate explanations for every row
    in a dataset.
    """

    explanations = []

    for index, row in df.iterrows():

        try:
            explanation = build_explanation(
                model,
                row,
                feature_columns,
                top_n=top_n,
            )

            for _, factor in explanation.iterrows():

                explanations.append(
                    {
                        "index": index,
                        "feature": factor["feature"],
                        "label": factor["label"],
                        "value": factor["value"],
                        "importance": factor["importance"],
                        "direction": factor["direction"],
                        "explanation": factor["explanation"],
                    }
                )

        except (
            ValueError,
            TypeError,
            KeyError,
        ):
            continue

    return pd.DataFrame(explanations)