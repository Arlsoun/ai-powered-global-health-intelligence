import pandas as pd

from src.preprocessing import prepare_disease_data


def test_prepare_disease_data():
    data = pd.DataFrame({
        "SpatialDim": ["NGA", "NGA", "NGA"],
        "TimeDim": [2022, 2023, 2024],
        "NumericValue": [1000, 1200, 1500],
        "IndicatorCode": [
            "MALARIA_TOTAL_CASES",
            "MALARIA_TOTAL_CASES",
            "MALARIA_TOTAL_CASES",
        ],
        "disease": ["malaria", "malaria", "malaria"],
    })

    result = prepare_disease_data(data)

    assert len(result) == 3
    assert "country_code" in result.columns
    assert "year" in result.columns
    assert "cases" in result.columns
    assert "cases_change" in result.columns
    assert "rolling_mean_3" in result.columns


def test_cases_are_numeric():
    data = pd.DataFrame({
        "SpatialDim": ["NGA", "NGA"],
        "TimeDim": [2023, 2024],
        "NumericValue": [1000, 1500],
        "IndicatorCode": [
            "MALARIA_TOTAL_CASES",
            "MALARIA_TOTAL_CASES",
        ],
        "disease": ["malaria", "malaria"],
    })

    result = prepare_disease_data(data)

    assert pd.api.types.is_numeric_dtype(result["cases"])