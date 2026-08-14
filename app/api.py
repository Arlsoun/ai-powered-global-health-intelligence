from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data_loader import get_disease_data
from src.preprocessing import prepare_disease_data
from src.time_series import prepare_time_series, calculate_trend
from src.anomaly_detection import detect_anomalies
from src.outbreak_risk import calculate_outbreak_risk


app = FastAPI(
    title="AI-Powered Global Health Intelligence API",
    description="API for disease surveillance and outbreak-risk intelligence.",
    version="1.0.0",
)


class DiseaseRequest(BaseModel):
    disease: str


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }


@app.get("/diseases")
def get_diseases():
    return {
        "diseases": [
            "malaria",
            "cholera",
            "tuberculosis",
            "measles",
            "meningitis",
        ]
    }


@app.post("/analyze")
def analyze_disease(request: DiseaseRequest):

    try:
        df = get_disease_data(request.disease)

        clean = prepare_disease_data(df)

        time_series = prepare_time_series(clean)

        time_series = detect_anomalies(time_series)

        time_series = calculate_trend(time_series)

        risk = calculate_outbreak_risk(time_series)

        latest = (
            risk.sort_values("year")
            .groupby("country_code")
            .tail(1)
            .copy()
        )

        results = latest[
            [
                "country_code",
                "year",
                "cases",
                "outbreak_probability",
                "outbreak_risk_level",
                "trend_direction",
                "is_anomaly",
            ]
        ].copy()

        results = results.fillna(0)

        return {
            "disease": request.disease,
            "observations": len(risk),
            "countries": len(results),
            "results": results.to_dict(orient="records"),
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=str(error),
        )