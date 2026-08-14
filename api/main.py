from fastapi import FastAPI, HTTPException
from src.data_loader import get_disease_data
from src.preprocessing import prepare_disease_data
from src.time_series import prepare_time_series, calculate_trend
from src.anomaly_detection import detect_anomalies
from src.outbreak_risk import calculate_outbreak_risk, get_country_ranking


app = FastAPI(
    title="AI-Powered Global Health Intelligence API",
    description="API for disease surveillance, anomaly detection, and outbreak risk analysis.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Global Health Intelligence API is running",
        "status": "online",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
    }


@app.get("/disease/{disease}")
def disease_analysis(disease: str):
    try:
        df = get_disease_data(disease)

        clean = prepare_disease_data(df)
        time_series = prepare_time_series(clean)
        time_series = detect_anomalies(time_series)
        time_series = calculate_trend(time_series)
        result = calculate_outbreak_risk(time_series)

        ranking = get_country_ranking(result)

        return {
            "disease": disease,
            "total_observations": len(result),
            "high_risk": int(
                (result["outbreak_risk_level"] == "HIGH").sum()
            ),
            "medium_risk": int(
                (result["outbreak_risk_level"] == "MEDIUM").sum()
            ),
            "low_risk": int(
                (result["outbreak_risk_level"] == "LOW").sum()
            ),
            "country_ranking": ranking.to_dict(orient="records"),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )