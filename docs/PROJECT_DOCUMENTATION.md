AI-Powered Global Health Intelligence System

1. Project Overview

The AI-Powered Global Health Intelligence System is a health surveillance and analytics application for monitoring infectious diseases and estimating outbreak risk at country level.

The system combines WHO disease data, data preprocessing, time-series analysis, anomaly detection, outbreak-risk scoring, machine learning, explainable predictions, a Streamlit dashboard, and a FastAPI service.

2. Supported Diseases

The system supports:

Malaria

Cholera

Tuberculosis

Measles

Meningitis

Disease data is retrieved from the WHO Global Health Observatory API.

3. Main Features

Disease Surveillance

The system retrieves disease observations and organizes them by country and year.

Data Preprocessing

Raw WHO observations are cleaned and transformed into a consistent structure for analysis.

Time-Series Analysis

Historical disease cases are analyzed to identify changes and trends over time.

Anomaly Detection

The system detects unusual observations in disease-case patterns.

Outbreak Risk Prediction

Each country-year observation receives an outbreak probability and risk classification.

Risk levels are:

LOW

MEDIUM

HIGH

Dashboard Filters

The Streamlit dashboard supports filtering by:

Disease

Country

Year

Countries are displayed using full country names rather than country-code abbreviations.

Warning and Alert Indicators

The dashboard displays global and country-level warnings based on calculated outbreak risk.

Machine Learning

The system compares:

Logistic Regression

Random Forest

Gradient Boosting

The models are evaluated using:

Accuracy

Precision

Recall

F1 Score

ROC AUC

Explainable Predictions

The system provides feature-level explanations for selected country predictions.

4. System Architecture

The system consists of four main layers:

Data layer

Analysis and machine-learning layer

Application layer

API layer

The data layer retrieves WHO observations.

The analysis layer performs preprocessing, time-series analysis, anomaly detection, risk scoring, and machine learning.

The application layer provides the Streamlit dashboard.

The API layer provides FastAPI endpoints for programmatic access.

5. Project Structure

ai-powered-global-health-intelligence/
│
├── api/
│ └── main.py
│
├── app/
│ ├── api.py
│ └── dashboard.py
│
├── data/
│ └── raw/
│
├── docs/
│ └── PROJECT_DOCUMENTATION.md
│
├── models/
├── notebooks/
│
├── src/
│ ├── anomaly_detection.py
│ ├── data_loader.py
│ ├── explainability.py
│ ├── ml_models.py
│ ├── outbreak_risk.py
│ ├── preprocessing.py
│ └── time_series.py
│
├── tests/
│ └── test_preprocessing.py
│
├── .streamlit/
│ └── config.toml
│
├── .gitignore
└── requirements.txt
