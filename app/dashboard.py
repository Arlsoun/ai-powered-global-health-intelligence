import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import pycountry

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import get_disease_data
from src.preprocessing import prepare_disease_data
from src.time_series import (
    prepare_time_series,
    calculate_trend,
)
from src.anomaly_detection import detect_anomalies
from src.outbreak_risk import (
    calculate_outbreak_risk,
    get_country_ranking,
)
from src.ml_models import (
    train_and_compare_models,
    get_best_model,
)
from src.explainability import build_explanation


st.set_page_config(
    page_title="Global Health Intelligence",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# COUNTRY NAME CONVERSION
# ============================================================

SPECIAL_COUNTRY_NAMES = {
    "GLOBAL": "Global",
    "AFR": "Africa",
    "AMR": "Americas",
    "EMR": "Eastern Mediterranean Region",
    "EUR": "Europe",
    "SEAR": "South-East Asia Region",
    "WPR": "Western Pacific Region",
}


def country_name(code):
    code = str(code).strip().upper()

    if code in SPECIAL_COUNTRY_NAMES:
        return SPECIAL_COUNTRY_NAMES[code]

    country = pycountry.countries.get(alpha_3=code)

    if country:
        name = country.name

        # Use common English names where pycountry uses formal names
        common_names = {
            "Türkiye": "Turkey",
            "Russian Federation": "Russia",
            "Viet Nam": "Vietnam",
            "Iran (Islamic Republic of)": "Iran",
            "Syrian Arab Republic": "Syria",
            "Bolivia, Plurinational State of": "Bolivia",
            "Venezuela, Bolivarian Republic of": "Venezuela",
            "Tanzania, United Republic of": "Tanzania",
            "Congo, The Democratic Republic of the": (
                "Democratic Republic of the Congo"
            ),
            "Congo": "Republic of the Congo",
            "Korea, Republic of": "South Korea",
            "Korea, Democratic People's Republic of": (
                "North Korea"
            ),
            "Lao People's Democratic Republic": "Laos",
            "Moldova, Republic of": "Moldova",
            "Brunei Darussalam": "Brunei",
            "Cabo Verde": "Cape Verde",
            "Côte d'Ivoire": "Ivory Coast",
            "Türkiye": "Turkey",
            "United States": "United States",
            "United Kingdom": "United Kingdom",
        }

        return common_names.get(name, name)

    return code


# ============================================================
# STYLE
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #07111F;
        color: #F8FAFC;
    }

    [data-testid="stHeader"] {
        background-color: #07111F;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #0F172A 0%,
            #172554 100%
        );
    }

    section[data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }

    .title {
        color: #F8FAFC;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #CBD5E1;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        color: #F8FAFC;
        font-size: 1.35rem;
        font-weight: 750;
        margin-top: 1.5rem;
        margin-bottom: 0.8rem;
    }

    div[data-testid="metric-container"] {
        background-color: #0D1B2E;
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 16px;
    }

    div[data-testid="metric-container"] label {
        color: #CBD5E1 !important;
    }

    div[data-testid="metric-container"]
    [data-testid="stMetricValue"] {
        color: #F8FAFC !important;
        font-weight: 800;
    }

    .high {
        background-color: #3B1115;
        border-left: 6px solid #EF4444;
        color: #FCA5A5;
        padding: 1rem;
        border-radius: 12px;
    }

    .medium {
        background-color: #3A1F0B;
        border-left: 6px solid #F97316;
        color: #FDBA74;
        padding: 1rem;
        border-radius: 12px;
    }

    .low {
        background-color: #0B2A1A;
        border-left: 6px solid #22C55E;
        color: #86EFAC;
        padding: 1rem;
        border-radius: 12px;
    }

    .info {
        background-color: #0B2340;
        border-left: 6px solid #38BDF8;
        color: #BAE6FD;
        padding: 1rem;
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="title">
         AI-Powered Global Health Intelligence System
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Disease surveillance, anomaly detection,
        outbreak-risk prediction and country-level intelligence
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA PIPELINE
# ============================================================

@st.cache_data(ttl=3600)
def load_pipeline_data(disease):
    df = get_disease_data(disease)

    clean = prepare_disease_data(df)

    time_series = prepare_time_series(clean)

    time_series = detect_anomalies(time_series)

    time_series = calculate_trend(time_series)

    risk = calculate_outbreak_risk(time_series)

    return risk


@st.cache_resource(ttl=3600)
def train_models(risk_data):
    comparison, models, X_test, y_test = (
        train_and_compare_models(risk_data)
    )

    best_name, best_model = get_best_model(
        comparison,
        models,
    )

    return (
        comparison,
        models,
        X_test,
        y_test,
        best_name,
        best_model,
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("🔎 Health Intelligence Filters")

st.sidebar.caption(
    "Select a disease and country to inspect outbreak risk."
)

disease_options = [
    "malaria",
    "cholera",
    "tuberculosis",
    "measles",
    "meningitis",
]

selected_disease = st.sidebar.selectbox(
    "Disease",
    disease_options,
    index=0,
)


# ============================================================
# LOAD DATA
# ============================================================

try:

    risk_df = load_pipeline_data(
        selected_disease
    )

except Exception as error:

    st.error(
        f"Unable to load health data: {error}"
    )

    st.stop()


# ============================================================
# MODEL
# ============================================================

try:

    (
        comparison,
        models,
        X_test,
        y_test,
        best_name,
        best_model,
    ) = train_models(
        risk_df
    )

except Exception as error:

    st.error(
        f"Unable to train the ML models: {error}"
    )

    st.stop()


# ============================================================
# COUNTRY FILTER
# ============================================================

country_codes = sorted(
    risk_df["country_code"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

country_options = {
    country_name(code): code
    for code in country_codes
}

country_names = sorted(
    country_options.keys()
)

selected_country_name = st.sidebar.selectbox(
    "Country",
    ["All Countries"] + country_names,
)

if selected_country_name == "All Countries":
    selected_country = "All Countries"
else:
    selected_country = country_options[
        selected_country_name
    ]


# ============================================================
# YEAR FILTER
# ============================================================

min_year = int(
    risk_df["year"].min()
)

max_year = int(
    risk_df["year"].max()
)

selected_year = st.sidebar.slider(
    "Year",
    min_value=min_year,
    max_value=max_year,
    value=max_year,
)


# ============================================================
# FILTER DATA
# ============================================================

filtered_df = risk_df[
    risk_df["year"] == selected_year
].copy()

if selected_country != "All Countries":

    filtered_df = filtered_df[
        filtered_df["country_code"]
        == selected_country
    ].copy()


# ============================================================
# TOP METRICS
# ============================================================

st.markdown(
    '<div class="section-title">📊 Global Health Status</div>',
    unsafe_allow_html=True,
)

total_observations = len(filtered_df)

high_risk = int(
    (
        filtered_df["outbreak_risk_level"]
        == "HIGH"
    ).sum()
)

medium_risk = int(
    (
        filtered_df["outbreak_risk_level"]
        == "MEDIUM"
    ).sum()
)

anomalies = int(
    filtered_df["is_anomaly"]
    .sum()
)


col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Observed Countries",
        f"{total_observations:,}",
    )

with col2:

    st.metric(
        "🔴 High Risk",
        f"{high_risk:,}",
    )

with col3:

    st.metric(
        "🟠 Medium Risk",
        f"{medium_risk:,}",
    )

with col4:

    st.metric(
        "⚠️ Anomalies",
        f"{anomalies:,}",
    )


# ============================================================
# ALERT STATUS
# ============================================================

st.markdown(
    '<div class="section-title">🚨 Outbreak Alert Status</div>',
    unsafe_allow_html=True,
)

if selected_country != "All Countries":

    if filtered_df.empty:

        st.markdown(
            """
            <div class="info">
                No data available for the selected country and year.
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        row = filtered_df.iloc[0]

        probability = float(
            row["outbreak_probability"]
        )

        level = row[
            "outbreak_risk_level"
        ]

        if level == "HIGH":

            st.markdown(
                f"""
                <div class="high">
                    🔴 <strong>HIGH OUTBREAK RISK</strong><br>
                    Country:
                    <strong>{selected_country_name}</strong><br>
                    Outbreak probability:
                    <strong>{probability:.1f}%</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

        elif level == "MEDIUM":

            st.markdown(
                f"""
                <div class="medium">
                    🟠 <strong>MEDIUM OUTBREAK RISK</strong><br>
                    Country:
                    <strong>{selected_country_name}</strong><br>
                    Outbreak probability:
                    <strong>{probability:.1f}%</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                f"""
                <div class="low">
                    🟢 <strong>LOW OUTBREAK RISK</strong><br>
                    Country:
                    <strong>{selected_country_name}</strong><br>
                    Outbreak probability:
                    <strong>{probability:.1f}%</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

else:

    if high_risk > 0:

        st.markdown(
            f"""
            <div class="high">
                🔴 <strong>GLOBAL WARNING</strong><br>
                {high_risk:,} country observations
                are classified as high outbreak risk.
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif medium_risk > 0:

        st.markdown(
            f"""
            <div class="medium">
                🟠 <strong>MONITORING REQUIRED</strong><br>
                {medium_risk:,} country observations
                are classified as medium outbreak risk.
            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            """
            <div class="low">
                🟢 <strong>LOW GLOBAL RISK</strong><br>
                No high or medium outbreak-risk
                observations were detected.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# COUNTRY RANKING
# ============================================================

st.markdown(
    '<div class="section-title"> Country Outbreak Ranking</div>',
    unsafe_allow_html=True,
)

ranking = get_country_ranking(
    risk_df
)

ranking = ranking[
    ranking["year"] == selected_year
].copy()

ranking_display = ranking[
    [
        "rank",
        "country_code",
        "year",
        "cases",
        "outbreak_probability",
        "outbreak_risk_level",
        "trend_direction",
    ]
].head(20).copy()

ranking_display["country"] = (
    ranking_display["country_code"]
    .apply(country_name)
)

ranking_display = ranking_display[
    [
        "rank",
        "country",
        "year",
        "cases",
        "outbreak_probability",
        "outbreak_risk_level",
        "trend_direction",
    ]
]

ranking_display = ranking_display.rename(
    columns={
        "cases": "Cases",
        "outbreak_probability": "Outbreak Probability (%)",
        "outbreak_risk_level": "Risk Level",
        "trend_direction": "Trend",
        "year": "Year",
        "rank": "Rank",
    }
)

st.dataframe(
    ranking_display,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# OUTBREAK PROBABILITY CHART
# ============================================================

st.markdown(
    '<div class="section-title">📈 Outbreak Probability</div>',
    unsafe_allow_html=True,
)

chart_df = ranking.head(15).copy()

if not chart_df.empty:

    chart_df["country"] = (
        chart_df["country_code"]
        .apply(country_name)
    )

    fig_probability = px.bar(
        chart_df,
        x="country",
        y="outbreak_probability",
        color="outbreak_risk_level",
        title=(
            f"{selected_disease.title()} outbreak "
            f"probability by country"
        ),
        labels={
            "country": "Country",
            "outbreak_probability":
                "Outbreak Probability (%)",
            "outbreak_risk_level":
                "Risk Level",
        },
    )

    fig_probability.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0D1B2E",
        plot_bgcolor="#0D1B2E",
    )

    st.plotly_chart(
        fig_probability,
        use_container_width=True,
    )


# ============================================================
# TIME SERIES
# ============================================================

st.markdown(
    '<div class="section-title">📉 Disease Time Series</div>',
    unsafe_allow_html=True,
)

if selected_country != "All Countries":

    ts_country = risk_df[
        risk_df["country_code"]
        == selected_country
    ].sort_values("year")

else:

    ts_country = (
        risk_df.groupby("year", as_index=False)
        ["cases"]
        .sum()
    )

if not ts_country.empty:

    fig_ts = px.line(
        ts_country,
        x="year",
        y="cases",
        markers=True,
        title=(
            f"{selected_disease.title()} cases over time"
        ),
        labels={
            "year": "Year",
            "cases": "Cases",
        },
    )

    fig_ts.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0D1B2E",
        plot_bgcolor="#0D1B2E",
    )

    st.plotly_chart(
        fig_ts,
        use_container_width=True,
    )


# ============================================================
# EXPLAINABLE PREDICTION
# ============================================================

st.markdown(
    '<div class="section-title">🧠 Explainable Prediction</div>',
    unsafe_allow_html=True,
)

if selected_country != "All Countries":

    country_rows = risk_df[
        risk_df["country_code"]
        == selected_country
    ].sort_values("year")

    if not country_rows.empty:

        explanation_row = country_rows.iloc[-1]

        feature_columns = list(
            X_test.columns
        )

        try:

            explanation = build_explanation(
                best_model,
                explanation_row,
                feature_columns,
                top_n=5,
            )

            if not explanation.empty:

                st.dataframe(
                    explanation[
                        [
                            "label",
                            "value",
                            "direction",
                            "explanation",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as error:

            st.warning(
                f"Explanation unavailable: {error}"
            )

else:

    st.info(
        "Select a country from the sidebar "
        "to view explainable prediction factors."
    )


# ============================================================
# MODEL COMPARISON
# ============================================================

st.markdown(
    '<div class="section-title">🤖 ML Model Comparison</div>',
    unsafe_allow_html=True,
)

comparison_display = comparison.copy()

for column in [
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "roc_auc",
]:

    if column in comparison_display.columns:

        comparison_display[column] = (
            comparison_display[column]
            .round(4)
        )

st.dataframe(
    comparison_display,
    use_container_width=True,
    hide_index=True,
)

st.success(
    f"Best model: {best_name}"
)


# ============================================================
# ANOMALY TABLE
# ============================================================

st.markdown(
    '<div class="section-title">⚠️ Detected Anomalies</div>',
    unsafe_allow_html=True,
)

anomaly_df = risk_df[
    risk_df["is_anomaly"] == True
].copy()

anomaly_df = anomaly_df.sort_values(
    [
        "year",
        "anomaly_score",
    ],
    ascending=[
        False,
        True,
    ],
)

if selected_country != "All Countries":

    anomaly_df = anomaly_df[
        anomaly_df["country_code"]
        == selected_country
    ]

anomaly_display = anomaly_df[
    [
        "country_code",
        "year",
        "cases",
        "anomaly_score",
        "outbreak_probability",
        "outbreak_risk_level",
        "trend_direction",
    ]
].head(30).copy()

anomaly_display["country"] = (
    anomaly_display["country_code"]
    .apply(country_name)
)

anomaly_display = anomaly_display[
    [
        "country",
        "year",
        "cases",
        "anomaly_score",
        "outbreak_probability",
        "outbreak_risk_level",
        "trend_direction",
    ]
]

anomaly_display = anomaly_display.rename(
    columns={
        "year": "Year",
        "cases": "Cases",
        "anomaly_score": "Anomaly Score",
        "outbreak_probability": "Outbreak Probability (%)",
        "outbreak_risk_level": "Risk Level",
        "trend_direction": "Trend",
    }
)

st.dataframe(
    anomaly_display,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Powered Global Health Intelligence System | "
    "WHO disease surveillance data | "
    "Anomaly detection | Time-series analysis | "
    "Machine learning | Explainable predictions"
)