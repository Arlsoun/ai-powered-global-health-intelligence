\# AI-Powered Global Health Intelligence System



A global health surveillance and intelligence system for monitoring infectious diseases, detecting unusual disease patterns, estimating outbreak risk, and providing country-level health insights.



Live application: https://ai-powered-global-health-intelligence-app.streamlit.app/



GitHub repository: https://github.com/Arlsoun/ai-powered-global-health-intelligence



\## Project Overview



The AI-Powered Global Health Intelligence System combines public health data, data preprocessing, time-series analysis, anomaly detection, outbreak-risk scoring, machine learning, explainable predictions, and interactive visualization.



The system retrieves disease observations from the WHO Global Health Observatory and transforms the data into country-level intelligence.



The application provides both an interactive Streamlit dashboard and a FastAPI service.



\## Supported Diseases



The system currently supports:



\* Malaria

\* Cholera

\* Tuberculosis

\* Measles

\* Meningitis



\## Main Features



\### Disease Surveillance



Retrieves infectious-disease observations and organizes them by country and year.



\### Data Preprocessing



Cleans and transforms raw WHO observations into a consistent structure suitable for analysis and machine learning.



\### Time-Series Analysis



Analyzes historical disease cases to identify changes, trends, and long-term patterns.



\### Anomaly Detection



Identifies unusual observations in disease-case patterns.



\### Outbreak-Risk Prediction



Calculates an outbreak probability for country-year observations.



Risk levels are classified as:



\* LOW

\* MEDIUM

\* HIGH



\### Dashboard Filters



The Streamlit dashboard provides filtering by:



\* Disease

\* Country

\* Year



Countries are displayed using full country names where available.



\### Warning and Alert Indicators



The dashboard provides global and country-level warnings based on calculated outbreak risk.



\### Machine Learning



The system compares three classification models:



\* Logistic Regression

\* Random Forest

\* Gradient Boosting



Models are evaluated using:



\* Accuracy

\* Precision

\* Recall

\* F1 Score

\* ROC AUC



\### Explainable Predictions



The system provides feature-level information to help explain selected country-level risk predictions.



\## System Architecture



The project consists of four main layers.



\### 1. Data Layer



Retrieves disease observations from the WHO Global Health Observatory API.



\### 2. Analysis and Machine Learning Layer



Performs:



\* Data preprocessing

\* Time-series analysis

\* Anomaly detection

\* Outbreak-risk scoring

\* Machine learning

\* Model evaluation

\* Explainability



\### 3. Application Layer



Provides the interactive Streamlit dashboard.



\### 4. API Layer



Provides FastAPI endpoints for programmatic access to the health intelligence pipeline.



\## Project Structure



```text

ai-powered-global-health-intelligence/

│

├── api/

│   └── main.py

│

├── app/

│   ├── api.py

│   └── dashboard.py

│

├── data/

│   └── raw/

│

├── docs/

│   └── PROJECT\_DOCUMENTATION.md

│

├── models/

│

├── notebooks/

│

├── src/

│   ├── anomaly\_detection.py

│   ├── data\_loader.py

│   ├── explainability.py

│   ├── ml\_models.py

│   ├── outbreak\_risk.py

│   ├── preprocessing.py

│   └── time\_series.py

│

├── tests/

│   └── test\_preprocessing.py

│

├── .streamlit/

│   └── config.toml

│

├── .gitignore

├── requirements.txt

└── README.md

```



\## Technologies



The project uses:



\* Python

\* Pandas

\* NumPy

\* Scikit-learn

\* Streamlit

\* FastAPI

\* Uvicorn

\* Plotly

\* Requests

\* Pycountry

\* Pytest

\* WHO Global Health Observatory API



\## Running Locally



\### 1. Clone the repository



```bash

git clone https://github.com/Arlsoun/ai-powered-global-health-intelligence.git

cd ai-powered-global-health-intelligence

```



\### 2. Create a virtual environment



```bash

python -m venv .venv

```



\### 3. Activate the environment



Windows PowerShell:



```powershell

.venv\\Scripts\\Activate.ps1

```



\### 4. Install dependencies



```bash

pip install -r requirements.txt

```



\### 5. Run the Streamlit dashboard



```bash

streamlit run app/dashboard.py

```



The dashboard will open locally in the browser.



\## Running the API



Start the FastAPI service with:



```bash

python -m uvicorn app.api:app --reload

```



The local API is available at:



```text

http://127.0.0.1:8000

```



\## API Endpoints



\### List supported diseases



```http

GET /diseases

```



Example response:



```json

{

&#x20; "diseases": \[

&#x20;   "malaria",

&#x20;   "cholera",

&#x20;   "tuberculosis",

&#x20;   "measles",

&#x20;   "meningitis"

&#x20; ]

}

```



\### Analyze a disease



```http

POST /analyze

```



Example request:



```json

{

&#x20; "disease": "measles"

}

```



The endpoint returns processed country-level observations including outbreak probability, risk level, and trend information.



\## Testing



Run the test suite with:



```bash

pytest

```



\## Deployment



The Streamlit dashboard is deployed through Streamlit Community Cloud.



Live application:



https://ai-powered-global-health-intelligence-app.streamlit.app/



The source code is hosted on GitHub.



\## Data Source



Disease observations are retrieved from the World Health Organization Global Health Observatory API.



The system processes the retrieved observations locally before performing analysis and prediction.



\## Documentation



Detailed project documentation is available in:



```text

docs/PROJECT\_DOCUMENTATION.md

```



\## Project Status



The project currently includes:



\* Disease surveillance

\* WHO data integration

\* Data preprocessing

\* Time-series analysis

\* Anomaly detection

\* Outbreak-risk scoring

\* Machine learning

\* Model comparison

\* Model evaluation

\* Explainable predictions

\* Dashboard filtering

\* Warning and alert indicators

\* FastAPI service

\* Automated tests

\* Project documentation

\* GitHub version control

\* Streamlit Cloud deployment



\## Author



Aliyu Abubakar



AI-Powered Global Health Intelligence System



