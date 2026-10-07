# Telco Customer Churn

An end-to-end machine-learning project that predicts whether a telecom customer is likely to churn. The project includes model training, MLflow model artifacts, a Streamlit web UI, an optional FastAPI API, Docker packaging, and GitHub Actions deployment to Render.

## Project Flow

```text
Raw CSV
  -> validation
  -> preprocessing
  -> feature engineering
  -> XGBoost training
  -> MLflow model artifact
  -> Streamlit/Docker/Render serving
```

The deployed application uses the bundled MLflow model under `src/serving/model`. The serving code transforms incoming customer data, aligns it with the saved feature schema, and returns either `Likely to churn` or `Not likely to churn`.

## Features

- XGBoost churn classification.
- Deterministic binary and one-hot feature encoding.
- MLflow model and metric tracking.
- Streamlit UI with saved customer examples.
- Optional FastAPI health and prediction endpoints.
- Docker image designed for Render.
- GitHub Actions workflow that checks Python files, publishes the image to Docker Hub, and triggers Render.

## Repository Structure

```text
.
|-- .github/workflows/ci.yml       # Build, publish, and deploy workflow
|-- dockerfile                     # Streamlit production image
|-- requirements.txt               # Direct project dependencies
|-- scripts/
|   |-- run_pipeline.py            # Complete training pipeline
|   |-- prepare_processed_data.py  # Prepare processed training data
|   |-- test_fastapi.py            # Manual API request test
|   |-- test_pipeline_phase1_data_features.py
|   `-- test_pipeline_phase2_modeling.py
|-- src/
|   |-- app/
|   |   |-- streamlit_app.py       # Main web UI
|   |   `-- main.py                # Optional FastAPI application
|   |-- data/                      # Loading and preprocessing
|   |-- features/                  # Feature engineering
|   |-- models/                    # Reusable training/evaluation helpers
|   |-- serving/
|   |   |-- inference.py           # Model loading and prediction
|   |   `-- model/                 # Bundled MLflow model artifacts
|   `-- utils/validate_data.py     # Training data validation
|-- notebooks/EDA.ipynb            # Exploratory analysis
`-- CLAUDE.md                     # Additional project notes
```

## Requirements

- Python 3.11 is recommended because the bundled model was created with Python 3.11.
- Docker Desktop is required for container testing.
- A raw dataset is required to retrain the model. Place it at:

```text
data/raw/Telco-Customer-Churn.csv
```

The dataset and generated training outputs are excluded from Git by the repository ignore rules. The bundled serving model is already included under `src/serving/model`.

## Install Locally

From the project root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the Streamlit UI

```powershell
streamlit run src/app/streamlit_app.py
```

Open `http://localhost:8501`.

The UI includes saved examples so you can test different customer profiles without entering every field manually. Its health endpoint is:

```text
http://localhost:8501/_stcore/health
```

Expected response: `ok`.

## Optional FastAPI API

Start the API:

```powershell
python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

Health check:

```powershell
Invoke-WebRequest http://localhost:8000/
```

Prediction request:

```powershell
$body = @{
    gender = "Male"
    SeniorCitizen = 0
    Partner = "Yes"
    Dependents = "No"
    PhoneService = "Yes"
    MultipleLines = "No"
    InternetService = "Fiber optic"
    OnlineSecurity = "No"
    OnlineBackup = "Yes"
    DeviceProtection = "No"
    TechSupport = "No"
    StreamingTV = "Yes"
    StreamingMovies = "Yes"
    Contract = "Month-to-month"
    PaperlessBilling = "Yes"
    PaymentMethod = "Electronic check"
    tenure = 5
    MonthlyCharges = 70.35
    TotalCharges = 350.75
} | ConvertTo-Json

Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -ContentType "application/json" -Body $body
```

## Train the Model

Place the raw CSV at `data/raw/Telco-Customer-Churn.csv`, then run:

```powershell
python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --target Churn
```

The pipeline validates data, preprocesses it, builds features, trains XGBoost, evaluates precision/recall/F1/ROC-AUC, and logs the model and feature metadata to MLflow.

Generated files are written to `data/processed`, `artifacts`, and `mlruns`. These are training outputs and are not required to run the bundled serving image.

## Docker

Build the image:

```powershell
docker build -t telco-churn-test -f dockerfile .
```

Run it:

```powershell
docker run --rm --name telco-churn-test -p 8501:8501 telco-churn-test
```

Open `http://localhost:8501` and check `http://localhost:8501/_stcore/health`.

The Dockerfile uses Python 3.11, copies the selected MLflow model to `/app/model`, and starts Streamlit using Render's `PORT` environment variable. Locally, it defaults to port 8501.

## Tests and Checks

The repository currently contains manual scripts rather than a pytest test suite:

```powershell
python scripts/test_pipeline_phase1_data_features.py
python scripts/test_pipeline_phase2_modeling.py
python scripts/test_fastapi.py
```

The phase-2 script runs an Optuna study and can take a long time. The FastAPI script requires the API to already be running on port 8000.

GitHub Actions also performs this Python compilation check:

```powershell
python -m compileall -q src scripts
```

## CI/CD

The workflow at `.github/workflows/ci.yml` runs on every push to `main`:

```text
Push to main
  -> compile Python files
  -> build Docker image
  -> push latest and commit SHA tags to Docker Hub
  -> trigger Render deploy hook
```

The workflow publishes:

```text
<dockerhub-username>/telco-churn:latest
<dockerhub-username>/telco-churn:<commit-sha>
```

Add these GitHub repository secrets under **Settings > Secrets and variables > Actions**:

```text
DOCKERHUB_USERNAME
DOCKERHUB_TOKEN
RENDER_DEPLOY_HOOK_URL
```

`DOCKERHUB_TOKEN` should be a Docker Hub personal access token with permission to push images. `RENDER_DEPLOY_HOOK_URL` is created in the Render service settings.

## Render Deployment

1. Create a Docker Hub repository named `telco-churn`.
2. Create a Render Web Service using **Existing Image**.
3. Configure the image as `docker.io/<your-dockerhub-username>/telco-churn:latest`.
4. Use port `8501`.
5. Use health check path `/_stcore/health`.
6. Add the Render deploy hook URL to GitHub as `RENDER_DEPLOY_HOOK_URL`.
7. Push to `main` and monitor the workflow under the GitHub **Actions** tab.

Render pulls the updated `latest` image after the workflow triggers the deploy hook.

## Model Metrics

The bundled MLflow run stores:

```text
Precision: 0.490
Recall: 0.821
F1: 0.614
ROC-AUC: 0.837
```

Accuracy was not logged for this bundled run. The relatively high recall means the model is tuned to catch more potential churners, which can produce more false positives.

## Notes

- Training and serving must use the same feature transformations and feature order.
- The serving model is loaded from the bundled MLflow artifact and aligned using `feature_columns.txt`.
- Do not commit Docker Hub tokens, Render deploy-hook URLs, or other secrets.
