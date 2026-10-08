# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Training Pipeline
```bash
# Run the complete ML training pipeline
python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --target Churn

# Prepare processed data only
python scripts/prepare_processed_data.py
```

### Testing
```bash
# Test data processing and feature engineering
python scripts/test_pipeline_phase1_data_features.py

# Test model training and evaluation
python scripts/test_pipeline_phase2_modeling.py

# Test FastAPI endpoints
python scripts/test_fastapi.py
```

### Local Development
```bash
# Run the Streamlit application locally
streamlit run src/app/streamlit_app.py

# Optional API entry point
python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

### Docker
```bash
# Build and run the containerized application
docker build -t telco-churn-app .
docker run -p 8501:8501 telco-churn-app
```

## Architecture Overview

### ML Pipeline Flow
This project implements a complete MLOps pipeline with two distinct phases:

**Training Pipeline** (`scripts/run_pipeline.py`):
1. **Data Loading** → **Data Validation** → **Preprocessing** → **Feature Engineering** → **XGBoost Training** → **MLflow Logging**
2. All artifacts (model, feature columns, preprocessing logic) are stored in MLflow for reproducibility

**Serving Pipeline** (`src/app/streamlit_app.py` + `src/serving/inference.py`):
1. **Streamlit Web UI** → **MLflow Model Loading** → **Feature Transformation** → **Prediction**
2. Feature processing mirrors training-time transformations for consistency

### MLflow Integration Patterns
- **Experiment Name**: "Telco Churn" (default, can be overridden)
- **Tracking URI**: File-based at `{project_root}/mlruns`
- **Logged Artifacts**: `model/` and `feature_columns.txt`
- **Tracked Metrics**: precision, recall, f1, roc_auc, train_time, pred_time, data_quality_pass
- **Parameters**: model type, threshold (default 0.35), test_size (default 0.2)

### Feature Engineering Consistency
Critical pattern: Training and serving must use identical feature transformations.

**Training** (`src/features/build_features.py`):
- Binary features (Yes/No, Male/Female) → deterministic 0/1 mapping
- Multi-category features → one-hot encoding with `drop_first=True`
- Boolean columns → integers

**Serving** (`src/serving/inference.py`):
- Uses fixed `BINARY_MAP` dictionary for consistent binary encoding
- Applies `pd.get_dummies()` with same parameters as training
- Feature alignment via `FEATURE_COLS` from training artifacts

### Model Loading and Serving
- **Container Path**: Model loaded from `/app/model` (MLflow pyfunc format)
- **Feature Order**: Enforced using `feature_columns.txt` from training
- **Prediction Format**: Returns "Likely to churn" or "Not likely to churn" strings

### Data Validation
- **Tool**: pandas-based validation checks
- **Location**: `src/utils/validate_data.py`
- **Checks**: CustomerID presence, gender values, numeric ranges for tenure/charges
- **Integration**: Results logged to MLflow as `data_quality_pass` metric

### Docker and Render Containerization
- **Base Image**: `python:3.11-slim`
- **Key Setting**: `PYTHONPATH=/app/src` for proper module imports
- **Model Artifacts**: Specific MLflow run copied to `/app/model` during build
- **Serving**: Streamlit on Render's `PORT` environment variable, defaulting to 8501 locally

### CI/CD Pipeline
- **Trigger**: Push to main branch
- **Actions**: Build Docker image → Push to Docker Hub → optionally trigger Render deploy hook
- **Requirements**: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`, and optional `RENDER_DEPLOY_HOOK_URL` secrets
- **Deployment**: Render Web Service configured to run the Docker Hub image

## Key Implementation Details

### XGBoost Model Configuration
Optimized hyperparameters are hardcoded in `scripts/run_pipeline.py:100-110`:
- `n_estimators=301`, `learning_rate=0.034`, `max_depth=7`
- `scale_pos_weight` calculated dynamically for class imbalance handling

### API Endpoints
- `GET /` - Health check returning `{"status": "ok"}`
- `POST /predict` - Accepts `CustomerData` Pydantic model with 18 customer attributes
- Streamlit UI - served by `src/app/streamlit_app.py`

### File System Layout
- `data/raw/` - Original datasets
- `data/processed/` - Cleaned datasets
- `mlruns/` - MLflow experiment tracking database
- `artifacts/` - Shared feature metadata (`feature_columns.json`)
- `src/serving/model/` - Local MLflow run copies for development

### Development Notes
- No formal test suite exists; use manual test scripts in `scripts/test_*.py`
- MLflow UI can be accessed with: `mlflow ui --backend-store-uri file:./mlruns`
- The project uses file-based MLflow tracking (not a tracking server)
- Model serving expects exact feature column order from training time