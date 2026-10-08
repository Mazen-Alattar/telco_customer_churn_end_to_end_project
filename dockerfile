# 1. Use the official lightweight Python base image
FROM python:3.11-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Copy only dependency file first (for Docker caching)
COPY requirements.txt .

# 4. Install Python dependencies (add curl if you use MLflow local tracking URI)
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 5. Copy the entire project into the image
COPY . .

# Explicitly copy the bundled model artifacts used by inference.py.
COPY src/serving/model /app/src/serving/model

# Copy the selected MLflow model and feature schema to the runtime path.
COPY src/serving/model/d798a256ed974d67b3e1f85f2ef29da8/artifacts/model /app/model
COPY src/serving/model/d798a256ed974d67b3e1f85f2ef29da8/artifacts/feature_columns.txt /app/model/feature_columns.txt

# Make the source package importable and show logs immediately.
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

# Render provides PORT; 8501 is the local default.
EXPOSE 8501

# Run the Streamlit application on Render's assigned port.
CMD ["sh", "-c", "streamlit run src/app/streamlit_app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
