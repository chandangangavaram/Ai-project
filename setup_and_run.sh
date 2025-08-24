#!/bin/bash

# Exit on error
set -e

# Create virtual environment if not exists
if [ ! -d "rag_env" ]; then
  echo "[INFO] Creating virtual environment..."
  python3 -m venv rag_env
fi

# Activate venv
echo "[INFO] Activating virtual environment..."
source rag_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
echo "[INFO] Installing dependencies..."
pip install -r requirements.txt

# Ingest documents
echo "[INFO] Running document ingestion..."
python ingest.py

# Start Streamlit app
echo "[INFO] Launching Streamlit UI..."
streamlit run app.py

