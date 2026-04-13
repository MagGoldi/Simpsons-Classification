#!/bin/bash
# Launch MLflow tracking UI
# URL: http://localhost:5001
# Docs: https://mlflow.org/docs/latest/tracking.html

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

uv run mlflow ui \
    --backend-store-uri "sqlite:///${PROJECT_ROOT}/mlflow.db" \
    --port 5001 \
    --host 0.0.0.0
