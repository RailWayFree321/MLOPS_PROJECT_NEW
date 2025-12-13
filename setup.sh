#!/bin/bash
# Quick startup script for the MLOps project
# This script helps start all services

set -e

PROJECT_ROOT="/home/sarmad/Study/MLOPs/Proj"
VENV="$PROJECT_ROOT/venv"
AIRFLOW_HOME="$PROJECT_ROOT/airflow_home"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== MLOps Marketing Creative Project ===${NC}"
echo ""

# Check if venv exists
if [ ! -d "$VENV" ]; then
    echo -e "${YELLOW}Virtual environment not found. Creating...${NC}"
    python3 -m venv "$VENV"
    echo -e "${GREEN}Virtual environment created${NC}"
fi

# Activate venv
source "$VENV/bin/activate"
export PYTHONPATH="$PROJECT_ROOT"

echo -e "${GREEN}✓ Environment activated${NC}"
echo ""

# Check if dataset version file exists
if [ ! -f "$PROJECT_ROOT/data/dataset_version.txt" ]; then
    echo -e "${YELLOW}Initializing dataset version...${NC}"
    mkdir -p "$PROJECT_ROOT/data"
    echo "1" > "$PROJECT_ROOT/data/dataset_version.txt"
    echo -e "${GREEN}✓ Dataset version initialized${NC}"
fi

echo ""
echo -e "${BLUE}Available services:${NC}"
echo "1. MLflow Server (port 5000)"
echo "2. Data Generation (port 5000)"
echo "3. Training Pipeline"
echo "4. Inference API (port 5001)"
echo "5. Airflow Setup"
echo "6. Run Tests"
echo ""
echo -e "${YELLOW}To start a service, run:${NC}"
echo "  Option 1: mlflow server --host 0.0.0.0 --port 5000"
echo "  Option 2: python -m src.data_generation"
echo "  Option 3: python -m src.train"
echo "  Option 4: python -m src.inference"
echo "  Option 5: airflow db init"
echo "  Option 6: pytest tests/ -v"
echo ""
echo -e "${GREEN}Virtual environment is ready!${NC}"
echo "Run any of the commands above to start a service."
