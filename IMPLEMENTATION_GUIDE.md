# MLOps Project - Implementation Guide

## Project Overview

This is a comprehensive MLOps project that builds a **Marketing Creative Generation System** for e-commerce products. It demonstrates a complete machine learning lifecycle with production-grade infrastructure using modern MLOps practices.

### Core Components

1. **Data Generation Service** - Flask app that generates synthetic product data
2. **Model Training Pipeline** - DistilBERT-based model for creative quality classification
3. **MLflow Integration** - Experiment tracking and model registry
4. **Airflow Orchestration** - Automated pipeline scheduling and execution
5. **Inference Service** - Flask API for model serving (Kubernetes-ready)
6. **Comprehensive Testing** - Unit tests for all components

---

## Project Structure

```
/home/sarmad/Study/MLOPs/Proj/
├── config/                      # Configuration files
│   └── config.yaml             # Main project configuration
├── src/                         # Source code
│   ├── data_generation.py      # Data generation Flask app
│   ├── train.py                # Model training pipeline
│   ├── utils.py                # Utility functions
│   ├── mlflow_registry.py      # MLflow integration
│   └── inference.py            # Inference Flask app
├── dags/                        # Airflow DAGs
│   └── marketing_creative_dag.py # Training pipeline DAG
├── tests/                       # Test files
│   ├── test_data_generation.py
│   ├── test_train.py
│   ├── test_utils.py
│   ├── test_mlflow_registry.py
│   └── conftest.py
├── data/                        # Data storage
│   └── training_data.csv       # Training dataset
├── models/                      # Model storage
│   └── best_model/            # Trained model directory
├── mlflow_artifacts/            # MLflow artifacts
├── logs/                        # Application logs
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables
└── IMPLEMENTATION_GUIDE.md      # This file
```

---

## Prerequisites

### System Requirements

- **OS**: Linux (tested on Ubuntu 20.04+)
- **Python**: 3.9+
- **Memory**: Minimum 8GB RAM (for model training)
- **Disk Space**: 5GB+ (for models and artifacts)

### Software Requirements

- Git
- Docker (for containerization)
- Docker Compose (optional, for local testing)
- pip (Python package manager)

---

## Installation & Setup

### Step 1: Clone/Setup Project

```bash
cd /home/sarmad/Study/MLOPs/Proj
git init
git remote add origin https://github.com/RailWayFree321/MLOPS_PROJECT_NEW.git
```

### Step 2: Create Python Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# OR
venv\Scripts\activate  # On Windows
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: PyTorch CPU version is included. If you have NVIDIA GPU and need CUDA support, modify `requirements.txt` accordingly.

### Step 4: Setup Environment Variables

The `.env` file is already configured with default values:

```env
FLASK_ENV=development
MLFLOW_TRACKING_URI=http://localhost:5000
DATA_GENERATION_URL=http://localhost:5000
INFERENCE_PORT=5001
PYTHONPATH=/home/sarmad/Study/MLOPs/Proj
```

Modify these if your setup differs.

### Step 5: Create Initial Dataset Version

```bash
mkdir -p data
echo "1" > data/dataset_version.txt
```

---

## Running the Components

### 1. Start MLflow Tracking Server

MLflow provides experiment tracking and model registry.

```bash
# In terminal 1
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlflow_artifacts
```

The MLflow UI will be available at: `http://localhost:5000`

### 2. Start Data Generation Flask App

```bash
# In terminal 2
source venv/bin/activate
python -m src.data_generation
```

Verify it works:
```bash
curl http://localhost:5000/generate_data
```

Should return 50 random product records.

### 3. Run Training Pipeline Manually

```bash
# In terminal 3
source venv/bin/activate
python -m src.train
```

This will:
- Load training data
- Train the DistilBERT model
- Log metrics to MLflow
- Save the model locally
- Check if accuracy is better than previous versions

### 4. Start Inference Service

```bash
# In terminal 4
source venv/bin/activate
export MODEL_PATH=/home/sarmad/Study/MLOPs/Proj/models/best_model
python -m src.inference
```

The inference API will be available at: `http://localhost:5001`

Test the inference endpoint:
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Premium Wireless Headphones",
    "description": "High-quality headphones with noise cancellation"
  }'
```

### 5. Setup and Run Airflow

#### Option A: Quick Setup (Development)

```bash
# In terminal 5
source venv/bin/activate

# Initialize Airflow database
export AIRFLOW_HOME=/home/sarmad/Study/MLOPs/Proj/airflow_home
mkdir -p $AIRFLOW_HOME
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com

# Copy DAG file
cp dags/marketing_creative_dag.py $AIRFLOW_HOME/dags/

# Start Airflow scheduler
airflow scheduler
```

In another terminal:
```bash
# Start Airflow webserver
export AIRFLOW_HOME=/home/sarmad/Study/MLOPs/Proj/airflow_home
airflow webserver --port 8080
```

#### Option B: Using Docker (Production-like)

```bash
# Create docker-compose.yml for Airflow (if needed)
# For now, use the manual setup above
```

**Airflow UI** will be available at: `http://localhost:8080`

- Username: `admin`
- Password: (set during user creation)

#### Trigger the DAG

1. Go to Airflow UI
2. Find `marketing_creative_pipeline` DAG
3. Click the trigger button
4. Monitor execution in the UI

Or via CLI:
```bash
export AIRFLOW_HOME=/home/sarmad/Study/MLOPs/Proj/airflow_home
airflow dags trigger marketing_creative_pipeline
```

---

## Testing

### Run All Tests

```bash
source venv/bin/activate
pytest tests/ -v
```

### Run Specific Test Module

```bash
# Test data generation
pytest tests/test_data_generation.py -v

# Test training pipeline
pytest tests/test_train.py -v

# Test utilities
pytest tests/test_utils.py -v

# Test MLflow registry
pytest tests/test_mlflow_registry.py -v
```

### Generate Coverage Report

```bash
pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

---

## API Endpoints

### Data Generation Service (Port 5000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate_data` | GET | Generate 50 random product records |
| `/health` | GET | Health check |

**Example**:
```bash
curl http://localhost:5000/generate_data | python -m json.tool
```

### Inference Service (Port 5001)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Generate creative for single product |
| `/batch_predict` | POST | Generate creatives for multiple products |
| `/health` | GET | Health check |
| `/ready` | GET | Kubernetes readiness probe |
| `/metrics` | GET | Prometheus metrics |
| `/info` | GET | Model information |

**Example - Single Prediction**:
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Premium Laptop",
    "description": "15-inch laptop with SSD storage"
  }'
```

**Example - Batch Prediction**:
```bash
curl -X POST http://localhost:5001/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "products": [
      {
        "title": "Laptop",
        "description": "15-inch laptop"
      },
      {
        "title": "Headphones",
        "description": "Wireless headphones"
      }
    ]
  }'
```

---

## Configuration Details

### config/config.yaml

Key configuration sections:

```yaml
# Data Configuration
data:
  num_records: 50              # Number of synthetic products
  output_path: "data/training_data.csv"
  version_file: "data/dataset_version.txt"

# Model Configuration
model:
  model_type: "distilbert"
  pretrained_model: "distilbert-base-uncased"
  epochs: 3
  batch_size: 16
  learning_rate: 2e-5

# MLflow Configuration
mlflow:
  tracking_uri: "http://localhost:5000"
  experiment_name: "marketing_creative_generator"

# Airflow Configuration
airflow:
  dag_id: "marketing_creative_pipeline"
  schedule_interval: "0 0 * * *"  # Daily
```

---

## Workflow Explanation

### Training Pipeline (Orchestrated by Airflow)

```
1. Generate Data
   ↓
2. Save Data & Increment Version
   ↓
3. Train Model
   ↓
4. Check & Register Model (if improved)
   ↓
5. Pipeline Summary
```

### Key Features

1. **Automatic Data Generation**: Generates synthetic product data on demand
2. **Model Versioning**: Each training run increments dataset version
3. **Experiment Tracking**: All training runs logged in MLflow
4. **Model Registry**: Only best models registered (based on accuracy)
5. **Inference Service**: Always serves the best registered model
6. **Comprehensive Logging**: All components produce detailed logs

---

## Monitoring & Debugging

### Check Logs

```bash
# Training pipeline logs
tail -f logs/pipeline.log

# Airflow logs
export AIRFLOW_HOME=/home/sarmad/Study/MLOPs/Proj/airflow_home
tail -f $AIRFLOW_HOME/logs/marketing_creative_pipeline/*
```

### MLflow Dashboard

Visit `http://localhost:5000` to:
- View all experiment runs
- Compare model metrics
- Access registered models
- Download model artifacts

### Airflow Dashboard

Visit `http://localhost:8080` to:
- Monitor DAG execution
- View task logs
- Track pipeline status

---

## Troubleshooting

### Issue: Model training is slow

**Solution**: 
- Use CPU-only PyTorch (already configured)
- Reduce epoch count in config.yaml
- Use smaller batch size

### Issue: MLflow server not starting

**Solution**:
```bash
# Kill existing process
lsof -i :5000
kill -9 <PID>

# Start MLflow again
mlflow server --host 0.0.0.0 --port 5000
```

### Issue: Airflow DAG not showing up

**Solution**:
```bash
# Ensure DAG file is in correct location
export AIRFLOW_HOME=/home/sarmad/Study/MLOPs/Proj/airflow_home
cp dags/marketing_creative_dag.py $AIRFLOW_HOME/dags/

# Check for syntax errors
python -m py_compile dags/marketing_creative_dag.py

# Restart scheduler
```

### Issue: Import errors in Python

**Solution**:
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH=/home/sarmad/Study/MLOPs/Proj

# Or activate virtual environment properly
source venv/bin/activate
```

---

## Next Steps (For Future Parts)

### Part 3: Docker & CI/CD

- [ ] Create Dockerfile for each service
- [ ] Build Docker images
- [ ] Push to DockerHub via GitHub Actions
- [ ] Setup GitHub Actions CI/CD pipeline

### Part 4: Kubernetes Deployment

- [ ] Create Kubernetes manifests
- [ ] Setup load balancing
- [ ] Configure ingress controller
- [ ] Deploy via GitHub Actions

### Part 5: Monitoring & Observability

- [ ] Deploy Prometheus
- [ ] Deploy Grafana
- [ ] Create custom metrics dashboards
- [ ] Monitor inference latency and model drift

---

## Development Notes

### Adding New Features

1. **New Data Source**: Modify `src/data_generation.py`
2. **New Model**: Update `src/train.py` and config
3. **New Metrics**: Modify `src/mlflow_registry.py`
4. **New API Endpoint**: Add to `src/inference.py`
5. **Add Tests**: Create corresponding test file

### Code Standards

- Use type hints for all functions
- Add docstrings to all classes/functions
- Follow PEP 8 style guide
- Write tests for new features
- Keep logs detailed but concise

---

## Support & Contact

For questions or issues:
1. Check troubleshooting section
2. Review logs in `logs/` directory
3. Check Airflow/MLflow dashboards
4. Review test cases for examples

---

## Version Info

- **Project Version**: 1.0.0
- **Last Updated**: January 2024
- **Airflow Version**: 2.7.0
- **Python**: 3.9+

---

**End of Implementation Guide**
