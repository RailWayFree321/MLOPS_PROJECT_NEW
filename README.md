# Marketing Creative Generator - MLOps Project

A production-grade MLOps system for automatically generating marketing creatives for e-commerce products using DistilBERT and modern ML infrastructure.

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- 8GB+ RAM
- Linux/Mac (or WSL on Windows)

### Installation (5 minutes)

```bash
# 1. Clone repository
cd /home/sarmad/Study/MLOPs/Proj

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialize dataset version
mkdir -p data
echo "1" > data/dataset_version.txt
```

## 📋 Components

| Component | Port | Purpose |
|-----------|------|---------|
| **MLflow Server** | 5000 | Experiment tracking & model registry |
| **Data Generation** | 5000 | Generates synthetic product data |
| **Inference API** | 5001 | Serves model predictions |
| **Airflow Web** | 8080 | Pipeline orchestration UI |

## 🏃 Running the Project

### Terminal 1: Start MLflow
```bash
mlflow server --host 0.0.0.0 --port 5000
```

### Terminal 2: Start Data Generation Service
```bash
source venv/bin/activate
python -m src.data_generation
```

### Terminal 3: Run Training (Manual or via Airflow)
```bash
source venv/bin/activate
python -m src.train
```

### Terminal 4: Start Inference API
```bash
source venv/bin/activate
python -m src.inference
```

### Terminal 5: Setup & Run Airflow
```bash
source venv/bin/activate
export AIRFLOW_HOME=/home/sarmad/Study/MLOPs/Proj/airflow_home
airflow db init
airflow users create --username admin --role Admin --email admin@example.com
cp dags/marketing_creative_dag.py $AIRFLOW_HOME/dags/
airflow scheduler
```

### Terminal 6: Airflow Web UI
```bash
export AIRFLOW_HOME=/home/sarmad/Study/MLOPs/Proj/airflow_home
airflow webserver --port 8080
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_data_generation.py -v
```

## 📡 API Examples

### Generate Data
```bash
curl http://localhost:5000/generate_data
```

### Single Prediction
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Premium Wireless Headphones",
    "description": "High-quality headphones with noise cancellation and 20-hour battery"
  }'
```

### Batch Predictions
```bash
curl -X POST http://localhost:5001/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
    "products": [
      {
        "title": "Laptop",
        "description": "15-inch laptop with SSD"
      },
      {
        "title": "Headphones",
        "description": "Wireless headphones"
      }
    ]
  }'
```

## 📊 Dashboards

- **MLflow UI**: http://localhost:5000
  - View experiments and runs
  - Compare models
  - Access model registry

- **Airflow UI**: http://localhost:8080
  - Monitor DAG execution
  - View task logs
  - Trigger pipelines manually

## 📁 Project Structure

```
src/
├── data_generation.py      # Data generation Flask app
├── train.py                # Model training pipeline
├── inference.py            # Inference Flask API
├── mlflow_registry.py      # Model registry integration
└── utils.py                # Utility functions

dags/
└── marketing_creative_dag.py  # Airflow orchestration DAG

tests/
├── test_data_generation.py
├── test_train.py
├── test_utils.py
├── test_mlflow_registry.py
└── conftest.py

config/
└── config.yaml             # Configuration file
```

## ⚙️ Configuration

Edit `config/config.yaml` to customize:
- Model parameters (epochs, batch size, learning rate)
- MLflow tracking URI
- Airflow schedule interval
- Data paths and versions

## 📚 Documentation

- **IMPLEMENTATION_GUIDE.md** - Complete setup and usage guide
- **STATUS.md** - Project completion status and roadmap

## 🔧 Troubleshooting

### Model training is slow
- Reduce epochs in config.yaml
- Reduce batch size
- Use smaller dataset

### MLflow port in use
```bash
lsof -i :5000
kill -9 <PID>
```

### Airflow DAG not showing
```bash
python -m py_compile dags/marketing_creative_dag.py
# Check error output
```

## 🚀 Next Steps

### Part 3: Docker & CI/CD
- Create Dockerfiles
- Setup GitHub Actions
- Push to DockerHub

### Part 4: Kubernetes
- Write K8s manifests
- Deploy to cloud
- Configure load balancing

### Part 5: Monitoring
- Deploy Prometheus
- Setup Grafana dashboards
- Configure alerts

## 📝 Key Features

✅ Automated data generation  
✅ DistilBERT model training  
✅ MLflow experiment tracking  
✅ Model versioning & registry  
✅ Airflow pipeline orchestration  
✅ REST inference API  
✅ Kubernetes-ready design  
✅ Comprehensive testing  
✅ Full logging & monitoring  
✅ Production-grade code structure  

## 📞 Support

Refer to IMPLEMENTATION_GUIDE.md for detailed troubleshooting and setup instructions.

---

**Status**: Part 1-2 Complete ✅ | Ready for Docker & CI/CD (Part 3)
