# ML_Project
MLOps project with FastAPI, MLflow, and Docker
=======

# ML Project 2 – MLOps with FastAPI, MLflow, and Docker

## 📌 Overview
This project demonstrates the end-to-end lifecycle of a Machine Learning model using **MLOps practices**.

It includes:
- Data preparation, training, and evaluation
- Model tracking with **MLflow**
- Serving predictions via **FastAPI**
- Containerization with **Docker** and process management via **Supervisor**
- Automation using a **Makefile**

## ⚙️ Project Structure
```
├── app.py                   # FastAPI application (prediction API)
├── main.py                  # Entry point for data prep, training, evaluation
├── model_pipeline.py        # ML pipeline definition
├── model.joblib             # Trained model artifact
├── mlflow.db                # MLflow backend store (SQLite)
├── mlruns/                  # MLflow experiment runs
├── Dockerfile               # Container build instructions
├── Makefile                 # Automation commands
├── supervisord.conf         # Supervisor config (FastAPI + MLflow)
├── requirements.txt         # Python dependencies
├── test_environment.py      # Environment validation script
└── venv/                    # Virtual environment (ignored in Git)
```

## 🚀 Setup Instructions

### 1. Local Environment
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the ML Pipeline
```bash
python main.py prepare_data
python main.py train
python main.py evaluate
```

### 3. Start MLflow UI
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```
Access MLflow at: http://localhost:5000

### 4. Start FastAPI
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
Access API docs at: http://localhost:8000/docs

## 🐳 Docker Deployment
```bash
docker build -t mlops-project:latest .
docker run -d -p 8000:8000 -p 5000:5000   -v "$(pwd)/mlruns:/app/mlruns"   -v "$(pwd)/mlflow.db:/app/mlflow.db"   mlops-project:latest
```
FastAPI → http://localhost:8000/docs  
MLflow → http://localhost:5000

## 📂 Makefile Targets
- `make install` → Create virtual environment and install dependencies  
- `make prepare_data` → Run data preparation  
- `make train` → Train the model  
- `make test` → Evaluate the model  
- `make api` → Start FastAPI server  
- `make mlflow_ui` → Start MLflow UI  
- `make docker-build` → Build Docker image  
- `make docker-run` → Run Docker container  
- `make clean` → Remove generated artifacts  

## ✅ Testing
```bash
python test_environment.py
pytest -q
```

## 📸 What to Submit
- GitHub repo link (exclude heavy artifacts via .gitignore)  
- Screenshots:
  - MLflow UI showing experiment runs and metrics
  - FastAPI docs page
  - Docker container running (`docker ps`)
- Short report (1–2 pages):
  - Context (dataset, model type)
  - Pipeline steps (prep → train → evaluate → deploy)
  - Ops setup (MLflow, FastAPI, Docker, Makefile)
  - Results (metrics, artifacts)
  - Challenges & improvements

## 🏆 Key Takeaway
This project shows how MLOps practices make ML models reproducible, trackable, and deployable in production environments.
