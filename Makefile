PYTHON=python3
VENV=venv

# 1. Setup target
install: ## Create venv and install dependencies
	$(PYTHON) -m venv $(VENV)
	. $(VENV)/bin/activate && pip install -r requirements.txt

setup: install ## Alias for install

# 2. CI targets
lint: ## Run linting checks
	. $(VENV)/bin/activate && pylint *.py

format: ## Auto-format code
	. $(VENV)/bin/activate && black .

# 3. ML pipeline targets
prepare_data: ## Run data preparation step
	. $(VENV)/bin/activate && $(PYTHON) main.py prepare_data

train: ## Train the model
	. $(VENV)/bin/activate && $(PYTHON) main.py train

test: ## Evaluate the model
	. $(VENV)/bin/activate && $(PYTHON) main.py evaluate

# 4. Utility
clean: ## Remove venv and temporary files
	rm -rf $(VENV) __pycache__ *.joblib

api: ## Run FastAPI server
	. venv/bin/activate && uvicorn app:app --reload --host 0.0.0.0 --port 8000

mlflow_ui: ## Start MLflow tracking server
	. venv/bin/activate && mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000

all: install prepare_data train test mlflow_ui api

# 5. Docker targets
docker-build: ## Build Docker image with FastAPI + MLflow
	docker build -t siikiimii/churn-api-mlflow:latest .

docker-run: ## Run Docker container with persistent MLflow storage
	docker run -d -p 8000:8000 -p 5000:5000 \
		-v $(PWD)/mlruns:/app/mlruns \
		-v $(PWD)/mlflow.db:/app/mlflow.db \
		siikiimii/churn-api-mlflow:latest

docker-push: ## Push Docker image to Docker Hub
	docker push mohamed_aziz/churn-api-mlflow:latest
