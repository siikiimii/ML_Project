# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install mlflow supervisor

# Expose FastAPI and MLflow ports
EXPOSE 8000 5000

# Copy supervisor config
COPY supervisord.conf /etc/supervisord.conf

# Command to start both FastAPI and MLflow
CMD ["supervisord", "-c", "/etc/supervisord.conf"]
