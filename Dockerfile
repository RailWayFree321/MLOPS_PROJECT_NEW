# Multi-stage build for inference container
FROM python:3.11-slim as base

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements-inference.txt .

# Install Python dependencies with PyTorch CPU
RUN pip install --no-cache-dir \
    -i https://download.pytorch.org/whl/cpu \
    torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 && \
    pip install --no-cache-dir -r requirements-inference.txt

# Copy the inference app and source code
COPY src/ ./src/
COPY config/ ./config/
COPY models/best_model/ ./models/best_model/

# Create logs directory
RUN mkdir -p logs

# Set environment variables
ENV FLASK_APP=src.inference:app
ENV INFERENCE_PORT=5001
ENV MODEL_PATH=./models/best_model
ENV PYTHONUNBUFFERED=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5001/health')" || exit 1

# Expose ports
EXPOSE 5001 9090

# Run the Flask app
CMD ["python", "-m", "src.inference"]
