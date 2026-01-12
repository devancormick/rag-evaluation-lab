FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm || true

# Copy application code
COPY . .

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "-m", "src.pipeline.run_experiment", "--config", "experiments/baseline.yaml"]

