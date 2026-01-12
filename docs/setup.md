# Setup Guide

This guide will help you set up the RAG Evaluation Laboratory on your system.

## Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) Docker and Docker Compose for containerized setup
- (Optional) CUDA-capable GPU for faster embedding computation

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd rag-evaluation-lab
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Required Models

Some models will be downloaded automatically on first use. For spaCy:

```bash
python -m spacy download en_core_web_sm
```

## Vector Database Setup

### FAISS

FAISS is included in the requirements and works out of the box. No additional setup needed.

### Qdrant

#### Option 1: Docker (Recommended)

```bash
docker run -p 6333:6333 qdrant/qdrant
```

#### Option 2: Local Installation

Follow the [Qdrant installation guide](https://qdrant.tech/documentation/guides/installation/).

### Milvus

#### Option 1: Docker Compose

```bash
docker-compose up -d milvus etcd
```

#### Option 2: Standalone Installation

Follow the [Milvus installation guide](https://milvus.io/docs/install_standalone-docker.md).

## Configuration

1. Copy and modify experiment configurations in the `experiments/` directory
2. Adjust vector database connection settings if using Milvus or Qdrant
3. Set device preference for embedding models (CPU/GPU)

## Verification

Run a simple test:

```bash
python -m src.pipeline.run_experiment --config experiments/baseline.yaml
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU device
2. **Vector DB connection errors**: Ensure services are running and ports are accessible
3. **Model download failures**: Check internet connection and Hugging Face access

