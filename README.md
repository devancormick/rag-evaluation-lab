# RAG Evaluation Laboratory

A comprehensive research framework for systematically evaluating Retrieval-Augmented Generation (RAG) systems across different configurations, chunking methods, vector databases, and embedding models.

## Overview

This laboratory provides a configurable framework to evaluate RAG systems by comparing:
- **Chunking Methods**: Fixed-size windows, TextTiling, C99, and layout-aware approaches
- **Vector Databases**: FAISS (Flat, IVF, HNSW), Milvus, and Qdrant
- **Embedding Models**: E5 and BGE families (base, large, multilingual variants)
- **Evaluation Datasets**: QASPER and HotpotQA

## Features

- Modular architecture for easy component swapping
- YAML/JSON-based experiment configuration
- Automated evaluation pipeline
- Comprehensive metrics: Precision@K, Recall@K, MRR, answer correctness, faithfulness, latency/throughput
- Results analysis and reporting dashboard
- Full reproducibility with Docker support

## Project Structure

```
rag-evaluation-lab/
├── src/
│   ├── chunking/          # Chunking method implementations
│   ├── embeddings/         # Embedding model wrappers
│   ├── vectordb/          # Vector database backends
│   ├── datasets/          # Dataset loaders
│   ├── evaluation/        # Evaluation metrics
│   ├── pipeline/          # Evaluation pipeline
│   ├── config/            # Configuration management
│   └── utils/             # Utility functions
├── experiments/           # Experiment configurations
├── results/               # Evaluation results
├── tests/                 # Unit and integration tests
├── docs/                  # Documentation
└── docker/                # Docker setup

```

## Quick Start

### Prerequisites

- Python 3.9+
- Docker (optional, for containerized setup)
- CUDA-capable GPU (recommended for embedding models)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd rag-evaluation-lab

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Experiments

```bash
# Run a single experiment
python -m src.pipeline.run_experiment --config experiments/baseline.yaml

# Run all experiments
python -m src.pipeline.run_all_experiments --config-dir experiments/
```

## Documentation

- [Setup Guide](docs/setup.md)
- [Architecture Overview](docs/architecture.md)
- [Usage Guide](docs/usage.md)
- [API Reference](docs/api-reference.md)

## License

MIT License

