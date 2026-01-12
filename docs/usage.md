# Usage Guide

## Running Experiments

### Single Experiment

Run a single experiment from a configuration file:

```bash
python -m src.pipeline.run_experiment --config experiments/baseline.yaml
```

### Creating Custom Experiments

1. Create a new YAML file in the `experiments/` directory
2. Configure chunking, embedding, vector DB, and dataset settings
3. Run the experiment

Example configuration:

```yaml
experiment_name: my_experiment
chunking:
  strategy: texttiling
  params:
    w: 20
    k: 10

embedding:
  model_family: e5
  model_name: e5-large

vectordb:
  backend: faiss
  params:
    index_type: hnsw

dataset:
  name: qasper
  split: test
  limit: 50

evaluation:
  k_values: [1, 5, 10]
  answer_metrics: [rouge_l, exact_match]

output_dir: results
```

## Configuration Options

### Chunking Strategies

#### Fixed Size
```yaml
strategy: fixed_size
params:
  chunk_size: 512
  chunk_overlap: 50
```

#### TextTiling
```yaml
strategy: texttiling
params:
  w: 20
  k: 10
  smoothing_width: 3
  smoothing_rounds: 1
```

#### C99
```yaml
strategy: c99
params:
  window_size: 200
  similarity_threshold: 0.5
```

#### Layout Aware
```yaml
strategy: layout_aware
params:
  max_chunk_size: 512
  preserve_structure: true
  respect_paragraphs: true
```

### Embedding Models

#### E5 Models
```yaml
model_family: e5
model_name: e5-base  # or e5-large, e5-base-v2, multilingual-e5-base, etc.
device: cuda  # or cpu, or null for auto-detect
```

#### BGE Models
```yaml
model_family: bge
model_name: bge-base  # or bge-large, bge-small, etc.
device: cuda
```

### Vector Databases

#### FAISS
```yaml
backend: faiss
params:
  index_type: flat  # or ivf, hnsw
  metric: cosine  # or l2
  # For IVF:
  nlist: 100
  # For HNSW:
  M: 32
  ef_construction: 200
```

#### Qdrant
```yaml
backend: qdrant
params:
  collection_name: rag_eval
  host: localhost
  port: 6333
```

#### Milvus
```yaml
backend: milvus
params:
  collection_name: rag_eval
  host: localhost
  port: 19530
```

## Results

Results are saved as JSON files in the `results/` directory with the following structure:

```json
{
  "experiment_name": "baseline_faiss_e5_fixed",
  "config": {...},
  "metrics": {
    "precision@1": 0.75,
    "recall@10": 0.85,
    "mrr": 0.80,
    "answer_rouge_l": 0.65,
    "mean_latency": 0.12,
    "throughput": 8.33
  },
  "num_examples": 100,
  "timestamp": "2024-01-01 12:00:00"
}
```

## Docker Usage

### Build and Run

```bash
docker-compose build
docker-compose up rag-eval
```

### With Vector Databases

```bash
# Start Qdrant
docker-compose up -d qdrant

# Start Milvus
docker-compose up -d milvus etcd

# Run experiment
docker-compose run rag-eval python -m src.pipeline.run_experiment --config experiments/bge_qdrant.yaml
```

## Programmatic Usage

```python
from src.config import load_config
from src.pipeline import ExperimentRunner

# Load configuration
config = load_config("experiments/baseline.yaml")

# Run experiment
runner = ExperimentRunner(config)
results = runner.run()

# Access metrics
print(results["metrics"])
```

