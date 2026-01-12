# API Reference

## Chunking Strategies

### ChunkingStrategy (Base Class)

Abstract base class for all chunking strategies.

#### Methods

- `chunk(text: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]`
  - Chunk a document into segments
  - Returns list of chunk dictionaries

- `get_config() -> Dict[str, Any]`
  - Return chunker configuration

### FixedSizeChunker

Fixed-size window chunking.

```python
from src.chunking import FixedSizeChunker

chunker = FixedSizeChunker(
    chunk_size=512,
    chunk_overlap=50,
    separator="\n\n"
)
chunks = chunker.chunk(text)
```

### TextTilingChunker

TextTiling algorithm for semantic segmentation.

```python
from src.chunking import TextTilingChunker

chunker = TextTilingChunker(
    w=20,
    k=10,
    smoothing_width=3,
    smoothing_rounds=1
)
chunks = chunker.chunk(text)
```

## Embedding Models

### EmbeddingModel (Base Class)

Abstract base class for embedding models.

#### Methods

- `encode(texts: List[str], **kwargs) -> np.ndarray`
  - Encode texts into embeddings

- `get_embedding_dim() -> int`
  - Return embedding dimension

### E5EmbeddingModel

```python
from src.embeddings import E5EmbeddingModel

model = E5EmbeddingModel(
    model_name="e5-base",
    device="cuda"
)
embeddings = model.encode_passages(texts)
query_emb = model.encode_queries([query])
```

### BGEEmbeddingModel

```python
from src.embeddings import BGEEmbeddingModel

model = BGEEmbeddingModel(
    model_name="bge-large",
    device="cuda"
)
embeddings = model.encode(texts)
```

## Vector Databases

### VectorDB (Base Class)

Abstract base class for vector databases.

#### Methods

- `create_index(dimension: int, **kwargs)`
  - Create a new index

- `add_vectors(vectors: np.ndarray, ids: List[str], metadata: List[Dict])`
  - Add vectors to the database

- `search(query_vector: np.ndarray, top_k: int) -> List[Dict]`
  - Search for similar vectors

### FAISSVectorDB

```python
from src.vectordb import FAISSVectorDB

db = FAISSVectorDB(
    index_type="hnsw",
    dimension=768,
    metric="cosine"
)
db.create_index(768)
db.add_vectors(vectors, ids, metadata)
results = db.search(query_vector, top_k=10)
```

## Evaluation Metrics

### compute_precision_at_k

```python
from src.evaluation import compute_precision_at_k

precision = compute_precision_at_k(
    retrieved_ids=["doc1", "doc2", "doc3"],
    relevant_ids=["doc1", "doc3"],
    k=5
)
```

### compute_recall_at_k

```python
from src.evaluation import compute_recall_at_k

recall = compute_recall_at_k(
    retrieved_ids=["doc1", "doc2"],
    relevant_ids=["doc1", "doc2", "doc3"],
    k=5
)
```

### evaluate_rag_system

```python
from src.evaluation import evaluate_rag_system

metrics = evaluate_rag_system(
    predictions=[{"answer": "...", "source_chunks": [...]}],
    ground_truth=[{"answer": "...", "relevant_ids": [...]}],
    retrieval_results=[[{"id": "...", "score": 0.9}]],
    timings=[0.1, 0.2, 0.15]
)
```

## RAG System

### RAGSystem

Complete RAG system for evaluation.

```python
from src.config import load_config
from src.pipeline import RAGSystem

config = load_config("experiments/baseline.yaml")
rag = RAGSystem(config)

# Index documents
rag.index_documents(documents, document_ids)

# Retrieve
results = rag.retrieve(query, top_k=10)

# Answer
answer = rag.answer(query, top_k=10)
```

## Experiment Runner

### ExperimentRunner

```python
from src.config import load_config
from src.pipeline import ExperimentRunner

config = load_config("experiments/baseline.yaml")
runner = ExperimentRunner(config)
results = runner.run()
```

