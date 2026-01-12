# Architecture Overview

## System Architecture

The RAG Evaluation Laboratory is designed with a modular architecture that allows easy swapping of components.

```
┌─────────────────────────────────────────────────────────┐
│                   Experiment Config                     │
│                  (YAML/JSON)                            │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Experiment Runner                           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  RAG System                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐             │
│  │ Chunking │→ │Embedding │→ │Vector DB │             │
│  └──────────┘  └──────────┘  └──────────┘             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Evaluation Metrics                         │
│  • Precision@K  • Recall@K  • MRR                      │
│  • Answer Correctness  • Faithfulness  • Latency       │
└─────────────────────────────────────────────────────────┘
```

## Component Details

### Chunking Module

- **FixedSizeChunker**: Simple fixed-size window chunking
- **TextTilingChunker**: Semantic segmentation using TextTiling algorithm
- **C99Chunker**: C99 algorithm for semantic segmentation
- **LayoutAwareChunker**: Structure-aware chunking for formatted documents

### Embedding Module

- **E5EmbeddingModel**: E5 family models (base, large, multilingual)
- **BGEEmbeddingModel**: BGE family models (base, large)

### Vector Database Module

- **FAISSVectorDB**: FAISS with multiple index types (Flat, IVF, HNSW)
- **MilvusVectorDB**: Milvus integration
- **QdrantVectorDB**: Qdrant integration

### Dataset Module

- **QASPERLoader**: Scientific literature QA dataset
- **HotpotQALoader**: Multi-hop reasoning dataset

### Evaluation Module

Comprehensive metrics for:
- Retrieval performance (Precision@K, Recall@K, MRR)
- Answer quality (ROUGE, BERTScore, Exact Match)
- Faithfulness to source material
- System performance (latency, throughput)

## Data Flow

1. **Configuration Loading**: Experiment config is loaded and validated
2. **Dataset Loading**: Dataset is loaded and preprocessed
3. **Indexing**: Documents are chunked, embedded, and stored in vector DB
4. **Retrieval**: Queries are embedded and searched in vector DB
5. **Evaluation**: Results are compared against ground truth
6. **Reporting**: Metrics are computed and saved

## Extensibility

The system is designed for easy extension:

- Add new chunking strategies by implementing `ChunkingStrategy`
- Add new embedding models by implementing `EmbeddingModel`
- Add new vector databases by implementing `VectorDB`
- Add new datasets by implementing `DatasetLoader`
- Add new metrics in the evaluation module

