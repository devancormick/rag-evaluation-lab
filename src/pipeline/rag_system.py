"""
RAG system implementation for evaluation.
"""

from typing import List, Dict, Any, Optional
import time
import numpy as np

from ..chunking import ChunkingStrategy, FixedSizeChunker, TextTilingChunker, C99Chunker, LayoutAwareChunker
from ..embeddings import EmbeddingModel, E5EmbeddingModel, BGEEmbeddingModel
from ..vectordb import VectorDB, FAISSVectorDB, MilvusVectorDB, QdrantVectorDB
from ..config import ExperimentConfig


class RAGSystem:
    """Complete RAG system for evaluation."""

    def __init__(self, config: ExperimentConfig):
        """
        Initialize RAG system from configuration.

        Args:
            config: Experiment configuration
        """
        self.config = config

        # Initialize chunker
        self.chunker = self._create_chunker()

        # Initialize embedding model
        self.embedding_model = self._create_embedding_model()

        # Initialize vector database
        self.vectordb = self._create_vectordb()

        # Indexed chunks
        self.chunk_store = {}  # Map chunk_id to chunk text

    def _create_chunker(self) -> ChunkingStrategy:
        """Create chunking strategy from config."""
        strategy = self.config.chunking.strategy
        params = self.config.chunking.params

        if strategy == "fixed_size":
            return FixedSizeChunker(**params)
        elif strategy == "texttiling":
            return TextTilingChunker(**params)
        elif strategy == "c99":
            return C99Chunker(**params)
        elif strategy == "layout_aware":
            return LayoutAwareChunker(**params)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

    def _create_embedding_model(self) -> EmbeddingModel:
        """Create embedding model from config."""
        family = self.config.embedding.model_family
        model_name = self.config.embedding.model_name
        device = self.config.embedding.device

        if family == "e5":
            return E5EmbeddingModel(model_name=model_name, device=device)
        elif family == "bge":
            return BGEEmbeddingModel(model_name=model_name, device=device)
        else:
            raise ValueError(f"Unknown embedding family: {family}")

    def _create_vectordb(self) -> VectorDB:
        """Create vector database from config."""
        backend = self.config.vectordb.backend
        params = self.config.vectordb.params

        if backend == "faiss":
            return FAISSVectorDB(**params)
        elif backend == "milvus":
            return MilvusVectorDB(**params)
        elif backend == "qdrant":
            return QdrantVectorDB(**params)
        else:
            raise ValueError(f"Unknown vector DB backend: {backend}")

    def index_documents(self, documents: List[str], document_ids: Optional[List[str]] = None):
        """
        Index documents by chunking, embedding, and storing in vector DB.

        Args:
            documents: List of document texts
            document_ids: Optional list of document IDs
        """
        if document_ids is None:
            document_ids = [f"doc_{i}" for i in range(len(documents))]

        all_chunks = []
        all_chunk_ids = []
        all_metadata = []

        # Chunk documents
        for doc_id, doc_text in zip(document_ids, documents):
            chunks = self.chunker.chunk(doc_text, metadata={"document_id": doc_id})

            for chunk in chunks:
                chunk_id = f"{doc_id}_chunk_{chunk['metadata'].get('chunk_index', len(all_chunks))}"
                all_chunks.append(chunk["text"])
                all_chunk_ids.append(chunk_id)
                all_metadata.append({
                    **chunk["metadata"],
                    "chunk_id": chunk_id,
                    "document_id": doc_id
                })
                self.chunk_store[chunk_id] = chunk["text"]

        # Embed chunks
        if self.config.embedding.model_family == "e5":
            chunk_embeddings = self.embedding_model.encode_passages(all_chunks)
        else:
            chunk_embeddings = self.embedding_model.encode_passages(all_chunks)

        # Create vector DB index
        embedding_dim = self.embedding_model.get_embedding_dim()
        self.vectordb.create_index(embedding_dim)

        # Add to vector DB
        self.vectordb.add_vectors(
            vectors=chunk_embeddings,
            ids=all_chunk_ids,
            metadata=all_metadata
        )

    def retrieve(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Query text
            top_k: Number of results to return

        Returns:
            List of retrieved chunks with metadata
        """
        # Encode query
        if self.config.embedding.model_family == "e5":
            query_embedding = self.embedding_model.encode_queries([query])[0]
        else:
            query_embedding = self.embedding_model.encode_queries([query])[0]

        # Search vector DB
        results = self.vectordb.search(query_embedding, top_k=top_k)

        # Enrich results with chunk text
        for result in results:
            chunk_id = result["id"]
            result["chunk_text"] = self.chunk_store.get(chunk_id, "")

        return results

    def answer(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Generate answer using retrieved context (simple implementation).

        Args:
            query: Query text
            top_k: Number of chunks to retrieve

        Returns:
            Dictionary with answer and source information
        """
        start_time = time.time()

        # Retrieve relevant chunks
        retrieved = self.retrieve(query, top_k=top_k)

        # Simple answer generation: concatenate top chunks
        # In a real system, this would use an LLM
        source_chunks = [r["chunk_text"] for r in retrieved]
        answer = "\n".join(source_chunks[:3])  # Use top 3 chunks

        latency = time.time() - start_time

        return {
            "answer": answer,
            "source_chunks": source_chunks,
            "retrieved_ids": [r["id"] for r in retrieved],
            "latency": latency
        }

    def clear(self):
        """Clear all indexed data."""
        self.vectordb.clear()
        self.chunk_store = {}

