"""
Configuration loader for experiment configurations.
"""

import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field, validator


class ChunkingConfig(BaseModel):
    """Configuration for chunking strategy."""
    strategy: str = Field(..., description="Chunking strategy name")
    params: Dict[str, Any] = Field(default_factory=dict, description="Strategy-specific parameters")


class EmbeddingConfig(BaseModel):
    """Configuration for embedding model."""
    model_family: str = Field(..., description="Model family (e5, bge)")
    model_name: str = Field(..., description="Specific model name")
    device: Optional[str] = Field(None, description="Device to use (cuda/cpu)")


class VectorDBConfig(BaseModel):
    """Configuration for vector database."""
    backend: str = Field(..., description="Vector DB backend (faiss, milvus, qdrant)")
    params: Dict[str, Any] = Field(default_factory=dict, description="Backend-specific parameters")


class DatasetConfig(BaseModel):
    """Configuration for dataset."""
    name: str = Field(..., description="Dataset name (qasper, hotpotqa)")
    split: str = Field("test", description="Dataset split to use")
    limit: Optional[int] = Field(None, description="Limit number of examples")


class EvaluationConfig(BaseModel):
    """Configuration for evaluation."""
    k_values: list = Field(default=[1, 5, 10], description="K values for retrieval metrics")
    answer_metrics: list = Field(default=["rouge_l", "exact_match"], description="Answer evaluation metrics")


class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""
    experiment_name: str = Field(..., description="Name of the experiment")
    chunking: ChunkingConfig = Field(..., description="Chunking configuration")
    embedding: EmbeddingConfig = Field(..., description="Embedding configuration")
    vectordb: VectorDBConfig = Field(..., description="Vector database configuration")
    dataset: DatasetConfig = Field(..., description="Dataset configuration")
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig, description="Evaluation configuration")
    output_dir: str = Field("results", description="Output directory for results")

    @validator("chunking")
    def validate_chunking_strategy(cls, v):
        valid_strategies = ["fixed_size", "texttiling", "c99", "layout_aware"]
        if v.strategy not in valid_strategies:
            raise ValueError(f"Invalid chunking strategy: {v.strategy}. Valid: {valid_strategies}")
        return v

    @validator("embedding")
    def validate_embedding_family(cls, v):
        valid_families = ["e5", "bge"]
        if v.model_family not in valid_families:
            raise ValueError(f"Invalid embedding family: {v.model_family}. Valid: {valid_families}")
        return v

    @validator("vectordb")
    def validate_vectordb_backend(cls, v):
        valid_backends = ["faiss", "milvus", "qdrant"]
        if v.backend not in valid_backends:
            raise ValueError(f"Invalid vector DB backend: {v.backend}. Valid: {valid_backends}")
        return v

    @validator("dataset")
    def validate_dataset_name(cls, v):
        valid_datasets = ["qasper", "hotpotqa"]
        if v.name not in valid_datasets:
            raise ValueError(f"Invalid dataset name: {v.name}. Valid: {valid_datasets}")
        return v


def load_config(config_path: str) -> ExperimentConfig:
    """
    Load experiment configuration from YAML or JSON file.

    Args:
        config_path: Path to configuration file

    Returns:
        ExperimentConfig object
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, "r") as f:
        if path.suffix.lower() == ".yaml" or path.suffix.lower() == ".yml":
            config_dict = yaml.safe_load(f)
        elif path.suffix.lower() == ".json":
            config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {path.suffix}")

    return ExperimentConfig(**config_dict)


def validate_config(config: ExperimentConfig) -> bool:
    """
    Validate experiment configuration.

    Args:
        config: ExperimentConfig to validate

    Returns:
        True if valid, raises ValueError otherwise
    """
    # Additional validation logic can be added here
    return True

