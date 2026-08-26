from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigurationError


@dataclass(frozen=True)
class PathsConfig:
    pdf_dir: str = "data/pdfs"
    qa_dataset: str = "data/qa/100diabetes_qa_dataset.json"
    checkpoint_dir: str = "checkpoints/paper"
    output_dir: str = "outputs/paper"


@dataclass(frozen=True)
class ModelsConfig:
    generator: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    embedding: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "auto"
    use_4bit: bool = False
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.95


@dataclass(frozen=True)
class Neo4jConfig:
    uri_env: str = "NEO4J_URI"
    username_env: str = "NEO4J_USERNAME"
    password_env: str = "NEO4J_PASSWORD"


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 1024
    chunk_overlap: int = 20
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


@dataclass(frozen=True)
class GraphConfig:
    semantic_neighbors: int = 5
    similarity_percentile: float = 70.0
    max_layers: int = 4
    min_nodes_per_layer: int = 2
    clustering_resolution: float = 1.0
    community_link_mode: str = "enhanced"
    community_edge_weight: float = 0.6
    community_embedding_weight: float = 0.4
    target_edge_density: float = 0.3


@dataclass(frozen=True)
class IndexConfig:
    embedding_dim: int = 384
    M: int = 16
    ef_construction: int = 200
    ef_search: int = 50
    traversal_k: int = 3


@dataclass(frozen=True)
class RetrievalConfig:
    relevance_threshold: float = 0.3
    final_context_items: int = 5
    weighting: str = "abstract"
    adaptive_threshold: float = 0.3


@dataclass(frozen=True)
class EvaluationConfig:
    max_queries: int = 100
    faithfulness_threshold: float = 0.3
    relevance_threshold: float = 0.4
    correctness_threshold: float = 0.3


@dataclass(frozen=True)
class HAGRAGConfig:
    seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def _build(cls: type, value: dict[str, Any] | None):
    return cls(**(value or {}))


def load_config(path: str | Path) -> HAGRAGConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    config = HAGRAGConfig(
        seed=int(raw.get("seed", 42)),
        paths=_build(PathsConfig, raw.get("paths")),
        models=_build(ModelsConfig, raw.get("models")),
        neo4j=_build(Neo4jConfig, raw.get("neo4j")),
        chunking=ChunkingConfig(
            **{
                **(raw.get("chunking") or {}),
                "separators": tuple((raw.get("chunking") or {}).get("separators", ChunkingConfig.separators)),
            }
        ),
        graph=_build(GraphConfig, raw.get("graph")),
        index=_build(IndexConfig, raw.get("index")),
        retrieval=_build(RetrievalConfig, raw.get("retrieval")),
        evaluation=_build(EvaluationConfig, raw.get("evaluation")),
    )
    validate_config(config)
    return config


def validate_config(config: HAGRAGConfig) -> None:
    if config.chunking.chunk_overlap >= config.chunking.chunk_size:
        raise ConfigurationError("chunk_overlap must be smaller than chunk_size")
    if config.index.embedding_dim <= 0 or config.index.M <= 0:
        raise ConfigurationError("index dimensions and M must be positive")
    if not 0 <= config.graph.similarity_percentile <= 100:
        raise ConfigurationError("similarity_percentile must be between 0 and 100")
    if config.retrieval.weighting not in {"abstract", "equal", "specific", "adaptive"}:
        raise ConfigurationError("retrieval.weighting must be abstract, equal, specific, or adaptive")
