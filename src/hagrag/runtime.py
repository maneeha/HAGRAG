from __future__ import annotations

from pathlib import Path

from .checkpoint import load_embeddings, load_hierarchy, save_embeddings, save_hierarchy
from .chnsw import CHNSWIndex
from .config import HAGRAGConfig
from .data import PDFProcessor
from .embeddings import SentenceTransformerEmbedder
from .errors import DataError
from .graph import embed_entities
from .hierarchy import CommunitySummarizer, build_hierarchy
from .indexing import build_index
from .io_utils import ensure_dir, seed_everything
from .llm import GenerationSettings, HuggingFaceGenerator
from .retrieval import HAGRAGRetriever
from .storage import Neo4jStore
from .triplets import TripletExtractor


def create_embedder(config: HAGRAGConfig):
    return SentenceTransformerEmbedder(config.models.embedding, config.models.device)


def create_generator(config: HAGRAGConfig):
    return HuggingFaceGenerator(
        config.models.generator,
        config.models.device,
        config.models.use_4bit,
        GenerationSettings(config.models.max_new_tokens, config.models.temperature, config.models.top_p),
    )


def checkpoint_paths(config: HAGRAGConfig, algorithm: str) -> dict[str, Path]:
    root = ensure_dir(Path(config.paths.checkpoint_dir) / algorithm)
    return {
        "embeddings": root / "entity_embeddings.pkl",
        "hierarchy": root / "hierarchy.json",
        "index": root / "chnsw_index.pkl",
    }


def build_system(config: HAGRAGConfig, algorithm: str = "leiden", clear_graph: bool = False) -> dict:
    seed_everything(config.seed)
    generator = create_generator(config)
    embedder = create_embedder(config)
    store = Neo4jStore.from_env(config.neo4j)
    store.verify()
    try:
        processor = PDFProcessor(config.paths.pdf_dir, Path(config.paths.checkpoint_dir) / "preprocessing")
        documents = processor.extract_documents()
        chunks = processor.create_chunks(
            documents,
            config.chunking.chunk_size,
            config.chunking.chunk_overlap,
            config.chunking.separators,
        )
        extractor = TripletExtractor(generator, Path(config.paths.checkpoint_dir) / "triplets")
        entities, relationships = extractor.extract(chunks)
        if clear_graph:
            store.clear()
        store.write_triplets(entities, relationships)
        details = store.entity_details()
        graph = store.to_networkx()
        if graph.number_of_nodes() == 0:
            raise DataError("Neo4j graph contains no Entity nodes after triplet extraction")

        embeddings = embed_entities(graph, details, embedder)
        summarizer = CommunitySummarizer(generator, details)
        hierarchy, _ = build_hierarchy(
            graph,
            embeddings,
            embedder,
            summarizer,
            algorithm,
            config.graph.max_layers,
            config.graph.min_nodes_per_layer,
            config.graph.clustering_resolution,
            config.seed,
            config.graph.semantic_neighbors,
            config.graph.similarity_percentile,
            config.graph.community_link_mode,
            config.graph.community_edge_weight,
            config.graph.community_embedding_weight,
            config.graph.target_edge_density,
        )
        index = build_index(
            embeddings,
            hierarchy,
            config.index.embedding_dim,
            config.index.M,
            config.index.ef_construction,
            config.index.ef_search,
        )
        paths = checkpoint_paths(config, algorithm)
        save_embeddings(paths["embeddings"], embeddings)
        save_hierarchy(paths["hierarchy"], hierarchy)
        index.save(paths["index"])
        return {
            "algorithm": algorithm,
            "documents": len(documents),
            "chunks": len(chunks),
            "entities": graph.number_of_nodes(),
            "relationships": graph.number_of_edges(),
            "layers": {layer: len(values) for layer, values in hierarchy.items()},
            "checkpoints": {key: str(value) for key, value in paths.items()},
        }
    finally:
        store.close()


def load_query_engine(config: HAGRAGConfig, algorithm: str, weighting: str | None = None):
    paths = checkpoint_paths(config, algorithm)
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        raise DataError(
            "Missing HAGRAG checkpoint(s): " + ", ".join(str(path) for path in missing) + ". Run `hagrag build` first."
        )
    generator = create_generator(config)
    embedder = create_embedder(config)
    store = Neo4jStore.from_env(config.neo4j)
    store.verify()
    try:
        details = store.entity_details()
    finally:
        store.close()
    index = CHNSWIndex.load(paths["index"])
    return HAGRAGRetriever(
        index=index,
        embedder=embedder,
        generator=generator,
        entity_details=details,
        relevance_threshold=config.retrieval.relevance_threshold,
        weighting=weighting or config.retrieval.weighting,
        adaptive_threshold=config.retrieval.adaptive_threshold,
        traversal_k=config.index.traversal_k,
        final_context_items=config.retrieval.final_context_items,
    )
