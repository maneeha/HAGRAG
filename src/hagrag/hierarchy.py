from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .clustering import cluster_graph
from .graph import augment_graph_with_similarity


@dataclass
class Community:
    id: str
    layer: int
    members: list[str]
    summary: str
    embedding: np.ndarray
    algorithm: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "layer": self.layer,
            "members": self.members,
            "summary": self.summary,
            "embedding": self.embedding.tolist(),
            "algorithm": self.algorithm,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "Community":
        return cls(
            id=value["id"],
            layer=int(value["layer"]),
            members=list(value["members"]),
            summary=value["summary"],
            embedding=np.asarray(value["embedding"], dtype=np.float32),
            algorithm=value["algorithm"],
        )


class CommunitySummarizer:
    def __init__(self, generator, entity_details: Mapping[str, dict]):
        self.generator = generator
        self.entity_details = entity_details

    def summarize(self, members: list[str], previous: Mapping[str, Community]) -> str:
        parts: list[str] = []
        for member in members:
            if member in previous:
                parts.append(previous[member].summary)
            else:
                info = self.entity_details.get(member, {})
                description = info.get("description", "")
                entity_type = info.get("type", "Unknown")
                parts.append(f"{member} ({entity_type}): {description}".strip())
        source = "\n".join(parts)[:10000]
        prompt = f"""Summarize the biomedical community below. Describe the common theme, key entities,
and important relationships. Use only the supplied information and keep the summary concise.

Community information:
{source}

Summary:"""
        return self.generator.generate(prompt, max_new_tokens=220, temperature=0.0)


def _member_embedding(member: str, base: Mapping[str, np.ndarray], previous: Mapping[str, Community]):
    if member in previous:
        return previous[member].embedding
    return base.get(member)


def _community_connection(
    first: Community,
    second: Community,
    lower_graph: nx.Graph,
    base_embeddings: Mapping[str, np.ndarray],
    previous: Mapping[str, Community],
    edge_weight: float,
    embedding_weight: float,
) -> float:
    possible = max(1, len(first.members) * len(second.members))
    crossing = sum(1 for u in first.members for v in second.members if lower_graph.has_edge(u, v))
    edge_score = crossing / possible

    first_vectors = [
        vector
        for member in first.members
        if (vector := _member_embedding(member, base_embeddings, previous)) is not None
    ]
    second_vectors = [
        vector
        for member in second.members
        if (vector := _member_embedding(member, base_embeddings, previous)) is not None
    ]
    semantic = 0.0
    if first_vectors and second_vectors:
        centroid_a = np.mean(first_vectors, axis=0)
        centroid_b = np.mean(second_vectors, axis=0)
        semantic = max(0.0, float(cosine_similarity([centroid_a], [centroid_b])[0][0]))
    return edge_weight * edge_score + embedding_weight * semantic


def _build_next_graph(
    communities: list[Community],
    lower_graph: nx.Graph,
    base_embeddings: Mapping[str, np.ndarray],
    previous: Mapping[str, Community],
    mode: str,
    edge_weight: float,
    embedding_weight: float,
    target_density: float,
) -> nx.Graph:
    graph = nx.Graph()
    for community in communities:
        graph.add_node(community.id, summary=community.summary)
    if len(communities) < 2:
        return graph

    scored: list[tuple[float, str, str]] = []
    for i, first in enumerate(communities):
        for second in communities[i + 1 :]:
            if mode == "paper":
                linked = any(lower_graph.has_edge(u, v) for u in first.members for v in second.members)
                if linked:
                    scored.append((1.0, first.id, second.id))
            else:
                score = _community_connection(
                    first,
                    second,
                    lower_graph,
                    base_embeddings,
                    previous,
                    edge_weight,
                    embedding_weight,
                )
                scored.append((score, first.id, second.id))

    if not scored:
        return graph
    if mode == "paper":
        threshold = 0.5
    else:
        positive = sorted((score for score, _, _ in scored if score > 0), reverse=True)
        if not positive:
            return graph
        index = min(len(positive) - 1, max(0, int(len(positive) * target_density)))
        threshold = max(0.01, positive[index])

    for score, first, second in scored:
        if score >= threshold:
            graph.add_edge(first, second, weight=score)
    return graph


def build_hierarchy(
    base_graph: nx.Graph,
    base_embeddings: Mapping[str, np.ndarray],
    embedder,
    summarizer: CommunitySummarizer,
    algorithm: str,
    max_layers: int,
    min_nodes_per_layer: int,
    resolution: float,
    seed: int,
    semantic_neighbors: int,
    similarity_percentile: float,
    community_link_mode: str = "enhanced",
    community_edge_weight: float = 0.6,
    community_embedding_weight: float = 0.4,
    target_edge_density: float = 0.3,
) -> tuple[dict[int, list[Community]], dict[int, nx.Graph]]:
    current_graph = base_graph.copy()
    previous: dict[str, Community] = {}
    hierarchy: dict[int, list[Community]] = {}
    layer_graphs: dict[int, nx.Graph] = {}

    # Layer 0 is reserved for base entities; community layers begin at Layer 1.
    for layer in range(1, max_layers):
        if current_graph.number_of_nodes() < min_nodes_per_layer:
            break
        current_embeddings = {
            node: previous[node].embedding if node in previous else base_embeddings[node]
            for node in current_graph.nodes()
            if node in previous or node in base_embeddings
        }
        augmented, _ = augment_graph_with_similarity(
            current_graph,
            current_embeddings,
            top_k=semantic_neighbors,
            percentile=similarity_percentile,
        )
        communities_raw = cluster_graph(augmented, algorithm, current_embeddings, resolution, seed)
        if not communities_raw:
            break

        communities: list[Community] = []
        for index, members_set in enumerate(communities_raw):
            members = sorted(members_set)
            summary = summarizer.summarize(members, previous)
            embedding = np.asarray(embedder.encode(summary), dtype=np.float32)
            communities.append(
                Community(
                    id=f"L{layer}_C{index}",
                    layer=layer,
                    members=members,
                    summary=summary,
                    embedding=embedding,
                    algorithm=algorithm,
                )
            )
        hierarchy[layer] = communities
        layer_graphs[layer] = augmented

        next_graph = _build_next_graph(
            communities,
            augmented,
            base_embeddings,
            previous,
            community_link_mode,
            community_edge_weight,
            community_embedding_weight,
            target_edge_density,
        )
        previous = {community.id: community for community in communities}
        current_graph = next_graph
        if len(communities) <= 1:
            break
    return hierarchy, layer_graphs
