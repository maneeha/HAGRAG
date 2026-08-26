from __future__ import annotations

from collections.abc import Mapping

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def embed_entities(graph: nx.Graph, entity_details: Mapping[str, dict], embedder) -> dict[str, np.ndarray]:
    names = list(graph.nodes())
    if not names:
        return {}
    texts = []
    for name in names:
        info = entity_details.get(name, {})
        description = info.get("description") or graph.nodes[name].get("description", "")
        texts.append(f"{name}. {description}".strip())
    matrix = embedder.encode(texts)
    return {name: np.asarray(matrix[i], dtype=np.float32) for i, name in enumerate(names)}


def augment_graph_with_similarity(
    graph: nx.Graph,
    embeddings: Mapping[str, np.ndarray],
    top_k: int = 5,
    percentile: float = 70.0,
) -> tuple[nx.Graph, float]:
    augmented = graph.copy()
    nodes = [node for node in augmented.nodes() if node in embeddings]
    if len(nodes) < 2:
        return augmented, 1.0

    matrix = np.vstack([embeddings[node] for node in nodes])
    similarities = cosine_similarity(matrix)
    upper = similarities[np.triu_indices(len(nodes), k=1)]
    threshold = float(np.percentile(upper, percentile)) if upper.size else 1.0

    for i, node in enumerate(nodes):
        order = np.argsort(similarities[i])[::-1]
        added = 0
        for j in order:
            if i == j:
                continue
            score = float(similarities[i, j])
            if score < threshold:
                break
            neighbor = nodes[j]
            if not augmented.has_edge(node, neighbor):
                augmented.add_edge(node, neighbor, weight=score, semantic=True)
            elif "weight" not in augmented[node][neighbor]:
                augmented[node][neighbor]["weight"] = score
            added += 1
            if added >= top_k:
                break
    return augmented, threshold
