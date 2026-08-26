from __future__ import annotations

import heapq
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors


class CHNSWIndex:
    def __init__(self, embedding_dim: int = 384, M: int = 16, ef_construction: int = 200, ef_search: int = 50):
        self.embedding_dim = embedding_dim
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.layers: list[nx.Graph] = []
        self.node_embeddings: dict[str, np.ndarray] = {}
        self.node_data: dict[str, dict] = {}
        self.inter_layer_links: dict[str, str] = {}

    @staticmethod
    def distance(first: np.ndarray, second: np.ndarray) -> float:
        return 1.0 - float(cosine_similarity([first], [second])[0][0])

    def add_node(self, node_id: str, embedding: np.ndarray, layer: int, data: dict | None = None) -> None:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch for {node_id}: expected {self.embedding_dim}, got {vector.shape[0]}"
            )
        while len(self.layers) <= layer:
            self.layers.append(nx.Graph())
        self.layers[layer].add_node(node_id)
        self.node_embeddings[node_id] = vector
        self.node_data[node_id] = data or {}

    def build_intra_layer_links(self) -> None:
        for graph in self.layers:
            nodes = list(graph.nodes())
            if len(nodes) <= 1:
                continue
            matrix = np.vstack([self.node_embeddings[node] for node in nodes])
            neighbors = min(self.M + 1, len(nodes))
            nn = NearestNeighbors(n_neighbors=neighbors, metric="cosine").fit(matrix)
            distances, indices = nn.kneighbors(matrix)
            for i, node in enumerate(nodes):
                for distance, j in zip(distances[i], indices[i], strict=True):
                    if i == j:
                        continue
                    graph.add_edge(
                        node,
                        nodes[j],
                        weight=float(distance),
                        similarity=float(1.0 - distance),
                    )

    def build_inter_layer_links(self) -> None:
        for layer in range(1, len(self.layers)):
            higher = list(self.layers[layer].nodes())
            lower = list(self.layers[layer - 1].nodes())
            if not higher or not lower:
                continue
            matrix = np.vstack([self.node_embeddings[node] for node in lower])
            nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(matrix)
            for node in higher:
                _, indices = nn.kneighbors([self.node_embeddings[node]])
                self.inter_layer_links[node] = lower[int(indices[0][0])]

    def search_layer(self, layer: int, query: np.ndarray, entry_point: str, k: int = 3) -> list[str]:
        if layer >= len(self.layers) or self.layers[layer].number_of_nodes() == 0:
            return []
        graph = self.layers[layer]
        if entry_point not in graph:
            entry_point = next(iter(graph.nodes()))
        initial = self.distance(query, self.node_embeddings[entry_point])
        queue = [(initial, entry_point)]
        best: dict[str, float] = {entry_point: initial}
        visited: set[str] = set()

        while queue and len(visited) < self.ef_search:
            _, current = heapq.heappop(queue)
            if current in visited:
                continue
            visited.add(current)
            for neighbor in graph.neighbors(current):
                if neighbor in visited:
                    continue
                query_distance = self.distance(query, self.node_embeddings[neighbor])
                edge = graph.get_edge_data(current, neighbor) or {}
                edge_distance = float(edge.get("weight", 1.0))
                adjusted = query_distance * edge_distance
                if adjusted < best.get(neighbor, float("inf")):
                    best[neighbor] = adjusted
                    heapq.heappush(queue, (adjusted, neighbor))

        return [node for node, _ in sorted(best.items(), key=lambda item: item[1])[:k]]

    def hierarchical_search(self, query: np.ndarray, k: int = 3) -> dict[int, list[str]]:
        if not self.layers:
            return {}
        top = len(self.layers) - 1
        while top >= 0 and self.layers[top].number_of_nodes() == 0:
            top -= 1
        if top < 0:
            return {}

        entry = next(iter(self.layers[top].nodes()))
        results: dict[int, list[str]] = {}
        for layer in range(top, -1, -1):
            if self.layers[layer].number_of_nodes() == 0:
                continue
            found = self.search_layer(layer, query, entry, k=k)
            results[layer] = found
            if layer > 0 and found:
                entry = self.inter_layer_links.get(found[0], next(iter(self.layers[layer - 1].nodes())))
        return results

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(self, handle)

    @classmethod
    def load(cls, path: str | Path) -> "CHNSWIndex":
        with Path(path).open("rb") as handle:
            value = pickle.load(handle)
        if not isinstance(value, cls):
            raise TypeError("Checkpoint does not contain a CHNSWIndex")
        return value
