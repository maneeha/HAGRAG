from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import math

import networkx as nx
import numpy as np


def _partition_to_communities(partition: Mapping[str, int]) -> list[set[str]]:
    groups: dict[int, set[str]] = defaultdict(set)
    for node, community in partition.items():
        groups[int(community)].add(str(node))
    return [members for _, members in sorted(groups.items()) if members]


def leiden_communities(graph: nx.Graph, resolution: float = 1.0, seed: int = 42) -> list[set[str]]:
    if graph.number_of_nodes() <= 1:
        return [set(graph.nodes())] if graph.number_of_nodes() else []
    from graspologic.partition import hierarchical_leiden

    assignments = hierarchical_leiden(graph, resolution=resolution, random_seed=seed)
    partition: dict[str, int] = {}
    for item in assignments:
        node = getattr(item, "node", None)
        cluster = getattr(item, "cluster", None)
        if node is not None and cluster is not None and str(node) not in partition:
            partition[str(node)] = int(cluster)
    for node in graph.nodes():
        partition.setdefault(str(node), max(partition.values(), default=-1) + 1)
    return _partition_to_communities(partition)


def louvain_communities(graph: nx.Graph, resolution: float = 1.0, seed: int = 42) -> list[set[str]]:
    if graph.number_of_nodes() <= 1:
        return [set(graph.nodes())] if graph.number_of_nodes() else []
    import community as community_louvain

    partition = community_louvain.best_partition(
        graph, weight="weight", resolution=resolution, random_state=seed
    )
    return _partition_to_communities(partition)


def agglomerative_communities(
    graph: nx.Graph,
    embeddings: Mapping[str, np.ndarray],
) -> list[set[str]]:
    nodes = [node for node in graph.nodes() if node in embeddings]
    missing = [node for node in graph.nodes() if node not in embeddings]
    if len(nodes) <= 1:
        communities = [set(nodes)] if nodes else []
        communities.extend({node} for node in missing)
        return communities

    from sklearn.cluster import AgglomerativeClustering

    n_clusters = max(2, int(math.sqrt(len(nodes) / 2)))
    n_clusters = min(n_clusters, len(nodes))
    matrix = np.vstack([embeddings[node] for node in nodes])
    labels = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward").fit_predict(matrix)
    groups: dict[int, set[str]] = defaultdict(set)
    for node, label in zip(nodes, labels, strict=True):
        groups[int(label)].add(node)
    communities = [group for _, group in sorted(groups.items())]
    communities.extend({node} for node in missing)
    return communities


def cluster_graph(
    graph: nx.Graph,
    algorithm: str,
    embeddings: Mapping[str, np.ndarray],
    resolution: float = 1.0,
    seed: int = 42,
) -> list[set[str]]:
    algorithm = algorithm.lower()
    if algorithm == "leiden":
        return leiden_communities(graph, resolution, seed)
    if algorithm == "louvain":
        return louvain_communities(graph, resolution, seed)
    if algorithm == "agglomerative":
        return agglomerative_communities(graph, embeddings)
    raise ValueError(f"Unsupported clustering algorithm: {algorithm}")
