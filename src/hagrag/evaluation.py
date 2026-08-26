from __future__ import annotations

from collections.abc import Mapping

import networkx as nx
import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity


_ROUGE = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    return float(cosine_similarity([first], [second])[0][0])


def response_metrics(
    query: str,
    response: str,
    reference: str,
    evidence_texts: list[str],
    embedder,
    faithfulness_threshold: float = 0.3,
    relevance_threshold: float = 0.4,
    correctness_threshold: float = 0.3,
) -> dict[str, float]:
    query_emb, response_emb, reference_emb = embedder.encode([query, response, reference])
    evidence_embs = embedder.encode(evidence_texts) if evidence_texts else np.empty((0, len(response_emb)))
    faithfulness_sim = (
        max(cosine(response_emb, evidence) for evidence in evidence_embs) if len(evidence_embs) else 0.0
    )
    relevance_sim = cosine(query_emb, response_emb)
    semantic = cosine(response_emb, reference_emb)
    rouge = _ROUGE.score(reference, response)
    correctness = (semantic + rouge["rougeL"].fmeasure) / 2.0
    return {
        "faithfulness": float(faithfulness_sim > faithfulness_threshold),
        "relevancy": float(relevance_sim > relevance_threshold),
        "correctness_score": correctness,
        "correctness_pass": float(correctness > correctness_threshold),
        "recall": rouge["rouge1"].recall,
        "semantic_similarity": semantic,
    }


def graph_quality(graph: nx.Graph, communities: list[set[str]]) -> dict[str, float]:
    if graph.number_of_nodes() == 0:
        return {"modularity": 0.0, "coverage": 0.0, "conductance": 0.0, "density": 0.0}
    valid = [set(c) & set(graph.nodes()) for c in communities]
    valid = [c for c in valid if c]
    if not valid:
        return {"modularity": 0.0, "coverage": 0.0, "conductance": 0.0, "density": nx.density(graph)}

    assigned: set[str] = set()
    partition: list[set[str]] = []
    for community in valid:
        unique = community - assigned
        if unique:
            partition.append(unique)
            assigned.update(unique)
    for node in graph.nodes():
        if node not in assigned:
            partition.append({node})

    modularity = nx.algorithms.community.quality.modularity(graph, partition, weight="weight")
    coverage = nx.algorithms.community.quality.partition_quality(graph, partition)[0]
    conductances = []
    for community in partition:
        if 0 < len(community) < graph.number_of_nodes():
            try:
                conductances.append(nx.conductance(graph, community, weight="weight"))
            except ZeroDivisionError:
                continue
    return {
        "modularity": float(modularity),
        "coverage": float(coverage),
        "conductance": float(np.mean(conductances)) if conductances else 0.0,
        "density": float(nx.density(graph)),
    }


def retrieval_metrics(
    retrieved_texts: list[str],
    reference: str,
    query: str,
    embedder,
    relevance_threshold: float = 0.3,
) -> dict[str, float]:
    if not retrieved_texts:
        return {"recall_at_k": 0.0, "mrr": 0.0, "semantic_precision": 0.0, "context_relevance": 0.0, "hit_rate": 0.0}
    query_emb = embedder.encode(query)
    reference_emb = embedder.encode(reference)
    retrieved_emb = embedder.encode(retrieved_texts)
    ref_scores = [cosine(reference_emb, emb) for emb in retrieved_emb]
    query_scores = [cosine(query_emb, emb) for emb in retrieved_emb]
    relevant = [score >= relevance_threshold for score in ref_scores]
    first = next((i for i, value in enumerate(relevant, 1) if value), None)
    return {
        "recall_at_k": float(any(relevant)),
        "mrr": 1.0 / first if first else 0.0,
        "semantic_precision": float(np.mean(ref_scores)),
        "context_relevance": float(np.mean(query_scores)),
        "hit_rate": float(any(relevant)),
    }


def mean_metrics(rows: list[Mapping[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}
