from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping


SPECIFIC_TERMS = (
    "does",
    "result",
    "finding",
    "study",
    "trial",
    "data",
    "experiment",
    "measured",
    "compared",
    "effect",
)
BROAD_TERMS = (
    "overview",
    "general",
    "broad",
    "overall",
    "explain",
    "describe",
    "types",
    "kinds",
    "about",
    "concept",
    "define",
    "introduction",
)


def query_specificity(query: str) -> float:
    lowered = query.lower()
    specific = sum(1 for term in SPECIFIC_TERMS if term in lowered)
    broad = sum(1 for term in BROAD_TERMS if term in lowered)
    total = specific + broad
    return (specific - broad) / total if total else 0.0


def layer_weight(
    strategy: str,
    layer: int,
    max_layer: int,
    query: str = "",
    adaptive_threshold: float = 0.3,
) -> float:
    if max_layer <= 0:
        return 1.0
    ratio = layer / max_layer
    strategy = {"original": "abstract", "reverse": "specific"}.get(strategy, strategy)
    if strategy == "abstract":
        return 1.0 + 0.5 * ratio
    if strategy == "equal":
        return 1.0
    if strategy == "specific":
        return 1.5 - 0.5 * ratio
    if strategy == "adaptive":
        specificity = query_specificity(query)
        if specificity > adaptive_threshold:
            return 1.5 - 0.5 * ratio
        if specificity < -adaptive_threshold:
            return 1.0 + 0.5 * ratio
        return 1.0
    raise ValueError(f"Unknown weighting strategy: {strategy}")


@dataclass
class RetrievedItem:
    id: str
    layer: int
    item_type: str
    text: str


@dataclass
class LayerReport:
    layer: int
    base_score: float
    weighted_score: float
    weight: float
    items: list[RetrievedItem]


class HAGRAGRetriever:
    def __init__(
        self,
        index,
        embedder,
        generator,
        entity_details: Mapping[str, dict],
        relevance_threshold: float = 0.3,
        weighting: str = "abstract",
        adaptive_threshold: float = 0.3,
        traversal_k: int = 3,
        final_context_items: int = 5,
    ):
        self.index = index
        self.embedder = embedder
        self.generator = generator
        self.entity_details = entity_details
        self.relevance_threshold = relevance_threshold
        self.weighting = weighting
        self.adaptive_threshold = adaptive_threshold
        self.traversal_k = traversal_k
        self.final_context_items = final_context_items

    def _item(self, node_id: str, layer: int) -> RetrievedItem:
        data = self.index.node_data.get(node_id, {})
        if data.get("type") == "community":
            return RetrievedItem(node_id, layer, "community", str(data.get("summary", "")))
        info = self.entity_details.get(node_id, {})
        description = info.get("description", "")
        entity_type = info.get("type", "Unknown")
        return RetrievedItem(node_id, layer, "entity", f"{node_id} ({entity_type}): {description}".strip())

    @staticmethod
    def _score_number(text: str) -> float:
        match = re.search(r"(?<!\d)(10(?:\.0+)?|[0-9](?:\.\d+)?)(?!\d)", text)
        if not match:
            return 0.0
        return max(0.0, min(10.0, float(match.group(1))))

    def _relevance(self, query: str, items: list[RetrievedItem]) -> float:
        content = "\n".join(item.text for item in items)[:5000]
        prompt = f"""Rate how relevant the supplied information is to the biomedical query.
Return one number from 0 to 10 and nothing else.

Query: {query}

Information:
{content}

Score:"""
        return self._score_number(self.generator.generate(prompt, max_new_tokens=8, temperature=0.0))

    def retrieve(self, query: str) -> list[LayerReport]:
        query_embedding = self.embedder.encode(query)
        raw = self.index.hierarchical_search(query_embedding, k=self.traversal_k)
        if not raw:
            return []
        max_layer = max(raw)
        reports: list[LayerReport] = []
        for layer in sorted(raw, reverse=True):
            items = [self._item(node_id, layer) for node_id in raw[layer]]
            if not items:
                continue
            base = self._relevance(query, items)
            weight = layer_weight(self.weighting, layer, max_layer, query, self.adaptive_threshold)
            weighted = base * weight
            if weighted >= self.relevance_threshold * 10.0:
                reports.append(LayerReport(layer, base, weighted, weight, items))
        reports.sort(key=lambda report: report.weighted_score, reverse=True)
        return reports

    def build_context(self, reports: list[LayerReport]) -> list[RetrievedItem]:
        selected: list[RetrievedItem] = []
        seen: set[str] = set()
        for report in reports:
            for item in report.items:
                if item.id in seen:
                    continue
                selected.append(item)
                seen.add(item.id)
                if len(selected) >= self.final_context_items:
                    return selected
        return selected

    def answer(self, query: str, context: list[RetrievedItem]) -> str:
        if not context:
            return "Insufficient contextual information was retrieved to answer the question."
        evidence = "\n\n".join(
            f"[Layer {item.layer} | {item.item_type}] {item.text}" for item in context
        )
        prompt = f"""Answer the biomedical question using only the retrieved HAGRAG evidence below.
Do not introduce unsupported biomedical claims. If the evidence is insufficient, state that clearly.

Question: {query}

Retrieved evidence:
{evidence}

Answer:"""
        return self.generator.generate(prompt)

    def query(self, query: str) -> dict:
        reports = self.retrieve(query)
        context = self.build_context(reports)
        response = self.answer(query, context)
        return {
            "query": query,
            "response": response,
            "weighting": self.weighting,
            "layers": [report.layer for report in reports],
            "filtered_layers": [
                {
                    "layer": report.layer,
                    "base_score": report.base_score,
                    "weight": report.weight,
                    "weighted_score": report.weighted_score,
                }
                for report in reports
            ],
            "context": [item.__dict__ for item in context],
        }
