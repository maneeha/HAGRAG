from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .chnsw import CHNSWIndex
from .hierarchy import Community


def build_index(
    base_embeddings: Mapping[str, np.ndarray],
    hierarchy: dict[int, list[Community]],
    embedding_dim: int = 384,
    M: int = 16,
    ef_construction: int = 200,
    ef_search: int = 50,
) -> CHNSWIndex:
    index = CHNSWIndex(embedding_dim, M, ef_construction, ef_search)

    for node_id, embedding in base_embeddings.items():
        index.add_node(node_id, embedding, layer=0, data={"type": "entity", "id": node_id})
    for layer, communities in hierarchy.items():
        for community in communities:
            index.add_node(
                community.id,
                community.embedding,
                layer=layer,
                data={
                    "type": "community",
                    "id": community.id,
                    "summary": community.summary,
                    "members": community.members,
                    "layer": layer,
                    "algorithm": community.algorithm,
                },
            )
    index.build_intra_layer_links()
    index.build_inter_layer_links()
    return index
