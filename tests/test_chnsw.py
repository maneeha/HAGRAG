import numpy as np

from hagrag.chnsw import CHNSWIndex


def unit(index: int, dimension: int = 4):
    value = np.zeros(dimension, dtype=np.float32)
    value[index] = 1.0
    return value


def test_hierarchical_search_returns_candidates():
    index = CHNSWIndex(embedding_dim=4, M=2, ef_search=20)
    index.add_node("e0", unit(0), 0, {"type": "entity"})
    index.add_node("e1", unit(1), 0, {"type": "entity"})
    index.add_node("c0", unit(0), 1, {"type": "community"})
    index.add_node("c1", unit(1), 1, {"type": "community"})
    index.build_intra_layer_links()
    index.build_inter_layer_links()
    result = index.hierarchical_search(unit(0), k=1)
    assert result[1] == ["c0"]
    assert result[0] == ["e0"]
