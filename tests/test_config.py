from hagrag.config import load_config


def test_paper_config_loads():
    config = load_config("configs/paper.yaml")
    assert config.chunking.chunk_size == 1024
    assert config.chunking.chunk_overlap == 20
    assert config.index.embedding_dim == 384
    assert config.index.M == 16
    assert config.index.ef_construction == 200
    assert config.index.ef_search == 50
