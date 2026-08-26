import pytest

from hagrag.retrieval import layer_weight, query_specificity


def test_layer_weighting_matches_paper_formulas():
    assert layer_weight("abstract", 0, 3) == pytest.approx(1.0)
    assert layer_weight("abstract", 3, 3) == pytest.approx(1.5)
    assert layer_weight("specific", 0, 3) == pytest.approx(1.5)
    assert layer_weight("specific", 3, 3) == pytest.approx(1.0)
    assert layer_weight("equal", 2, 3) == pytest.approx(1.0)


def test_adaptive_weighting_uses_query_characteristics():
    assert query_specificity("What did the trial result show?") > 0
    assert query_specificity("Give an overview and explain the concept") < 0
    assert layer_weight("adaptive", 0, 3, "trial result") == pytest.approx(1.5)
    assert layer_weight("adaptive", 3, 3, "overview concept") == pytest.approx(1.5)
