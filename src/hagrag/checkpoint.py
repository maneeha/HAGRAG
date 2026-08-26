from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from .hierarchy import Community


def save_embeddings(path: str | Path, embeddings: dict[str, np.ndarray]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(embeddings, handle)


def load_embeddings(path: str | Path) -> dict[str, np.ndarray]:
    with Path(path).open("rb") as handle:
        value = pickle.load(handle)
    return {str(key): np.asarray(vector, dtype=np.float32) for key, vector in value.items()}


def save_hierarchy(path: str | Path, hierarchy: dict[int, list[Community]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    serializable = {str(layer): [community.to_dict() for community in values] for layer, values in hierarchy.items()}
    target.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def load_hierarchy(path: str | Path) -> dict[int, list[Community]]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(layer): [Community.from_dict(item) for item in values] for layer, values in value.items()}
