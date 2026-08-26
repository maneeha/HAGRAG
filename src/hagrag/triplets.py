from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from .data import TextChunk
from .io_utils import ensure_dir, read_json, write_json

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EntityRecord:
    name: str
    entity_type: str
    description: str
    attributes: dict
    document_id: str


@dataclass(frozen=True)
class RelationshipRecord:
    source: str
    target: str
    relation_type: str
    description: str
    attributes: dict
    document_id: str


def _json_candidates(text: str) -> Iterable[str]:
    cleaned = re.sub(r"<\|.*?\|>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"```(?:json)?\s*|```", "", cleaned, flags=re.IGNORECASE).strip()
    yield cleaned
    for opening, closing in (("{", "}"), ("[", "]")):
        start = cleaned.find(opening)
        end = cleaned.rfind(closing)
        if 0 <= start < end:
            yield cleaned[start : end + 1]


def parse_triplet_response(text: str, document_id: str = "") -> tuple[list[EntityRecord], list[RelationshipRecord]]:
    data = None
    for candidate in _json_candidates(text):
        try:
            data = json.loads(candidate)
            break
        except json.JSONDecodeError:
            continue
    if data is None:
        return [], []

    if isinstance(data, list):
        if len(data) == 2 and all(isinstance(item, list) for item in data):
            entity_items, relationship_items = data
        else:
            entity_items = [item for item in data if isinstance(item, dict) and "name" in item]
            relationship_items = [
                item for item in data if isinstance(item, dict) and "source" in item and "target" in item
            ]
    elif isinstance(data, dict):
        entity_items = data.get("entities") or []
        relationship_items = data.get("relationships") or data.get("relations") or []
    else:
        return [], []

    entities: list[EntityRecord] = []
    for item in entity_items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        if not name:
            continue
        entities.append(
            EntityRecord(
                name=name,
                entity_type=str(item.get("type") or item.get("entity_type") or "Unknown").strip(),
                description=str(item.get("description") or "").strip(),
                attributes=item.get("attributes") if isinstance(item.get("attributes"), dict) else {},
                document_id=document_id,
            )
        )

    relationships: list[RelationshipRecord] = []
    for item in relationship_items:
        if not isinstance(item, dict):
            continue
        source = str(item.get("source") or "").strip()
        target = str(item.get("target") or "").strip()
        if not source or not target:
            continue
        relationships.append(
            RelationshipRecord(
                source=source,
                target=target,
                relation_type=str(item.get("type") or item.get("relation_type") or "RELATED_TO").strip(),
                description=str(item.get("description") or "").strip(),
                attributes=item.get("attributes") if isinstance(item.get("attributes"), dict) else {},
                document_id=document_id,
            )
        )
    return entities, relationships


class TripletExtractor:
    def __init__(self, generator, checkpoint_dir: str | Path):
        self.generator = generator
        self.checkpoint_dir = ensure_dir(checkpoint_dir)
        self.progress_file = self.checkpoint_dir / "triplet_progress.json"
        self.cache_file = self.checkpoint_dir / "triplets.jsonl"
        self.processed = set()
        if self.progress_file.exists():
            self.processed = set(read_json(self.progress_file).get("processed", []))

    @staticmethod
    def _prompt(chunk: TextChunk) -> str:
        return f"""Extract biomedical entities and relationships from the text below.
Return valid JSON only with this schema:
{{
  "entities": [
    {{"name": "", "type": "", "description": "", "attributes": {{}}}}
  ],
  "relationships": [
    {{"source": "", "target": "", "type": "", "description": "", "attributes": {{}}}}
  ]
}}
Use at most 5 entities and 3 relationships. Do not add information that is not stated in the text.

Text:
{chunk.text}
"""

    def _cached(self) -> dict[str, tuple[list[EntityRecord], list[RelationshipRecord]]]:
        cached = {}
        if not self.cache_file.exists():
            return cached
        with self.cache_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                entities = [EntityRecord(**item) for item in row.get("entities", [])]
                relationships = [RelationshipRecord(**item) for item in row.get("relationships", [])]
                cached[row["chunk_id"]] = (entities, relationships)
        return cached

    def extract(self, chunks: list[TextChunk]) -> tuple[list[EntityRecord], list[RelationshipRecord]]:
        cached = self._cached()
        all_entities: list[EntityRecord] = []
        all_relationships: list[RelationshipRecord] = []
        for chunk in tqdm(chunks, desc="Extracting graph triplets"):
            if chunk.chunk_id in cached:
                entities, relationships = cached[chunk.chunk_id]
            else:
                response = self.generator.generate(self._prompt(chunk), max_new_tokens=700, temperature=0.0)
                entities, relationships = parse_triplet_response(response, chunk.document_id)
                with self.cache_file.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "chunk_id": chunk.chunk_id,
                                "entities": [asdict(item) for item in entities],
                                "relationships": [asdict(item) for item in relationships],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                self.processed.add(chunk.chunk_id)
                write_json(self.progress_file, {"processed": sorted(self.processed)})
            all_entities.extend(entities)
            all_relationships.extend(relationships)
        return all_entities, all_relationships
