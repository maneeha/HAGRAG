from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Iterable

import networkx as nx
from neo4j import GraphDatabase

from .config import Neo4jConfig
from .errors import ConfigurationError
from .triplets import EntityRecord, RelationshipRecord


class Neo4jStore:
    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    @classmethod
    def from_env(cls, config: Neo4jConfig) -> "Neo4jStore":
        uri = os.getenv(config.uri_env, "")
        username = os.getenv(config.username_env, "")
        password = os.getenv(config.password_env, "")
        missing = [
            name
            for name, value in (
                (config.uri_env, uri),
                (config.username_env, username),
                (config.password_env, password),
            )
            if not value
        ]
        if missing:
            raise ConfigurationError(f"Missing Neo4j environment variable(s): {', '.join(missing)}")
        return cls(uri, username, password)

    def verify(self) -> None:
        self.driver.verify_connectivity()

    def close(self) -> None:
        self.driver.close()

    def clear(self) -> None:
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n").consume()

    def write_triplets(
        self,
        entities: Iterable[EntityRecord],
        relationships: Iterable[RelationshipRecord],
    ) -> None:
        entity_rows = [
            {
                **asdict(item),
                "attributes": json.dumps(item.attributes, ensure_ascii=False),
            }
            for item in entities
        ]
        relationship_rows = [
            {
                **asdict(item),
                "attributes": json.dumps(item.attributes, ensure_ascii=False),
            }
            for item in relationships
        ]
        with self.driver.session() as session:
            if entity_rows:
                session.run(
                    """
                    UNWIND $rows AS row
                    MERGE (e:Entity {name: row.name})
                    SET e.type = row.entity_type,
                        e.description = CASE WHEN row.description <> '' THEN row.description ELSE e.description END,
                        e.attributes = row.attributes,
                        e.doc_id = row.document_id
                    """,
                    rows=entity_rows,
                ).consume()
            if relationship_rows:
                session.run(
                    """
                    UNWIND $rows AS row
                    MATCH (s:Entity {name: row.source})
                    MATCH (t:Entity {name: row.target})
                    MERGE (s)-[r:RELATION {type: row.relation_type}]->(t)
                    SET r.description = row.description,
                        r.attributes = row.attributes,
                        r.doc_id = row.document_id
                    """,
                    rows=relationship_rows,
                ).consume()

    def entity_details(self) -> dict[str, dict]:
        cache: dict[str, dict] = {}
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (e:Entity)
                RETURN e.name AS name, e.type AS type, e.description AS description,
                       e.attributes AS attributes, e.doc_id AS document_id
                """
            )
            for record in result:
                attrs = record["attributes"] or "{}"
                try:
                    attrs = json.loads(attrs) if isinstance(attrs, str) else attrs
                except json.JSONDecodeError:
                    attrs = {}
                cache[record["name"]] = {
                    "name": record["name"],
                    "type": record["type"] or "Unknown",
                    "description": record["description"] or "",
                    "attributes": attrs or {},
                    "document_id": record["document_id"] or "",
                }
        return cache

    def to_networkx(self, include_isolates: bool = False) -> nx.Graph:
        graph = nx.Graph()
        with self.driver.session() as session:
            if include_isolates:
                for record in session.run(
                    "MATCH (e:Entity) RETURN e.name AS name, e.type AS type, e.description AS description"
                ):
                    graph.add_node(
                        record["name"],
                        type=record["type"] or "Unknown",
                        description=record["description"] or "",
                    )
            for record in session.run(
                """
                MATCH (s:Entity)-[r:RELATION]->(t:Entity)
                RETURN s.name AS source, t.name AS target,
                       s.type AS source_type, s.description AS source_description,
                       t.type AS target_type, t.description AS target_description,
                       r.type AS type, r.description AS description
                """
            ):
                graph.add_node(
                    record["source"],
                    type=record["source_type"] or "Unknown",
                    description=record["source_description"] or "",
                )
                graph.add_node(
                    record["target"],
                    type=record["target_type"] or "Unknown",
                    description=record["target_description"] or "",
                )
                graph.add_edge(
                    record["source"],
                    record["target"],
                    type=record["type"] or "RELATED_TO",
                    description=record["description"] or "",
                    weight=1.0,
                )
        return graph
