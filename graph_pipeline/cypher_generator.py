from __future__ import annotations

from graph_pipeline.models import Node, Relationship


def generate_node_merge(node: Node) -> tuple[str, dict]:
    """Return a parameterized Cypher MERGE statement and its parameters for a Node.

    Uses MERGE on {id: $id} so re-ingestion is idempotent. Writes extraction_source
    as its string value so Cypher callers don't need to know Python enum internals.
    The caller passes the returned params dict to session.run(cypher, params).
    """
    cypher = (
        f"MERGE (n:{node.label} {{id: $id}})\n"
        f"SET n += $props\n"
        f"SET n.ingested_at = datetime()\n"
        f"SET n.extraction_source = $extraction_source"
    )
    params = {
        "id": node.id,
        "props": node.properties,
        "extraction_source": node.extraction_source.value,
    }
    return cypher, params


def generate_relationship_merge(rel: Relationship) -> tuple[str, dict]:
    """Return a parameterized Cypher MERGE statement and its parameters for a Relationship.

    Both MATCH clauses include the node label so Neo4j can use the label+id
    composite index rather than scanning the full graph. Falls back to labelless
    MATCH if the label is empty (defensive — extractors should always set labels).
    """
    from_clause = f"(a:{rel.from_label} {{id: $from_id}})" if rel.from_label else "(a {id: $from_id})"
    to_clause = f"(b:{rel.to_label} {{id: $to_id}})" if rel.to_label else "(b {id: $to_id})"
    cypher = (
        f"MATCH {from_clause}\n"
        f"MATCH {to_clause}\n"
        f"MERGE (a)-[r:{rel.type}]->(b)\n"
        f"SET r.extraction_source = $extraction_source"
    )
    params = {
        "from_id": rel.from_id,
        "to_id": rel.to_id,
        "extraction_source": rel.extraction_source.value,
    }
    return cypher, params


def generate_node_merge_batch(label: str) -> str:
    """UNWIND template for a homogeneous batch of nodes with the same label."""
    return (
        f"UNWIND $rows AS row\n"
        f"MERGE (n:{label} {{id: row.id}})\n"
        f"SET n += row.props\n"
        f"SET n.ingested_at = datetime()\n"
        f"SET n.extraction_source = row.extraction_source"
    )


def generate_relationship_merge_batch(from_label: str, to_label: str, rel_type: str) -> str:
    """UNWIND template for a homogeneous batch of relationships with the same type triple.

    Falls back to labelless MATCH if from_label or to_label is empty — correct when
    the endpoint label spans multiple concrete types (e.g. XModule + ApiModule).
    Labelless MATCH performs a full node scan; prefer a non-empty label when the
    target type is known.
    """
    from_match = (
        f"(a:{from_label} {{id: row.from_id}})" if from_label else "(a {id: row.from_id})"
    )
    to_match = (
        f"(b:{to_label} {{id: row.to_id}})" if to_label else "(b {id: row.to_id})"
    )
    return (
        f"UNWIND $rows AS row\n"
        f"MATCH {from_match}\n"
        f"MATCH {to_match}\n"
        f"MERGE (a)-[r:{rel_type}]->(b)\n"
        f"SET r.extraction_source = row.extraction_source"
    )


def generate_constraint_statements(labels: list[str]) -> list[str]:
    """Return one CREATE CONSTRAINT statement per label, enforcing id uniqueness.

    Uniqueness constraints implicitly create an index AND prevent silent duplicate
    creation — safer than a plain CREATE INDEX.
    """
    return [
        f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE"
        for label in labels
    ]


def generate_extraction_source_index_statements(labels: list[str]) -> list[str]:
    """Return one CREATE INDEX statement per label for the extraction_source property.

    Allows efficient filtering by extraction source without scanning all nodes of a label.
    Indexes are created with IF NOT EXISTS so re-running ingest is safe.
    """
    return [
        f"CREATE INDEX IF NOT EXISTS FOR (n:{label}) ON (n.extraction_source)"
        for label in labels
    ]
