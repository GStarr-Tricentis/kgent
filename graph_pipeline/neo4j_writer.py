from __future__ import annotations

import logging
from dataclasses import dataclass, field

from graph_pipeline.cypher_generator import (
    generate_constraint_statements,
    generate_extraction_source_index_statements,
    generate_node_merge,
    generate_node_merge_batch,
    generate_relationship_merge,
    generate_relationship_merge_batch,
)
from graph_pipeline.models import Node, Relationship

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class WriteResult:
    nodes_created: int = 0
    nodes_matched: int = 0
    relationships_created: int = 0
    relationships_matched: int = 0
    errors: list[str] = field(default_factory=list)

    def merge(self, other: WriteResult) -> None:
        """Accumulate counts from another WriteResult into this one."""
        self.nodes_created += other.nodes_created
        self.nodes_matched += other.nodes_matched
        self.relationships_created += other.relationships_created
        self.relationships_matched += other.relationships_matched
        self.errors.extend(other.errors)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _counters_from_summary(summary) -> dict[str, int]:
    """Extract node/rel created/deleted counts from a neo4j ResultSummary."""
    c = summary.counters
    return {
        "nodes_created": getattr(c, "nodes_created", 0),
        "relationships_created": getattr(c, "relationships_created", 0),
    }


async def _run_batch(
    session,
    statements: list[tuple[str, dict]],
    batch_index: int,
    result: WriteResult,
    count_key_created: str,
    count_key_matched: str,
) -> bool:
    """Execute a list of (cypher, params) tuples in a single transaction.

    Returns True on success, False on failure (writes the error into result).
    """
    tx = None
    try:
        tx = await session.begin_transaction()
        total_created = 0
        for cypher, params in statements:
            _result = await tx.run(cypher, **params)
            summary = await _result.consume()
            counts = _counters_from_summary(summary)
            total_created += counts.get(count_key_created, 0)
        await tx.commit()

        row_count = sum(len(params["rows"]) for _, params in statements)
        total_matched = row_count - total_created
        setattr(result, count_key_created, getattr(result, count_key_created) + total_created)
        setattr(result, count_key_matched, getattr(result, count_key_matched) + max(0, total_matched))
        return True
    except Exception as exc:
        if tx is not None:
            try:
                await tx.rollback()
            except Exception:
                pass
        error_msg = f"Batch {batch_index} failed: {exc}"
        logger.error(error_msg)
        result.errors.append(error_msg)
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def create_constraints(labels: list[str], driver) -> None:
    """Create uniqueness constraints and extraction_source indexes for all node labels.

    Raises on failure — do not attempt writes without constraints in place.
    """
    statements = generate_constraint_statements(labels)
    index_statements = generate_extraction_source_index_statements(labels)
    async with driver.session() as session:
        for stmt in statements + index_statements:
            try:
                await session.run(stmt)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to create constraint/index for statement '{stmt}': {exc}"
                ) from exc


async def write_nodes(
    nodes: list[Node],
    driver,
    batch_size: int = 2000,
) -> WriteResult:
    """Write nodes in batches. All batches are attempted; errors accumulate in result.errors."""
    result = WriteResult()
    if not nodes:
        return result

    by_label: dict[str, list[Node]] = {}
    for node in nodes:
        by_label.setdefault(node.label, []).append(node)

    async with driver.session() as session:
        batch_index = 0
        for label, label_nodes in by_label.items():
            cypher = generate_node_merge_batch(label)
            for i in range(0, len(label_nodes), batch_size):
                chunk = label_nodes[i : i + batch_size]
                params = {
                    "rows": [
                        {
                            "id": n.id,
                            "props": n.properties,
                            "extraction_source": n.extraction_source.value,
                        }
                        for n in chunk
                    ]
                }
                await _run_batch(
                    session, [(cypher, params)], batch_index, result,
                    count_key_created="nodes_created",
                    count_key_matched="nodes_matched",
                )
                batch_index += 1
    return result


async def write_relationships(
    rels: list[Relationship],
    driver,
    batch_size: int = 2000,
) -> WriteResult:
    """Write relationships in batches. All batches are attempted; errors accumulate in result.errors."""
    result = WriteResult()
    if not rels:
        return result

    by_triple: dict[tuple[str, str, str], list[Relationship]] = {}
    for r in rels:
        by_triple.setdefault((r.from_label, r.to_label, r.type), []).append(r)

    async with driver.session() as session:
        batch_index = 0
        for (from_label, to_label, rel_type), triple_rels in by_triple.items():
            cypher = generate_relationship_merge_batch(from_label, to_label, rel_type)
            for i in range(0, len(triple_rels), batch_size):
                chunk = triple_rels[i : i + batch_size]
                params = {
                    "rows": [
                        {
                            "from_id": r.from_id,
                            "to_id": r.to_id,
                            "extraction_source": r.extraction_source.value,
                        }
                        for r in chunk
                    ]
                }
                await _run_batch(
                    session, [(cypher, params)], batch_index, result,
                    count_key_created="relationships_created",
                    count_key_matched="relationships_matched",
                )
                batch_index += 1
    return result


async def soft_delete_nodes(
    node_ids: list[str],
    driver,
    labels: list[str] | None = None,
) -> int:
    """Set deleted_at = datetime() on nodes whose id is in node_ids.

    Pass labels (the node labels in use for this dataset) to enable indexed
    lookups. Without labels, falls back to a labelless scan — correct but slower.
    """
    if not node_ids:
        return 0
    total = 0
    async with driver.session() as session:
        if labels:
            for label in labels:
                result = await session.run(
                    f"UNWIND $ids AS id MATCH (n:{label} {{id: id}}) "
                    f"SET n.deleted_at = datetime() RETURN count(n) AS cnt",
                    ids=node_ids,
                )
                record = await result.single()
                total += record["cnt"] if record else 0
        else:
            result = await session.run(
                "UNWIND $ids AS id MATCH (n {id: id}) "
                "SET n.deleted_at = datetime() RETURN count(n) AS cnt",
                ids=node_ids,
            )
            record = await result.single()
            total = record["cnt"] if record else 0
    return total


async def write_all(
    nodes: list[Node],
    rels: list[Relationship],
    driver,
    batch_size: int = 2000,
) -> WriteResult:
    """Full write: constraints → nodes → relationships.

    Nodes are written before relationships. All batches are attempted even when
    some fail; errors accumulate in WriteResult.errors. Relationship writes that
    reference nodes from failed batches will produce their own Neo4j errors, which
    are also recorded. The caller is responsible for distinguishing fatal errors
    from skipped-relationship warnings.
    """
    labels = list({n.label for n in nodes})
    if labels:
        await create_constraints(labels, driver)

    result = WriteResult()
    node_result = await write_nodes(nodes, driver, batch_size=batch_size)
    result.merge(node_result)

    rel_result = await write_relationships(rels, driver, batch_size=batch_size)
    result.merge(rel_result)

    return result


# ---------------------------------------------------------------------------
# Session-scoped helpers — used by WriteBuffer to share one open session
# ---------------------------------------------------------------------------

async def _write_nodes_to_session(
    nodes: list[Node],
    session,
    batch_size: int,
    result: WriteResult,
) -> None:
    """Write nodes through an already-open session; mutates result in place."""
    if not nodes:
        return

    by_label: dict[str, list[Node]] = {}
    for node in nodes:
        by_label.setdefault(node.label, []).append(node)

    batch_index = 0
    for label, label_nodes in by_label.items():
        cypher = generate_node_merge_batch(label)
        for i in range(0, len(label_nodes), batch_size):
            chunk = label_nodes[i : i + batch_size]
            params = {
                "rows": [
                    {
                        "id": n.id,
                        "props": n.properties,
                        "extraction_source": n.extraction_source.value,
                    }
                    for n in chunk
                ]
            }
            await _run_batch(
                session, [(cypher, params)], batch_index, result,
                count_key_created="nodes_created",
                count_key_matched="nodes_matched",
            )
            batch_index += 1


async def _write_rels_to_session(
    rels: list[Relationship],
    session,
    batch_size: int,
    result: WriteResult,
) -> None:
    """Write relationships through an already-open session; mutates result in place."""
    if not rels:
        return

    by_triple: dict[tuple[str, str, str], list[Relationship]] = {}
    for r in rels:
        by_triple.setdefault((r.from_label, r.to_label, r.type), []).append(r)

    batch_index = 0
    for (from_label, to_label, rel_type), triple_rels in by_triple.items():
        cypher = generate_relationship_merge_batch(from_label, to_label, rel_type)
        for i in range(0, len(triple_rels), batch_size):
            chunk = triple_rels[i : i + batch_size]
            params = {
                "rows": [
                    {
                        "from_id": r.from_id,
                        "to_id": r.to_id,
                        "extraction_source": r.extraction_source.value,
                    }
                    for r in chunk
                ]
            }
            await _run_batch(
                session, [(cypher, params)], batch_index, result,
                count_key_created="relationships_created",
                count_key_matched="relationships_matched",
            )
            batch_index += 1


# ---------------------------------------------------------------------------
# WriteBuffer
# ---------------------------------------------------------------------------

class WriteBuffer:
    """Accumulate extracted nodes and relationships; flush to Neo4j in batches.

    Holds a single open session for its lifetime so repeated flushes share the
    same connection. Must be used as an async context manager:

        async with WriteBuffer(driver, batch_size=2000) as buf:
            await buf.add_node(node)
            ...
        # session closed; any buffered remainder flushed before close
    """

    def __init__(self, driver, batch_size: int = 2000) -> None:
        self._driver = driver
        self._batch_size = batch_size
        self._nodes: list[Node] = []
        self._rels: list[Relationship] = []
        self._session = None
        self.result = WriteResult()

    async def __aenter__(self) -> "WriteBuffer":
        self._session = self._driver.session()
        await self._session.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is None:
                await self.flush_all()
        finally:
            if self._session is not None:
                await self._session.__aexit__(exc_type, exc_val, exc_tb)
        return False

    async def add_node(self, node: Node) -> None:
        self._nodes.append(node)
        if len(self._nodes) >= self._batch_size:
            await self._flush_nodes()

    async def add_rel(self, rel: Relationship) -> None:
        self._rels.append(rel)
        if len(self._rels) >= self._batch_size:
            await self._flush_rels()

    async def flush_all(self) -> None:
        """Flush any remaining buffered nodes and relationships."""
        if self._nodes:
            await self._flush_nodes()
        if self._rels:
            await self._flush_rels()

    async def _flush_nodes(self) -> None:
        await _write_nodes_to_session(self._nodes, self._session, self._batch_size, self.result)
        self._nodes = []

    async def _flush_rels(self) -> None:
        await _write_rels_to_session(self._rels, self._session, self._batch_size, self.result)
        self._rels = []
