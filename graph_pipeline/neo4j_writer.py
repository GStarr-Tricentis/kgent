from __future__ import annotations

import logging
from dataclasses import dataclass, field

from graph_pipeline.cypher_generator import (
    generate_constraint_statements,
    generate_extraction_source_index_statements,
    generate_node_merge,
    generate_relationship_merge,
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

        total_matched = len(statements) - total_created
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
    batch_size: int = 500,
) -> WriteResult:
    """Write nodes in batches. All batches are attempted; errors accumulate in result.errors."""
    result = WriteResult()
    if not nodes:
        return result

    statements = [generate_node_merge(n) for n in nodes]
    batches = [statements[i : i + batch_size] for i in range(0, len(statements), batch_size)]

    async with driver.session() as session:
        for batch_index, batch in enumerate(batches):
            await _run_batch(
                session,
                batch,
                batch_index,
                result,
                count_key_created="nodes_created",
                count_key_matched="nodes_matched",
            )

    return result


async def write_relationships(
    rels: list[Relationship],
    driver,
    batch_size: int = 500,
) -> WriteResult:
    """Write relationships in batches. All batches are attempted; errors accumulate in result.errors."""
    result = WriteResult()
    if not rels:
        return result

    valid_rels = []
    for r in rels:
        if not r.from_label or not r.to_label:
            msg = f"Skipping relationship {r.type} ({r.from_id} -> {r.to_id}): missing label"
            logger.warning(msg)
            result.errors.append(msg)
        else:
            valid_rels.append(r)
    rels = valid_rels

    if not rels:
        return result

    statements = [generate_relationship_merge(r) for r in rels]
    batches = [statements[i : i + batch_size] for i in range(0, len(statements), batch_size)]

    async with driver.session() as session:
        for batch_index, batch in enumerate(batches):
            await _run_batch(
                session,
                batch,
                batch_index,
                result,
                count_key_created="relationships_created",
                count_key_matched="relationships_matched",
            )

    return result


async def soft_delete_nodes(node_ids: list[str], driver, dataset_id: str) -> int:
    """Set deleted_at = datetime() on nodes whose id is in node_ids. Returns matched count."""
    if not node_ids:
        return 0
    async with driver.session() as session:
        result = await session.run(
            "UNWIND $ids AS id MATCH (n {id: id}) SET n.deleted_at = datetime() RETURN count(n) AS cnt",
            ids=node_ids,
        )
        record = await result.single()
        return record["cnt"] if record else 0


async def write_all(
    nodes: list[Node],
    rels: list[Relationship],
    driver,
    batch_size: int = 500,
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
