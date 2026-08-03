"""Integration tests for graph_pipeline/neo4j_writer.py.

Requires a live local Neo4j instance. Run with:
    pytest --integration tests/integration/test_neo4j_writer.py

Each test uses a unique label prefix (derived from a UUID) to avoid colliding with
real graph data. Nodes and constraints created during tests are cleaned up in teardown.
"""
import os
import uuid

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
async def driver():
    """Return an authenticated async Neo4j driver; skip if credentials are missing."""
    neo4j = pytest.importorskip("neo4j", reason="neo4j package not installed")
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    username = os.environ.get("NEO4J_USERNAME", "neo4j")
    password = os.environ.get("NEO4J_PASSWORD", "")
    if not password:
        pytest.skip("NEO4J_PASSWORD not set")
    drv = neo4j.AsyncGraphDatabase.driver(uri, auth=(username, password))
    yield drv
    await drv.close()


@pytest.fixture()
def label_prefix():
    """Unique label prefix per test to avoid cross-test or cross-run collisions."""
    return f"Test{uuid.uuid4().hex[:8].capitalize()}"


@pytest.fixture()
async def cleanup(driver, label_prefix):
    """Yield label_prefix, then delete all nodes with that label prefix after the test."""
    yield label_prefix
    async with driver.session() as session:
        for suffix in ["Node", "Rel", "NodeA", "NodeB", "Batch"]:
            label = f"{label_prefix}{suffix}"
            await session.run(f"MATCH (n:{label}) DETACH DELETE n")
            await session.run(
                f"DROP CONSTRAINT {label}_id IF EXISTS"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_test_node(label, uid, name, dataset_id="test_ds"):
    from graph_pipeline.models import ExtractionSource, Node
    return Node(
        id=f"{dataset_id}:{uid}",
        label=label,
        properties={"name": name},
        source_record_id=uid,
        extraction_source=ExtractionSource.RULE_BASED,
    )


def make_test_rel(from_id, to_id, from_label, to_label, rel_type="TEST_REL"):
    from graph_pipeline.models import ExtractionSource, Relationship
    return Relationship(
        from_id=from_id,
        to_id=to_id,
        from_label=from_label,
        to_label=to_label,
        type=rel_type,
        properties={},
        source_record_id=from_id.split(":")[-1],
        extraction_source=ExtractionSource.RULE_BASED,
    )


async def count_nodes(driver, label):
    async with driver.session() as s:
        result = await s.run(f"MATCH (n:{label}) RETURN count(n) AS c")
        record = await result.single()
        return record["c"]


async def count_rels(driver, rel_type):
    async with driver.session() as s:
        result = await s.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS c")
        record = await result.single()
        return record["c"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestWriteAll:
    async def test_write_all_correct_counts(self, driver, cleanup):
        """write_all returns correct created counts for a fresh graph."""
        from graph_pipeline.neo4j_writer import write_all

        label = f"{cleanup}Node"
        rel_label_a = f"{cleanup}NodeA"
        rel_label_b = f"{cleanup}NodeB"

        nodes = [
            make_test_node(label, "n-001", "Alpha"),
            make_test_node(label, "n-002", "Beta"),
        ]
        rels = [
            make_test_rel(
                f"test_ds:n-001", f"test_ds:n-002",
                label, label,
                rel_type=f"REL_{cleanup.upper()}",
            )
        ]

        result = await write_all(nodes, rels, driver, batch_size=500)

        assert result.nodes_created == 2
        assert result.nodes_matched == 0
        assert result.relationships_created == 1
        assert result.relationships_matched == 0
        assert result.errors == []

    async def test_write_all_idempotent(self, driver, cleanup):
        """Re-running the same write produces nodes_created=0, nodes_matched=N."""
        from graph_pipeline.neo4j_writer import write_all

        label = f"{cleanup}Node"
        nodes = [
            make_test_node(label, "n-001", "Alpha"),
            make_test_node(label, "n-002", "Beta"),
        ]

        await write_all(nodes, [], driver, batch_size=500)
        result = await write_all(nodes, [], driver, batch_size=500)

        assert result.nodes_created == 0
        assert result.nodes_matched == 2
        assert result.errors == []

    async def test_write_all_nodes_present_in_graph(self, driver, cleanup):
        """Nodes actually appear in Neo4j after write_all."""
        from graph_pipeline.neo4j_writer import write_all

        label = f"{cleanup}Node"
        nodes = [
            make_test_node(label, "n-001", "Alpha"),
            make_test_node(label, "n-002", "Beta"),
            make_test_node(label, "n-003", "Gamma"),
        ]

        await write_all(nodes, [], driver, batch_size=500)
        assert await count_nodes(driver, label) == 3

    async def test_write_all_relationships_present_in_graph(self, driver, cleanup):
        """Relationships actually appear in Neo4j after write_all."""
        from graph_pipeline.neo4j_writer import write_all

        label = f"{cleanup}Node"
        rel_type = f"REL_{cleanup.upper()}"
        nodes = [
            make_test_node(label, "n-001", "Alpha"),
            make_test_node(label, "n-002", "Beta"),
        ]
        rels = [make_test_rel("test_ds:n-001", "test_ds:n-002", label, label, rel_type)]

        await write_all(nodes, rels, driver, batch_size=500)
        assert await count_rels(driver, rel_type) == 1

    async def test_write_all_relationship_idempotent(self, driver, cleanup):
        """Re-running write_all with same relationship: relationships_created=0, matched=1."""
        from graph_pipeline.neo4j_writer import write_all

        label = f"{cleanup}Node"
        rel_type = f"REL_{cleanup.upper()}"
        nodes = [
            make_test_node(label, "n-001", "Alpha"),
            make_test_node(label, "n-002", "Beta"),
        ]
        rels = [make_test_rel("test_ds:n-001", "test_ds:n-002", label, label, rel_type)]

        await write_all(nodes, rels, driver, batch_size=500)
        result = await write_all(nodes, rels, driver, batch_size=500)

        assert result.relationships_created == 0
        assert result.relationships_matched == 1

    async def test_write_result_dataclass_fields(self, driver, cleanup):
        """WriteResult has the expected fields."""
        from graph_pipeline.neo4j_writer import WriteResult, write_all

        label = f"{cleanup}Node"
        nodes = [make_test_node(label, "n-001", "Alpha")]
        result = await write_all(nodes, [], driver, batch_size=500)

        assert isinstance(result, WriteResult)
        assert hasattr(result, "nodes_created")
        assert hasattr(result, "nodes_matched")
        assert hasattr(result, "relationships_created")
        assert hasattr(result, "relationships_matched")
        assert hasattr(result, "errors")
        assert isinstance(result.errors, list)


@pytest.mark.integration
class TestBatchBehaviour:
    async def test_small_batch_size_all_nodes_written(self, driver, cleanup):
        """Batch size smaller than total node count; all nodes still written."""
        from graph_pipeline.neo4j_writer import write_all

        label = f"{cleanup}Batch"
        nodes = [make_test_node(label, f"n-{i:03d}", f"Node {i}") for i in range(7)]

        result = await write_all(nodes, [], driver, batch_size=3)

        assert result.nodes_created == 7
        assert result.errors == []
        assert await count_nodes(driver, label) == 7

    async def test_mid_batch_failure_earlier_batches_intact(self, driver, cleanup):
        """Batch 1 is committed before batch 2 starts; a batch 2 failure leaves batch 1 intact."""
        from graph_pipeline.neo4j_writer import create_constraints, write_nodes

        label = f"{cleanup}Batch"
        await create_constraints([label], driver)

        first_batch = [make_test_node(label, f"n-{i:03d}", f"Node {i}") for i in range(3)]
        await write_nodes(first_batch, driver, batch_size=500)
        assert await count_nodes(driver, label) == 3

        from unittest.mock import AsyncMock, MagicMock

        failing_driver = MagicMock()
        failing_session = AsyncMock()
        failing_session.__aenter__.return_value = failing_session
        failing_session.__aexit__.return_value = False
        failing_session.begin_transaction.side_effect = RuntimeError("Simulated Neo4j failure")
        failing_driver.session.return_value = failing_session

        second_batch = [make_test_node(label, f"x-{i:03d}", f"Extra {i}") for i in range(3)]
        result = await write_nodes(second_batch, failing_driver, batch_size=500)

        assert await count_nodes(driver, label) == 3
        assert len(result.errors) >= 1


def _make_mock_driver(fail_on_batch_index: int | None = None):
    """Return (driver, call_counter) where call_counter['n'] tracks begin_transaction calls.

    If fail_on_batch_index is set, tx.commit() raises RuntimeError on that batch.
    All other batches succeed and report nodes_created=1 per statement.
    """
    from unittest.mock import AsyncMock, MagicMock

    call_counter = {"n": 0}

    async def _begin_transaction():
        idx = call_counter["n"]
        call_counter["n"] += 1

        tx = AsyncMock()

        async def _run(cypher, **kwargs):
            rows = kwargs.get("rows", [])
            row_count = len(rows) if rows else 1
            summary = MagicMock()
            summary.counters.nodes_created = row_count
            summary.counters.relationships_created = row_count
            run_result = AsyncMock()
            run_result.consume.return_value = summary
            return run_result

        tx.run.side_effect = _run

        if idx == fail_on_batch_index:
            tx.commit.side_effect = RuntimeError(f"Simulated failure on batch {idx}")
        return tx

    session = AsyncMock()
    session.__aenter__.return_value = session
    session.__aexit__.return_value = False
    session.begin_transaction.side_effect = _begin_transaction

    driver = MagicMock()
    driver.session.return_value = session
    return driver, call_counter


class TestFailSoftBehaviour:
    """Verify fail-soft batch behaviour using mock drivers — no live Neo4j required."""

    async def test_all_batches_attempted_when_middle_fails(self):
        """When the middle batch fails, the remaining batch is still attempted."""
        from graph_pipeline.neo4j_writer import write_nodes

        # 6 nodes → 3 batches of 2; batch index 1 (middle) will fail
        nodes = [make_test_node("Label", f"n{i}", f"Node {i}") for i in range(6)]
        driver, call_counter = _make_mock_driver(fail_on_batch_index=1)

        await write_nodes(nodes, driver, batch_size=2)

        assert call_counter["n"] == 3, (
            f"Expected 3 begin_transaction calls (one per batch), got {call_counter['n']}"
        )

    async def test_error_recorded_for_failing_batch(self):
        """A batch failure produces exactly one error entry in WriteResult.errors."""
        from graph_pipeline.neo4j_writer import write_nodes

        nodes = [make_test_node("Label", f"n{i}", f"Node {i}") for i in range(4)]
        driver, _ = _make_mock_driver(fail_on_batch_index=0)

        result = await write_nodes(nodes, driver, batch_size=2)

        assert len(result.errors) == 1
        assert "Batch 0 failed" in result.errors[0]
        assert "Simulated failure" in result.errors[0]

    async def test_successful_batches_counted_despite_middle_failure(self):
        """nodes_created reflects the two successful batches, not the failed one."""
        from graph_pipeline.neo4j_writer import write_nodes

        # 6 nodes → 3 batches of 2; batch 1 fails, batches 0 and 2 succeed (2 nodes each)
        nodes = [make_test_node("Label", f"n{i}", f"Node {i}") for i in range(6)]
        driver, _ = _make_mock_driver(fail_on_batch_index=1)

        result = await write_nodes(nodes, driver, batch_size=2)

        assert result.nodes_created == 4
        assert len(result.errors) == 1

    async def test_write_relationships_all_batches_attempted_on_failure(self):
        """write_relationships continues past a failing batch."""
        from graph_pipeline.neo4j_writer import write_relationships
        from graph_pipeline.models import ExtractionSource, Relationship

        def _make_rel(i):
            return Relationship(
                from_id=f"ds:n{i}",
                to_id=f"ds:n{i + 1}",
                from_label="Label",
                to_label="Label",
                type="REL",
                properties={},
                source_record_id=f"n{i}",
                extraction_source=ExtractionSource.RULE_BASED,
            )

        rels = [_make_rel(i) for i in range(6)]
        driver, call_counter = _make_mock_driver(fail_on_batch_index=0)

        await write_relationships(rels, driver, batch_size=2)

        assert call_counter["n"] == 3

    async def test_write_all_attempts_relationships_despite_node_errors(self):
        """write_all calls write_relationships even when write_nodes produced errors."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from graph_pipeline.neo4j_writer import WriteResult, write_all
        from graph_pipeline.models import ExtractionSource, Node, Relationship

        node_result = WriteResult(nodes_created=2, errors=["Batch 0 failed: timeout"])
        rel_result = WriteResult(relationships_created=1)

        nodes = [Node(
            id="ds:n1", label="L", properties={},
            source_record_id="n1", extraction_source=ExtractionSource.RULE_BASED,
        )]
        rels = [Relationship(
            from_id="ds:n1", to_id="ds:n2", from_label="L", to_label="L",
            type="REL", properties={}, source_record_id="n1",
            extraction_source=ExtractionSource.RULE_BASED,
        )]

        with patch("graph_pipeline.neo4j_writer.write_nodes", new=AsyncMock(return_value=node_result)), \
             patch("graph_pipeline.neo4j_writer.write_relationships", new=AsyncMock(return_value=rel_result)) as mock_rels, \
             patch("graph_pipeline.neo4j_writer.create_constraints", new=AsyncMock()):
            result = await write_all(nodes, rels, MagicMock())

        mock_rels.assert_called_once()
        assert result.relationships_created == 1
        assert result.errors == ["Batch 0 failed: timeout"]


@pytest.mark.integration
class TestCreateConstraints:
    async def test_constraint_created(self, driver, cleanup):
        """create_constraints does not raise and the constraint exists afterwards."""
        from graph_pipeline.neo4j_writer import create_constraints

        label = f"{cleanup}Node"
        await create_constraints([label], driver)

        async with driver.session() as session:
            result = await session.run(
                "SHOW CONSTRAINTS WHERE labelsOrTypes = [$label]",
                label=label,
            )
            constraints = await result.data()
        assert len(constraints) >= 1

    async def test_constraint_idempotent(self, driver, cleanup):
        """Running create_constraints twice does not raise."""
        from graph_pipeline.neo4j_writer import create_constraints

        label = f"{cleanup}Node"
        await create_constraints([label], driver)
        await create_constraints([label], driver)  # should not raise
