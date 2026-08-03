"""Unit tests for graph_pipeline/neo4j_writer.py WriteBuffer (no live Neo4j required)."""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_node(i, label="Label"):
    from graph_pipeline.models import ExtractionSource, Node
    return Node(
        id=f"ds:n{i}",
        label=label,
        properties={"name": f"node{i}"},
        source_record_id=f"n{i}",
        extraction_source=ExtractionSource.RULE_BASED,
    )


def _make_rel(i, label="Label"):
    from graph_pipeline.models import ExtractionSource, Relationship
    return Relationship(
        from_id=f"ds:n{i}",
        to_id=f"ds:n{i + 1}",
        from_label=label,
        to_label=label,
        type="REL",
        properties={},
        source_record_id=f"n{i}",
        extraction_source=ExtractionSource.RULE_BASED,
    )


def _make_driver():
    """Return (driver, session) mocks suitable for WriteBuffer tests."""
    session = AsyncMock()
    session.__aenter__.return_value = session
    session.__aexit__.return_value = False
    driver = MagicMock()
    driver.session.return_value = session
    return driver, session


# ---------------------------------------------------------------------------
# WriteBuffer tests
# ---------------------------------------------------------------------------

class TestWriteBuffer:
    async def test_write_buffer_flushes_on_node_batch_size(self):
        """Auto-flush fires when _nodes reaches batch_size; the +1 node is buffered."""
        from graph_pipeline.neo4j_writer import WriteBuffer

        driver, _ = _make_driver()
        flush_calls: list[list[str]] = []

        async def mock_write_nodes_to_session(nodes, session, batch_size, result):
            flush_calls.append([n.id for n in nodes])

        batch_size = 3
        with patch("graph_pipeline.neo4j_writer._write_nodes_to_session",
                   side_effect=mock_write_nodes_to_session):
            async with WriteBuffer(driver, batch_size=batch_size) as buf:
                for i in range(batch_size + 1):
                    await buf.add_node(_make_node(i))
                # one auto-flush fired (at node #3); the 4th node is still buffered
                assert len(flush_calls) == 1
                assert len(flush_calls[0]) == batch_size

    async def test_write_buffer_flushes_on_rel_batch_size(self):
        """Auto-flush fires when _rels reaches batch_size."""
        from graph_pipeline.neo4j_writer import WriteBuffer

        driver, _ = _make_driver()
        flush_calls: list[list[str]] = []

        async def mock_write_rels_to_session(rels, session, batch_size, result):
            flush_calls.append([r.from_id for r in rels])

        batch_size = 2
        with patch("graph_pipeline.neo4j_writer._write_rels_to_session",
                   side_effect=mock_write_rels_to_session):
            async with WriteBuffer(driver, batch_size=batch_size) as buf:
                for i in range(batch_size + 1):
                    await buf.add_rel(_make_rel(i))
                assert len(flush_calls) == 1
                assert len(flush_calls[0]) == batch_size

    async def test_write_buffer_flush_all_writes_remainder(self):
        """flush_all() writes nodes that never triggered an auto-flush."""
        from graph_pipeline.neo4j_writer import WriteBuffer

        driver, _ = _make_driver()
        flushed_ids: list[str] = []

        async def mock_write_nodes_to_session(nodes, session, batch_size, result):
            flushed_ids.extend(n.id for n in nodes)

        with patch("graph_pipeline.neo4j_writer._write_nodes_to_session",
                   side_effect=mock_write_nodes_to_session):
            async with WriteBuffer(driver, batch_size=10) as buf:
                for i in range(3):                  # 3 < batch_size=10 → no auto-flush
                    await buf.add_node(_make_node(i))
                assert flushed_ids == []            # nothing written yet
                await buf.flush_all()
                assert len(flushed_ids) == 3        # all 3 flushed explicitly

    async def test_write_buffer_accumulates_result(self):
        """result.nodes_created reflects totals across all flushes."""
        from graph_pipeline.neo4j_writer import WriteBuffer

        driver, _ = _make_driver()

        async def mock_write_nodes_to_session(nodes, session, batch_size, result):
            result.nodes_created += 1               # +1 per flush call, not per node

        with patch("graph_pipeline.neo4j_writer._write_nodes_to_session",
                   side_effect=mock_write_nodes_to_session):
            async with WriteBuffer(driver, batch_size=2) as buf:
                await buf.add_node(_make_node(0))
                await buf.add_node(_make_node(1))   # len == batch_size → flush #1
                await buf.add_node(_make_node(2))   # buffered; flushed by __aexit__
            # __aexit__ calls flush_all() → flush #2
            assert buf.result.nodes_created == 2

    async def test_write_buffer_single_session_across_flushes(self):
        """Every flush call receives the same session instance."""
        from graph_pipeline.neo4j_writer import WriteBuffer

        driver, _ = _make_driver()
        captured_sessions: list = []

        async def mock_write_nodes_to_session(nodes, session, batch_size, result):
            captured_sessions.append(session)

        with patch("graph_pipeline.neo4j_writer._write_nodes_to_session",
                   side_effect=mock_write_nodes_to_session):
            async with WriteBuffer(driver, batch_size=2) as buf:
                for i in range(5):                  # triggers 2 auto-flushes; 1 via __aexit__
                    await buf.add_node(_make_node(i))

        assert len(captured_sessions) == 3
        assert all(s is captured_sessions[0] for s in captured_sessions)

    async def test_write_buffer_context_manager_closes_session_on_error(self):
        """Session __aexit__ is called even when an exception propagates."""
        from graph_pipeline.neo4j_writer import WriteBuffer

        driver, mock_session = _make_driver()

        with pytest.raises(RuntimeError, match="simulated error"):
            async with WriteBuffer(driver, batch_size=5) as buf:
                raise RuntimeError("simulated error")

        mock_session.__aexit__.assert_called_once()
