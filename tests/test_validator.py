"""Tests for graph_pipeline/validator.py.

Run with: pytest tests/test_validator.py
All tests run in dry-run mode (driver=None) unless noted.
"""
from graph_pipeline.models import ExtractionSource, Node, Relationship


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def make_node(id, label="TestCase", source_record_id=None):
    return Node(
        id=id,
        label=label,
        properties={"name": "test"},
        source_record_id=source_record_id or id.split(":")[-1],
        extraction_source=ExtractionSource.RULE_BASED,
    )


def make_rel(from_id, to_id, type="COVERS", from_label="TestCase", to_label="Requirement"):
    return Relationship(
        from_id=from_id,
        to_id=to_id,
        from_label=from_label,
        to_label=to_label,
        type=type,
        properties={},
        source_record_id=from_id.split(":")[-1],
        extraction_source=ExtractionSource.RULE_BASED,
    )


def make_shared_ctx(node_type_names=None):
    from graph_pipeline.context_store import SharedContext, SharedNodeType
    node_types = [
        SharedNodeType(name=n, maps_to=n, source_datasets=["ds1"])
        for n in (node_type_names or [])
    ]
    return SharedContext(version=1, node_types=node_types)


# ---------------------------------------------------------------------------
# check_referential_integrity — dry-run (driver=None)
# ---------------------------------------------------------------------------

class TestReferentialIntegrityDryRun:
    async def test_all_endpoints_present_no_errors(self):
        from graph_pipeline.validator import check_referential_integrity
        nodes = [make_node("ds1:tc-001"), make_node("ds1:req-001", label="Requirement")]
        rels = [make_rel("ds1:tc-001", "ds1:req-001")]
        errors = await check_referential_integrity(nodes, rels, driver=None)
        assert errors == []

    async def test_missing_from_id_is_warning_in_dry_run(self):
        from graph_pipeline.validator import check_referential_integrity
        nodes = [make_node("ds1:req-001", label="Requirement")]
        rels = [make_rel("ds1:tc-missing", "ds1:req-001")]
        errors = await check_referential_integrity(nodes, rels, driver=None)
        assert len(errors) == 1
        assert errors[0].severity == "warning"
        assert "ds1:tc-missing" in errors[0].message

    async def test_missing_to_id_is_warning_in_dry_run(self):
        from graph_pipeline.validator import check_referential_integrity
        nodes = [make_node("ds1:tc-001")]
        rels = [make_rel("ds1:tc-001", "ds1:req-missing")]
        errors = await check_referential_integrity(nodes, rels, driver=None)
        assert len(errors) == 1
        assert errors[0].severity == "warning"
        assert "ds1:req-missing" in errors[0].message

    async def test_both_endpoints_missing_two_warnings(self):
        from graph_pipeline.validator import check_referential_integrity
        nodes = []
        rels = [make_rel("ds1:a-missing", "ds1:b-missing")]
        errors = await check_referential_integrity(nodes, rels, driver=None)
        assert len(errors) == 2

    async def test_no_relationships_no_errors(self):
        from graph_pipeline.validator import check_referential_integrity
        nodes = [make_node("ds1:tc-001")]
        errors = await check_referential_integrity(nodes, [], driver=None)
        assert errors == []

    async def test_no_nodes_no_rels_no_errors(self):
        from graph_pipeline.validator import check_referential_integrity
        errors = await check_referential_integrity([], [], driver=None)
        assert errors == []

    async def test_returns_list_of_validation_errors(self):
        from graph_pipeline.validator import ValidationError, check_referential_integrity
        nodes = [make_node("ds1:tc-001")]
        rels = [make_rel("ds1:tc-001", "ds1:missing")]
        errors = await check_referential_integrity(nodes, rels, driver=None)
        assert all(isinstance(e, ValidationError) for e in errors)

    async def test_error_has_record_id(self):
        from graph_pipeline.validator import check_referential_integrity
        nodes = [make_node("ds1:tc-001")]
        rels = [make_rel("ds1:tc-001", "ds1:req-missing")]
        errors = await check_referential_integrity(nodes, rels, driver=None)
        assert errors[0].record_id is not None

    async def test_same_id_referenced_twice_one_warning(self):
        """Two rels pointing to the same missing id should produce one warning per unique missing id."""
        from graph_pipeline.validator import check_referential_integrity
        nodes = [make_node("ds1:tc-001")]
        rels = [
            make_rel("ds1:tc-001", "ds1:missing"),
            make_rel("ds1:tc-001", "ds1:missing"),
        ]
        errors = await check_referential_integrity(nodes, rels, driver=None)
        missing_messages = [e.message for e in errors if "ds1:missing" in e.message]
        assert len(missing_messages) == 1


# ---------------------------------------------------------------------------
# check_referential_integrity — with driver (live Neo4j)
# These are marked integration and require a live Neo4j instance.
# We test only the dry-run→warning / driver→error distinction via a mock driver.
# ---------------------------------------------------------------------------

class TestReferentialIntegrityWithDriver:
    def _make_mock_driver(self, existing_ids: set):
        """Async mock driver that returns only IDs present in existing_ids."""
        from unittest.mock import AsyncMock, MagicMock

        async def _run(query, **params):
            queried = params.get("ids", [])
            found = [i for i in queried if i in existing_ids]
            result = AsyncMock()
            result.data.return_value = [{"id": i} for i in found]
            return result

        session = AsyncMock()
        session.__aenter__.return_value = session
        session.__aexit__.return_value = False
        session.run.side_effect = _run

        driver = MagicMock()
        driver.session.return_value = session
        return driver

    async def test_missing_from_batch_but_found_in_neo4j_no_error(self):
        from graph_pipeline.validator import check_referential_integrity
        nodes = [make_node("ds1:tc-001")]
        rels = [make_rel("ds1:tc-001", "ds1:req-001")]
        driver = self._make_mock_driver({"ds1:req-001"})
        errors = await check_referential_integrity(nodes, rels, driver=driver)
        assert errors == []

    async def test_missing_from_batch_and_neo4j_is_error(self):
        from graph_pipeline.validator import check_referential_integrity
        nodes = [make_node("ds1:tc-001")]
        rels = [make_rel("ds1:tc-001", "ds1:req-ghost")]
        driver = self._make_mock_driver(set())
        errors = await check_referential_integrity(nodes, rels, driver=driver)
        assert len(errors) == 1
        assert errors[0].severity == "error"


# ---------------------------------------------------------------------------
# check_referential_integrity — label-scoped query routing
# ---------------------------------------------------------------------------

async def test_labeled_endpoints_use_label_scoped_query():
    from unittest.mock import AsyncMock, MagicMock
    from graph_pipeline.validator import check_referential_integrity

    nodes = []
    rels = [make_rel("ds1:tc-001", "ds1:req-001")]

    session = AsyncMock()
    run_result = AsyncMock()
    run_result.data = AsyncMock(return_value=[{"id": "ds1:tc-001"}, {"id": "ds1:req-001"}])
    session.run = AsyncMock(return_value=run_result)
    driver = MagicMock()
    driver.session.return_value.__aenter__ = AsyncMock(return_value=session)
    driver.session.return_value.__aexit__ = AsyncMock(return_value=False)

    errors = await check_referential_integrity(nodes, rels, driver)

    assert errors == []
    queries = [call.args[0] for call in session.run.call_args_list]
    assert all("TestCase" in q or "Requirement" in q for q in queries)
    assert not any(q.strip().startswith("UNWIND $ids AS id MATCH (n {") for q in queries)


async def test_unlabelled_endpoints_use_fallback_query():
    from unittest.mock import AsyncMock, MagicMock
    from graph_pipeline.validator import check_referential_integrity

    nodes = []
    rels = [make_rel("ds1:tc-001", "ds1:req-001", from_label="", to_label="")]

    session = AsyncMock()
    run_result = AsyncMock()
    run_result.data = AsyncMock(return_value=[])
    session.run = AsyncMock(return_value=run_result)
    driver = MagicMock()
    driver.session.return_value.__aenter__ = AsyncMock(return_value=session)
    driver.session.return_value.__aexit__ = AsyncMock(return_value=False)

    await check_referential_integrity(nodes, rels, driver)

    queries = [call.args[0] for call in session.run.call_args_list]
    assert any("MATCH (n {id: id})" in q for q in queries)


async def test_mixed_labeled_and_unlabelled_endpoints():
    from unittest.mock import AsyncMock, MagicMock
    from graph_pipeline.validator import check_referential_integrity

    nodes = []
    rels = [
        make_rel("ds1:tc-001", "ds1:req-001"),
        make_rel("ds1:unknown-001", "ds1:unknown-002", from_label="", to_label=""),
    ]

    session = AsyncMock()
    run_result = AsyncMock()
    run_result.data = AsyncMock(return_value=[])
    session.run = AsyncMock(return_value=run_result)
    driver = MagicMock()
    driver.session.return_value.__aenter__ = AsyncMock(return_value=session)
    driver.session.return_value.__aexit__ = AsyncMock(return_value=False)

    await check_referential_integrity(nodes, rels, driver)

    queries = [call.args[0] for call in session.run.call_args_list]
    assert any("TestCase" in q or "Requirement" in q for q in queries)
    assert any("MATCH (n {id: id})" in q for q in queries)


async def test_neo4j_exception_caught_all_missing_returned_as_errors():
    from unittest.mock import AsyncMock, MagicMock
    from graph_pipeline.validator import check_referential_integrity

    nodes = []
    rels = [make_rel("ds1:tc-001", "ds1:req-001")]

    session = AsyncMock()
    session.run = AsyncMock(side_effect=Exception("Neo4j unavailable"))
    driver = MagicMock()
    driver.session.return_value.__aenter__ = AsyncMock(return_value=session)
    driver.session.return_value.__aexit__ = AsyncMock(return_value=False)

    errors = await check_referential_integrity(nodes, rels, driver)

    assert len(errors) == 2
    assert all(e.severity == "error" for e in errors)


async def test_endpoints_in_batch_produce_no_errors():
    from graph_pipeline.validator import check_referential_integrity

    nodes = [make_node("ds1:tc-001"), make_node("ds1:req-001", label="Requirement")]
    rels = [make_rel("ds1:tc-001", "ds1:req-001")]

    errors = await check_referential_integrity(nodes, rels, driver=None)
    assert errors == []


# ---------------------------------------------------------------------------
# check_label_coverage
# ---------------------------------------------------------------------------

class TestLabelCoverage:
    def test_known_label_no_warning(self):
        from graph_pipeline.validator import check_label_coverage
        nodes = [make_node("ds1:tc-001", label="TestCase")]
        sc = make_shared_ctx(node_type_names=["TestCase"])
        warnings = check_label_coverage(nodes, sc)
        assert warnings == []

    def test_unknown_label_is_warning(self):
        from graph_pipeline.validator import check_label_coverage
        nodes = [make_node("ds1:tc-001", label="UnknownType")]
        sc = make_shared_ctx(node_type_names=["TestCase"])
        warnings = check_label_coverage(nodes, sc)
        assert len(warnings) == 1
        assert warnings[0].severity == "warning"
        assert "UnknownType" in warnings[0].message

    def test_phantom_folder_always_allowed(self):
        """Folder (phantom) nodes are structural artifacts and should not produce warnings."""
        from graph_pipeline.validator import check_label_coverage
        from graph_pipeline.models import ExtractionSource
        nodes = [
            Node(
                id="ds1:path:Root",
                label="Folder",
                properties={"name": "Root"},
                source_record_id="",
                extraction_source=ExtractionSource.PHANTOM,
            )
        ]
        sc = make_shared_ctx(node_type_names=["TestCase"])  # Folder not in shared_ctx
        warnings = check_label_coverage(nodes, sc)
        assert warnings == []

    def test_empty_nodes_no_warnings(self):
        from graph_pipeline.validator import check_label_coverage
        sc = make_shared_ctx()
        assert check_label_coverage([], sc) == []

    def test_returns_list_of_validation_errors(self):
        from graph_pipeline.validator import ValidationError, check_label_coverage
        nodes = [make_node("ds1:tc-001", label="Ghost")]
        sc = make_shared_ctx()
        result = check_label_coverage(nodes, sc)
        assert all(isinstance(e, ValidationError) for e in result)

    def test_duplicate_labels_one_warning(self):
        """Multiple nodes with the same unknown label produce only one warning."""
        from graph_pipeline.validator import check_label_coverage
        nodes = [
            make_node("ds1:tc-001", label="Ghost"),
            make_node("ds1:tc-002", label="Ghost"),
        ]
        sc = make_shared_ctx(node_type_names=["TestCase"])
        warnings = check_label_coverage(nodes, sc)
        ghost_warnings = [w for w in warnings if "Ghost" in w.message]
        assert len(ghost_warnings) == 1


# ---------------------------------------------------------------------------
# spot_check
# ---------------------------------------------------------------------------

class TestSpotCheck:
    def _make_records(self, n=10):
        return [
            {"uniqueId": f"tc-{i:03d}", "typeName": "TestCase", "name": f"Test {i}"}
            for i in range(n)
        ]

    def test_returns_spot_check_report(self):
        from graph_pipeline.validator import SpotCheckReport, spot_check
        nodes = [make_node(f"ds1:tc-{i:03d}") for i in range(10)]
        report = spot_check(nodes, [], self._make_records(10), n=3)
        assert isinstance(report, SpotCheckReport)

    def test_sampled_count_capped_at_n(self):
        from graph_pipeline.validator import spot_check
        nodes = [make_node(f"ds1:tc-{i:03d}") for i in range(10)]
        report = spot_check(nodes, [], self._make_records(10), n=3)
        assert len(report.sampled) == 3

    def test_node_found_true_when_present(self):
        from graph_pipeline.validator import spot_check
        nodes = [make_node(f"ds1:tc-{i:03d}") for i in range(10)]
        records = self._make_records(10)
        report = spot_check(nodes, [], records, n=5)
        assert all(r.node_found for r in report.sampled)

    def test_node_found_false_when_absent(self):
        from graph_pipeline.validator import spot_check
        # Nodes list is empty — no nodes exist
        records = [{"uniqueId": "tc-001", "typeName": "TestCase", "name": "Login"}]
        report = spot_check([], [], records, n=1)
        assert len(report.sampled) == 1
        assert report.sampled[0].node_found is False

    def test_relationships_listed_for_record(self):
        from graph_pipeline.validator import spot_check
        nodes = [make_node("ds1:tc-001"), make_node("ds1:req-001", label="Requirement")]
        rels = [make_rel("ds1:tc-001", "ds1:req-001", type="COVERS")]
        records = [{"uniqueId": "tc-001", "typeName": "TestCase", "name": "Login"}]
        report = spot_check(nodes, rels, records, n=1)
        assert report.sampled[0].record_id == "tc-001"
        assert "COVERS" in report.sampled[0].relationships

    def test_total_counts_correct(self):
        from graph_pipeline.validator import spot_check
        nodes = [make_node(f"ds1:tc-{i:03d}") for i in range(5)]
        rels = [make_rel("ds1:tc-000", "ds1:tc-001")]
        records = self._make_records(5)
        report = spot_check(nodes, rels, records, n=2)
        assert report.total_nodes == 5
        assert report.total_relationships == 1

    def test_n_larger_than_records_samples_all(self):
        from graph_pipeline.validator import spot_check
        nodes = [make_node(f"ds1:tc-{i:03d}") for i in range(3)]
        records = self._make_records(3)
        report = spot_check(nodes, [], records, n=100)
        assert len(report.sampled) == 3

    def test_empty_records_returns_empty_report(self):
        from graph_pipeline.validator import spot_check
        report = spot_check([], [], [], n=5)
        assert report.sampled == []
        assert report.total_nodes == 0
        assert report.total_relationships == 0

    def test_spot_check_record_fields(self):
        from graph_pipeline.validator import SpotCheckRecord, spot_check
        nodes = [make_node("ds1:tc-001")]
        records = [{"uniqueId": "tc-001", "typeName": "TestCase", "name": "Login"}]
        report = spot_check(nodes, [], records, n=1)
        rec = report.sampled[0]
        assert isinstance(rec, SpotCheckRecord)
        assert rec.record_id == "tc-001"
        assert isinstance(rec.node_found, bool)
        assert isinstance(rec.relationships, list)


# ---------------------------------------------------------------------------
# ValidationError dataclass
# ---------------------------------------------------------------------------

class TestValidationErrorDataclass:
    def test_fields(self):
        from graph_pipeline.validator import ValidationError
        e = ValidationError(severity="error", message="bad node", record_id="tc-001")
        assert e.severity == "error"
        assert e.message == "bad node"
        assert e.record_id == "tc-001"

    def test_record_id_can_be_none(self):
        from graph_pipeline.validator import ValidationError
        e = ValidationError(severity="warning", message="unknown label", record_id=None)
        assert e.record_id is None


# ---------------------------------------------------------------------------
# Generic (non-Tosca) validator tests
# ---------------------------------------------------------------------------

class TestGenericValidator:
    def test_spot_check_custom_id_field(self):
        """spot_check resolves nodes correctly when records use a non-default id field."""
        from graph_pipeline.validator import spot_check

        nodes = [make_node("hr:e1", label="Engineer")]
        records = [{"employee_id": "e1", "name": "Alice"}]
        report = spot_check(nodes, [], records, id_field="employee_id")
        assert len(report.sampled) == 1
        assert report.sampled[0].node_found is True
