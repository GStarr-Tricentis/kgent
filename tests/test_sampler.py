"""Tests for graph_pipeline/sampler.py.

Run with: pytest tests/test_sampler.py
"""
import random


# ---------------------------------------------------------------------------
# Fixtures helpers
# ---------------------------------------------------------------------------

def make_records(type_counts: dict[str, int], extra_keys: dict | None = None) -> list[dict]:
    """Build a synthetic list[dict] with the given typeName distribution."""
    records = []
    for type_name, count in type_counts.items():
        for i in range(count):
            r = {"uniqueId": f"{type_name}-{i}", "typeName": type_name, "name": f"{type_name} {i}"}
            if extra_keys:
                r.update(extra_keys)
            records.append(r)
    return records


def make_records_with_nested(n: int = 5, array_len: int = 10) -> list[dict]:
    """Records that each have a nested array of objects longer than 3."""
    return [
        {
            "uniqueId": f"r{i}",
            "typeName": "TestCase",
            "steps": [{"step": j, "action": "click"} for j in range(array_len)],
            "tags": ["a", "b", "c", "d"],  # array of scalars — not truncated
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# sample_records — stratified sampling
# ---------------------------------------------------------------------------

class TestSampleRecords:
    def test_returns_list_of_dicts(self):
        from graph_pipeline.sampler import sample_records
        records = make_records({"TestCase": 20})
        result = sample_records(records, n=10)
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_at_least_3_per_type(self):
        from graph_pipeline.sampler import sample_records
        records = make_records({"TestCase": 30, "XModule": 30, "Folder": 30})
        result = sample_records(records, n=20)
        counts = {}
        for r in result:
            counts[r["typeName"]] = counts.get(r["typeName"], 0) + 1
        for type_name in ["TestCase", "XModule", "Folder"]:
            assert counts.get(type_name, 0) >= 3, f"{type_name} has fewer than 3 records"

    def test_fewer_than_3_includes_all(self):
        from graph_pipeline.sampler import sample_records
        records = make_records({"RareType": 2, "CommonType": 50})
        result = sample_records(records, n=20)
        rare_count = sum(1 for r in result if r["typeName"] == "RareType")
        assert rare_count == 2  # both included since only 2 exist

    def test_total_does_not_exceed_n(self):
        from graph_pipeline.sampler import sample_records
        records = make_records({"A": 100, "B": 100})
        result = sample_records(records, n=30)
        assert len(result) <= 30

    def test_returns_all_when_fewer_than_n(self):
        from graph_pipeline.sampler import sample_records
        records = make_records({"A": 5})
        result = sample_records(records, n=50)
        assert len(result) == 5

    def test_no_typename_falls_back_to_random(self):
        from graph_pipeline.sampler import sample_records
        records = [{"id": i, "value": i * 2} for i in range(100)]
        result = sample_records(records, n=20)
        assert len(result) == 20
        assert all("id" in r for r in result)

    def test_empty_input(self):
        from graph_pipeline.sampler import sample_records
        assert sample_records([], n=50) == []

    def test_nested_arrays_of_objects_truncated_to_3(self):
        from graph_pipeline.sampler import sample_records
        records = make_records_with_nested(n=5, array_len=10)
        result = sample_records(records, n=10)
        for r in result:
            assert len(r["steps"]) <= 3, "nested object array should be capped at 3"

    def test_scalar_arrays_not_truncated(self):
        from graph_pipeline.sampler import sample_records
        records = make_records_with_nested(n=3, array_len=10)
        result = sample_records(records, n=10)
        for r in result:
            # tags is an array of strings (scalars), not objects — leave untouched
            assert len(r["tags"]) == 4

    def test_stratified_proportional_within_budget(self):
        """Larger types should contribute more records than smaller ones."""
        from graph_pipeline.sampler import sample_records
        records = make_records({"Big": 60, "Small": 10})
        result = sample_records(records, n=30)
        big_count = sum(1 for r in result if r["typeName"] == "Big")
        small_count = sum(1 for r in result if r["typeName"] == "Small")
        assert big_count > small_count

    def test_no_duplicate_records_in_sample(self):
        """No record appears twice in the output regardless of sample size or type distribution."""
        from graph_pipeline.sampler import sample_records
        records = make_records({"A": 10, "B": 10, "C": 10})
        # Run many times because sampling is random
        for _ in range(30):
            result = sample_records(records, n=15)
            unique_ids = {r["uniqueId"] for r in result}
            assert len(unique_ids) == len(result), (
                f"Duplicate records in sample: {len(result)} items, "
                f"{len(unique_ids)} unique IDs"
            )


# ---------------------------------------------------------------------------
# summarize_structure
# ---------------------------------------------------------------------------

class TestSummarizeStructure:
    def test_returns_string(self):
        from graph_pipeline.sampler import summarize_structure
        records = make_records({"TestCase": 5})
        assert isinstance(summarize_structure(records), str)

    def test_contains_top_level_keys(self):
        from graph_pipeline.sampler import summarize_structure
        records = make_records({"TestCase": 3})
        summary = summarize_structure(records)
        assert "uniqueId" in summary
        assert "typeName" in summary
        assert "name" in summary

    def test_contains_typename_distribution(self):
        from graph_pipeline.sampler import summarize_structure
        records = make_records({"TestCase": 5, "XModule": 3})
        summary = summarize_structure(records)
        assert "TestCase" in summary
        assert "XModule" in summary
        assert "5" in summary
        assert "3" in summary

    def test_identifies_fk_fields(self):
        from graph_pipeline.sampler import summarize_structure
        records = [
            {"uniqueId": "a1", "typeName": "TestCase", "moduleUniqueId": "m1", "parentId": "p1"},
            {"uniqueId": "a2", "typeName": "TestCase", "moduleUniqueId": "m2", "parentId": "p2"},
        ]
        summary = summarize_structure(records)
        assert "moduleUniqueId" in summary
        assert "parentId" in summary

    def test_identifies_nested_array_fields(self):
        from graph_pipeline.sampler import summarize_structure
        records = make_records_with_nested(n=3, array_len=5)
        summary = summarize_structure(records)
        assert "steps" in summary

    def test_empty_records(self):
        from graph_pipeline.sampler import summarize_structure
        result = summarize_structure([])
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Generic (non-Tosca) sampling
# ---------------------------------------------------------------------------

class TestGenericSampling:
    def test_stratified_sampling_custom_type_field(self):
        """Explicit type_field='kind' stratifies correctly on non-Tosca data."""
        from graph_pipeline.sampler import sample_records
        records = [{"id": str(i), "kind": "A" if i % 2 == 0 else "B"} for i in range(40)]
        result = sample_records(records, n=20, type_field="kind")
        kinds = {r["kind"] for r in result}
        assert "A" in kinds and "B" in kinds

    def test_heuristic_type_field_detection(self):
        """Heuristic detects 'type' field when type_field is not passed explicitly."""
        from graph_pipeline.sampler import sample_records
        records = [{"id": str(i), "type": "X" if i % 2 == 0 else "Y"} for i in range(40)]
        result = sample_records(records, n=20)
        types = {r["type"] for r in result}
        assert "X" in types and "Y" in types
