"""scripts/ingest.py — Full graph pipeline ingestion CLI.

Usage:
    python scripts/ingest.py --file path/to/data.jsonl [--dataset-id my_dataset]
                             [--model qwen2.5:14b] [--dry-run] [--skip-review]
                             [--sample-size 50] [--batch-size 500]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import anyio

# Ensure project root is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent))


def _step(n: int, total: int, msg: str) -> None:
    print(f"\n[{n}/{total}] {msg}")


def _indent(msg: str) -> None:
    print(f"      {msg}")


def _diff_canonical_names(existing, proposed) -> list[str]:
    """Return lines describing canonical name differences between two DatasetContexts."""
    lines = []
    existing_nodes = {nt.name: nt.maps_to for nt in existing.node_types}
    proposed_nodes = {nt.name: nt.maps_to for nt in proposed.node_types}
    for name, canonical in proposed_nodes.items():
        if name in existing_nodes and existing_nodes[name] != canonical:
            lines.append(f"  node type '{name}': '{existing_nodes[name]}' → '{canonical}'")

    existing_rels = {rt.name: rt.maps_to for rt in existing.relationship_types}
    proposed_rels = {rt.name: rt.maps_to for rt in proposed.relationship_types}
    for name, canonical in proposed_rels.items():
        if name in existing_rels and existing_rels[name] != canonical:
            lines.append(f"  rel type '{name}': '{existing_rels[name]}' → '{canonical}'")

    return lines


def _schema_preview(ctx: "DatasetContext") -> str:
    """Return a human-readable string summarising a proposed DatasetContext.

    Used to print schema proposals when --dry-run suppresses saving to disk.
    """
    lines = ["      Schema proposal (not saved — dry-run):"]
    lines.append(f"        id_field: {ctx.id_field}  |  type_field: {ctx.type_field}")
    if ctx.node_types:
        lines.append(f"        node_types ({len(ctx.node_types)}):")
        for nt in ctx.node_types:
            lines.append(f"          {nt.name} → {nt.maps_to}")
    if ctx.relationship_types:
        lines.append(f"        relationship_types ({len(ctx.relationship_types)}):")
        for rt in ctx.relationship_types:
            lines.append(f"          {rt.name} ({rt.from_type} → {rt.to_type})")
    if ctx.implicit_relationships:
        impl_names = ", ".join(ir.maps_to for ir in ctx.implicit_relationships)
        lines.append(
            f"        implicit_relationships ({len(ctx.implicit_relationships)}): {impl_names}"
        )
    if ctx.ambiguous_fields:
        lines.append(f"        ambiguous_fields: {', '.join(ctx.ambiguous_fields)}")
    if ctx.ambiguous_field_rules:
        lines.append(f"        ambiguous_field_rules ({len(ctx.ambiguous_field_rules)}):")
        for rule in ctx.ambiguous_field_rules:
            direction_arrow = "→" if rule.direction == "out" else "←"
            lines.append(
                f"          {rule.field} (split on {rule.delimiter!r}) "
                f"{direction_arrow} {rule.rel_type} [{rule.to_type}]"
            )
    return "\n".join(lines)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a dataset into the Neo4j knowledge graph")
    parser.add_argument("--file", required=True, help="Path to the data file to ingest")
    parser.add_argument("--dataset-id", default=None, help="Dataset identifier (default: file stem)")
    parser.add_argument("--model", default=None, help="Override model for schema discovery")
    parser.add_argument("--dry-run", action="store_true", help="Run extraction and validation without writing to Neo4j")
    parser.add_argument("--skip-review", action="store_true", help="Skip human review if canonical names unchanged")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--force-rediscover", action="store_true",
                        help="Re-run schema discovery even if the dataset fingerprint is unchanged")
    parser.add_argument("--full-ingest", action="store_true",
                        help="Process all records regardless of per-record hash cache; "
                             "also ensures the shared context merge runs even when record hashes are unchanged")
    parser.add_argument("--prune-deleted", action="store_true",
                        help="Soft-delete nodes for records no longer present in the source file "
                             "(sets deleted_at; nodes remain in the graph)")
    parser.add_argument("--config", default="kgent/config/config.yaml")
    parser.add_argument("--provider", default="local", choices=["local", "tricentis"],
                        help="Model provider (default: local)")
    args = parser.parse_args()

    from kgent.config.loader import load_config, load_dotenv
    load_dotenv()
    config = load_config(args.config)
    gp = config.graph_pipeline

    file_path = args.file
    dataset_id = args.dataset_id or Path(file_path).stem
    from kgent.models.factory import make_backend
    backend = await make_backend(config, provider=args.provider, model_override=args.model)
    sample_size = args.sample_size or gp.default_sample_size
    batch_size = args.batch_size or gp.default_batch_size

    TOTAL_STEPS = 8

    # -------------------------------------------------------------------------
    # Step 1: Pre-scan (Pass 1) — sample, fingerprint, and diff in one stream pass
    # -------------------------------------------------------------------------
    _step(1, TOTAL_STEPS, f"Sampling {sample_size} records from {Path(file_path).name}...")
    from graph_pipeline.loaders import stream as stream_file
    from graph_pipeline.sampler import prescan
    from graph_pipeline.context_store import load_record_hashes

    stored_hashes = {} if args.full_ingest else load_record_hashes(dataset_id)
    scan = prescan(
        stream_file(file_path),
        id_field="uniqueId",  # Tosca default; overridden after schema discovery
        stored_hashes=stored_hashes,
        sample_size=sample_size,
    )
    sample = scan.sample
    fingerprint = scan.fingerprint

    if scan.type_counts:
        type_summary = ", ".join(f"{t}({c})" for t, c in scan.type_counts.most_common())
    else:
        type_summary = f"{scan.total_records} records (no type field detected)"
    _indent(f"Loaded {scan.total_records} records. Types: {type_summary}")

    # -------------------------------------------------------------------------
    # Step 2: Load shared context
    # -------------------------------------------------------------------------
    _step(2, TOTAL_STEPS, "Loading shared context...")
    from graph_pipeline.context_store import load_dataset_context, load_shared_context
    shared_ctx = load_shared_context()
    _indent(f"v{shared_ctx.version}, {len(shared_ctx.node_types)} known node types")

    prior_ctx = load_dataset_context(dataset_id)

    # -------------------------------------------------------------------------
    # Step 3: Schema discovery (cached when fingerprint matches prior context)
    # -------------------------------------------------------------------------
    _step(3, TOTAL_STEPS, "Proposing dataset context...")
    ctx_path = Path(gp.context_dir) / "datasets" / f"{dataset_id}.yaml"

    if (
        prior_ctx is not None
        and prior_ctx.source_fingerprint == fingerprint
        and not args.force_rediscover
    ):
        proposed_ctx = prior_ctx
        _indent(
            "Fingerprint matches prior context — skipping schema discovery. "
            "Use --force-rediscover to override."
        )
    else:
        from graph_pipeline.schema_discovery import propose_dataset_context, validate_proposed_context
        proposed_ctx = await propose_dataset_context(
            sample=sample,
            shared_context=shared_ctx,
            backend=backend,
            dataset_id=dataset_id,
        )

        warnings = validate_proposed_context(proposed_ctx, sample)
        _indent(f"Proposed {len(proposed_ctx.node_types)} node types, "
                f"{len(proposed_ctx.relationship_types)} relationship types")
        for w in warnings:
            _indent(f"  ⚠ {w}")

        proposed_ctx.source_fingerprint = fingerprint
        if args.dry_run:
            print(_schema_preview(proposed_ctx))
            _indent("(dry-run: context not saved to disk)")
        else:
            from graph_pipeline.context_store import save_dataset_context
            save_dataset_context(proposed_ctx)
            _indent(f"dataset_context saved to {ctx_path}")

    # -------------------------------------------------------------------------
    # Step 4: Human review (or skip if unchanged)
    # -------------------------------------------------------------------------
    _step(4, TOTAL_STEPS, "Reviewing dataset context...")
    prior_version_exists = prior_ctx is not None

    require_review = True
    if args.dry_run:
        require_review = False
        _indent("(dry-run: skipping human review)")
    elif args.skip_review:
        if not prior_version_exists:
            _indent("No prior context found — review required on first ingestion.")
        else:
            diffs = _diff_canonical_names(prior_ctx, proposed_ctx)
            if diffs:
                _indent("Canonical name changes detected — review required:")
                for d in diffs:
                    _indent(d)
            else:
                require_review = False
                _indent("Canonical names unchanged — skipping review.")

    if require_review:
        print(f"\n      Edit {ctx_path} if needed, then press Enter to continue.")
        print("      (Ctrl+C to abort)")
        try:
            await anyio.to_thread.run_sync(input)
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            sys.exit(0)

    if args.dry_run:
        dataset_ctx = proposed_ctx
    else:
        dataset_ctx = load_dataset_context(dataset_id)
        if dataset_ctx is None:
            print(f"ERROR: context file not found at {ctx_path}", file=sys.stderr)
            sys.exit(1)

    # -------------------------------------------------------------------------
    # Incremental: report changed / deleted counts; early-exit if nothing to do
    # -------------------------------------------------------------------------
    unchanged_count = scan.total_records - len(scan.ingest_ids)
    if unchanged_count:
        _indent(
            f"{unchanged_count} unchanged record(s) skipped; "
            f"{len(scan.ingest_ids)} to process."
        )
    if scan.deleted_ids:
        _indent(f"{len(scan.deleted_ids)} deleted record(s) detected.")
    if not scan.ingest_ids and not (args.prune_deleted and scan.deleted_ids):
        _indent("All records unchanged — nothing to ingest.")
        print("\nDone.")
        return

    # -------------------------------------------------------------------------
    # Driver initialisation (needed by Steps 5 and 7)
    # -------------------------------------------------------------------------
    driver = None
    if not args.dry_run:
        import neo4j as _neo4j
        uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
        username = os.environ.get("NEO4J_USERNAME", "neo4j")
        password = os.environ.get("NEO4J_PASSWORD", "")
        if not password:
            print("ERROR: NEO4J_PASSWORD not set in environment or .env file", file=sys.stderr)
            sys.exit(1)
        driver = _neo4j.AsyncGraphDatabase.driver(uri, auth=(username, password))

    # -------------------------------------------------------------------------
    # Pass 2: Build extraction indices (index-build stream pass)
    # -------------------------------------------------------------------------
    from graph_pipeline.extractor import build_extraction_indices
    indices = build_extraction_indices(
        (r for r in stream_file(file_path)
         if str(r.get(dataset_ctx.id_field, "")) in scan.ingest_ids),
        dataset_ctx,
    )

    # -------------------------------------------------------------------------
    # Step 5: Streaming extract + write (Pass 3a: nodes, Pass 3b: rels)
    # -------------------------------------------------------------------------
    _step(5, TOTAL_STEPS, "Extracting nodes and relationships (two-pass)...")
    from graph_pipeline.extractor import extract_and_write_stream
    from graph_pipeline.neo4j_writer import WriteBuffer, WriteResult

    if dataset_ctx.ambiguous_fields and not dataset_ctx.ambiguous_field_rules:
        _indent(
            "ambiguous_fields present but no ambiguous_field_rules found — "
            "ambiguous field extraction skipped. Re-run with --force-rediscover to resolve."
        )

    write_result = WriteResult()
    if not args.dry_run:
        async with WriteBuffer(driver, batch_size=batch_size) as buffer:
            _indent("Pass 3a: writing nodes...")
            await extract_and_write_stream(
                (r for r in stream_file(file_path)
                 if str(r.get(dataset_ctx.id_field, "")) in scan.ingest_ids),
                dataset_ctx, shared_ctx, indices, buffer,
                write_rels=False,
            )
            _indent(f"Pass 3a done: {buffer.result.nodes_created} nodes created, "
                    f"{buffer.result.nodes_matched} matched")

            _indent("Pass 3b: writing relationships...")
            await extract_and_write_stream(
                (r for r in stream_file(file_path)
                 if str(r.get(dataset_ctx.id_field, "")) in scan.ingest_ids),
                dataset_ctx, shared_ctx, indices, buffer,
                write_nodes=False,
            )
            _indent(f"Pass 3b done: {buffer.result.relationships_created} rels created, "
                    f"{buffer.result.relationships_matched} matched")

        write_result = buffer.result

    # -------------------------------------------------------------------------
    # Step 6: Validate
    # -------------------------------------------------------------------------
    _step(6, TOTAL_STEPS, "Validating...")
    from graph_pipeline.validator import check_label_coverage
    from graph_pipeline.models import ExtractionSource, Node

    phantom_labels = (
        {dataset_ctx.hierarchy_config.phantom_label}
        if dataset_ctx.hierarchy_config is not None
        else set()
    )
    # In the streaming path nodes are written to Neo4j, not retained in memory.
    # Synthesise representative nodes from dataset_ctx to check label coverage.
    _probe_nodes = [
        Node(id="", label=nt.maps_to, properties={}, source_record_id="",
             extraction_source=ExtractionSource.RULE_BASED)
        for nt in dataset_ctx.node_types
    ]
    for _nc in dataset_ctx.nested_collections:
        _probe_nodes.append(Node(
            id="", label=_nc.child_label, properties={}, source_record_id="",
            extraction_source=ExtractionSource.RULE_BASED,
        ))
    label_issues = check_label_coverage(_probe_nodes, shared_ctx, phantom_labels=phantom_labels)
    warnings_labels = [e for e in label_issues if e.severity == "warning"]
    if warnings_labels:
        for w in warnings_labels:
            _indent(f"  ⚠ {w.message}")

    _indent(f"{write_result.nodes_created} nodes created, {write_result.nodes_matched} matched")
    _indent(f"{write_result.relationships_created} rels created, {write_result.relationships_matched} matched")
    if write_result.errors:
        for err in write_result.errors:
            prefix = "  ⚠" if err.startswith("Skipping relationship") else "  ✗"
            _indent(f"{prefix} {err}")

    # -------------------------------------------------------------------------
    # Step 7: Post-write: soft-delete pruned records and save hashes
    # -------------------------------------------------------------------------
    _step(7, TOTAL_STEPS, "Writing to Neo4j..." if not args.dry_run else "Writing to Neo4j... (DRY RUN — skipped)")

    if not args.dry_run:
        fatal_errors = [e for e in write_result.errors if not e.startswith("Skipping relationship")]
        if fatal_errors:
            print("\nERROR: write errors occurred.", file=sys.stderr)
            await driver.close()
            sys.exit(1)
        if args.prune_deleted and scan.deleted_ids:
            from graph_pipeline.neo4j_writer import soft_delete_nodes
            namespaced = [f"{dataset_id}:{rid}" for rid in scan.deleted_ids]
            node_labels = [nt.maps_to for nt in dataset_ctx.node_types]
            soft_deleted_count = await soft_delete_nodes(namespaced, driver, labels=node_labels)
            _indent(f"  {soft_deleted_count} node(s) soft-deleted (deleted_at set; not removed from graph)")
            _indent("  Query with: MATCH (n) WHERE n.deleted_at IS NOT NULL")
        await driver.close()
        from graph_pipeline.context_store import save_record_hashes
        save_record_hashes(dataset_id, scan.current_hashes)
    else:
        _indent("(dry run — no data written)")

    # -------------------------------------------------------------------------
    # Step 8: Merge dataset context into shared context
    # -------------------------------------------------------------------------
    _step(8, TOTAL_STEPS, "Merging dataset context into shared context...")
    if args.dry_run:
        _indent("(dry-run: skipping merge into shared context)")
    else:
        from graph_pipeline.context_store import MergeConflict, merge_into_shared, save_dataset_context

        while True:
            try:
                updated_shared = merge_into_shared(dataset_ctx)
                break
            except MergeConflict as conflict:
                print(f"\n  CONFLICT: \"{conflict.source_name}\"")
                print(f"    existing → {conflict.existing_canonical}   (from: {conflict.existing_dataset})")
                print(f"    proposed → {conflict.proposed_canonical}   (from: {conflict.new_dataset})")
                try:
                    print("  Enter canonical name to use, or Ctrl+C to abort: ", end="", flush=True)
                    chosen = (await anyio.to_thread.run_sync(input)).strip()
                except (KeyboardInterrupt, EOFError):
                    print("\nAborted.")
                    sys.exit(0)
                if not chosen:
                    continue
                if conflict.type == "node":
                    for nt in dataset_ctx.node_types:
                        if nt.name == conflict.source_name:
                            nt.maps_to = chosen
                else:
                    for rt in dataset_ctx.relationship_types:
                        if rt.name == conflict.source_name:
                            rt.maps_to = chosen
                save_dataset_context(dataset_ctx)

        new_types = [nt.name for nt in dataset_ctx.node_types
                     if nt.name not in {n.name for n in shared_ctx.node_types}]
        added_str = f" (added: {', '.join(new_types)})" if new_types else ""
        _indent(f"shared_context updated to v{updated_shared.version}{added_str}")
        _indent("shared_context saved.")

    print("\nDone.")


if __name__ == "__main__":
    anyio.run(main)
