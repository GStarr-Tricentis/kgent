from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import yaml
from openai import OpenAI

from graph_pipeline.context_store import (
    AssociationConfig,
    DatasetContext,
    DatasetNodeType,
    DatasetRelationshipType,
    HierarchyConfig,
    ImplicitRelationship,
    NestedCollection,
    SharedContext,
)
from graph_pipeline.sampler import summarize_structure

logger = logging.getLogger(__name__)

_NODES_PROMPT_PATH = Path(__file__).parent / "prompts" / "schema_proposal_nodes.txt"
_RELS_PROMPT_PATH = Path(__file__).parent / "prompts" / "schema_proposal_relationships.txt"
_AMBIGUOUS_PROMPT_PATH = Path(__file__).parent / "prompts" / "schema_proposal_ambiguous.txt"

_SHARED_CONTEXT_CHAR_BUDGET = 6_000 * 4  # ~6K tokens before switching to condensed form


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _serialize_shared_context(shared_context: SharedContext) -> str:
    full = yaml.dump(shared_context.model_dump(), allow_unicode=True, sort_keys=False)
    if len(full) <= _SHARED_CONTEXT_CHAR_BUDGET:
        return full
    condensed: dict = {
        "node_types": [{"name": nt.name, "maps_to": nt.maps_to} for nt in shared_context.node_types],
        "relationship_types": [
            {"name": rt.name, "maps_to": rt.maps_to, "from": rt.from_type, "to": rt.to_type}
            for rt in shared_context.relationship_types
        ],
    }
    return yaml.dump(condensed, allow_unicode=True, sort_keys=False)


def _clean_llm_output(raw: str) -> str:
    """Strip think blocks, markdown fences, and extract the first JSON value."""
    text = raw.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    # Extract first JSON array or object
    match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if match:
        text = match.group(0)
    return text.strip()


def _strip_llm_wrapper(raw: str) -> str:
    """Strip think blocks and markdown fences; return inner text (may contain multiple JSON values)."""
    text = raw.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return text.strip()


def _extract_json_values(text: str) -> list:
    """Extract all top-level JSON values (arrays or objects) from text using incremental decode."""
    decoder = json.JSONDecoder()
    values = []
    i = 0
    while i < len(text):
        while i < len(text) and text[i] in " \t\n\r":
            i += 1
        if i >= len(text):
            break
        try:
            val, end = decoder.raw_decode(text, i)
            values.append(val)
            i = end
        except json.JSONDecodeError:
            i += 1
    return values


def _llm_call(client: OpenAI, model: str, prompt: str, max_retries: int, label: str) -> str:
    """Call the LLM, retrying on empty responses. Returns raw content."""
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.choices[0].message.content or ""
            logger.debug("%s attempt %d raw response:\n%s", label, attempt, raw)
            if raw.strip():
                return raw
            last_error = ValueError("empty response")
        except Exception as exc:
            logger.warning("%s attempt %d error: %s", label, attempt, exc)
            last_error = exc
    raise RuntimeError(f"{label} failed after {max_retries} attempts. Last error: {last_error}")


# ---------------------------------------------------------------------------
# Call 1: node types
# ---------------------------------------------------------------------------

def _propose_node_types(
    sample: list[dict],
    shared_context: SharedContext,
    client: OpenAI,
    model: str,
    max_retries: int,
) -> tuple[list[DatasetNodeType], dict]:
    """Return (node_types, structural_config). structural_config is {} if the LLM omits it."""
    template = _NODES_PROMPT_PATH.read_text(encoding="utf-8")
    prompt = template.format(
        shared_context_yaml=_serialize_shared_context(shared_context),
        structure_summary=summarize_structure(sample),
        sample_records_json=json.dumps(sample[:10], indent=2, ensure_ascii=False),
    )

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        raw = _llm_call(client, model, prompt, max_retries=1, label=f"node_types attempt {attempt}")
        try:
            text = _strip_llm_wrapper(raw)
            values = _extract_json_values(text)
            if not values:
                raise ValueError("no JSON found in response")
            first = values[0]
            if not isinstance(first, list):
                raise ValueError(f"expected JSON array as first value, got {type(first).__name__}")
            node_types = [DatasetNodeType(**item) for item in first]

            structural_config: dict = {}
            if len(values) >= 2 and isinstance(values[1], dict):
                structural_config = values[1]

            logger.info("node_types succeeded on attempt %d: %d types", attempt, len(node_types))
            return node_types, structural_config
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("node_types attempt %d failed: %s", attempt, exc)
            last_error = exc

    raise RuntimeError(f"node type proposal failed after {max_retries} attempts. Last error: {last_error}")


# ---------------------------------------------------------------------------
# Call 2: relationship types
# ---------------------------------------------------------------------------

def _propose_relationship_types(
    sample: list[dict],
    node_types: list[DatasetNodeType],
    client: OpenAI,
    model: str,
    max_retries: int,
) -> tuple[list[DatasetRelationshipType], list[ImplicitRelationship], dict]:
    template = _RELS_PROMPT_PATH.read_text(encoding="utf-8")
    node_types_json = json.dumps(
        [{"name": nt.name, "maps_to": nt.maps_to} for nt in node_types],
        indent=2,
    )
    # Pick the 10 most structurally rich records — most keys + nested objects/arrays.
    # Richer records are more likely to expose FK fields, nested references, or association arrays
    # regardless of what the data format looks like.
    def _richness(r: dict) -> int:
        score = len(r)
        for v in r.values():
            if isinstance(v, dict):
                score += len(v)
            elif isinstance(v, list) and v and isinstance(v[0], dict):
                score += len(v) * 2
        return score

    rel_sample = sorted(sample, key=_richness, reverse=True)[:50]

    prompt = template.format(
        node_types_json=node_types_json,
        structure_summary=summarize_structure(sample),
        sample_records_json=json.dumps(rel_sample, indent=2, ensure_ascii=False),
    )

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        raw = _llm_call(client, model, prompt, max_retries=1, label=f"rel_types attempt {attempt}")
        try:
            text = _clean_llm_output(raw)
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError(f"expected a JSON object, got {type(data).__name__}")

            rel_types = [
                DatasetRelationshipType(**item)
                for item in data.get("relationship_types", [])
            ]
            implicit_rels = [
                ImplicitRelationship(**item)
                for item in data.get("implicit_relationships", [])
            ]
            assoc_raw = data.get("association_config")
            assoc_config: dict = assoc_raw if isinstance(assoc_raw, dict) else {}
            logger.info(
                "rel_types succeeded on attempt %d: %d rel types, %d implicit",
                attempt, len(rel_types), len(implicit_rels),
            )
            return rel_types, implicit_rels, assoc_config
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("rel_types attempt %d failed: %s", attempt, exc)
            last_error = exc

    raise RuntimeError(f"relationship type proposal failed after {max_retries} attempts. Last error: {last_error}")


# ---------------------------------------------------------------------------
# Call 3: ambiguous fields
# ---------------------------------------------------------------------------

def _propose_ambiguous_fields(
    sample: list[dict],
    client: OpenAI,
    model: str,
    max_retries: int,
) -> list[str]:
    """Return field names whose values may contain implicit entity/relationship references."""
    template = _AMBIGUOUS_PROMPT_PATH.read_text(encoding="utf-8")
    prompt = template.format(
        structure_summary=summarize_structure(sample),
        sample_records_json=json.dumps(sample[:10], indent=2, ensure_ascii=False),
    )

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        raw = _llm_call(client, model, prompt, max_retries=1, label=f"ambiguous_fields attempt {attempt}")
        try:
            text = _clean_llm_output(raw)
            data = json.loads(text)
            if not isinstance(data, list):
                raise ValueError(f"expected JSON array, got {type(data).__name__}")
            fields = [f for f in data if isinstance(f, str)]
            logger.info("ambiguous_fields succeeded on attempt %d: %s", attempt, fields)
            return fields
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning("ambiguous_fields attempt %d failed: %s", attempt, exc)
            last_error = exc

    logger.warning("ambiguous_fields failed after %d attempts; defaulting to []. Last error: %s", max_retries, last_error)
    return []


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def propose_dataset_context(
    sample: list[dict],
    shared_context: SharedContext,
    model: str,
    ollama_base_url: str,
    dataset_id: str = "new_dataset",
    max_retries: int = 3,
) -> DatasetContext:
    """Propose a DatasetContext via two focused LLM calls: nodes first, then relationships.

    Splitting into two calls produces more reliable JSON output from smaller models.
    Both calls use temperature=0 for deterministic output.
    """
    import datetime
    client = OpenAI(base_url=ollama_base_url, api_key="ollama")

    node_types, structural_config = _propose_node_types(sample, shared_context, client, model, max_retries)
    rel_types, implicit_rels, assoc_config_dict = _propose_relationship_types(
        sample, node_types, client, model, max_retries
    )
    ambiguous_fields = _propose_ambiguous_fields(sample, client, model, max_retries)

    # hierarchy_config
    hierarchy_config: HierarchyConfig | None = None
    hc_raw = structural_config.get("hierarchy_config")
    if isinstance(hc_raw, dict) and hc_raw.get("field"):
        try:
            hierarchy_config = HierarchyConfig(**{k: v for k, v in hc_raw.items() if v is not None})
        except Exception as exc:
            logger.warning("Could not parse hierarchy_config: %s", exc)

    # nested_collections
    nested_collections: list[NestedCollection] = []
    for nc in structural_config.get("nested_collections", []):
        if isinstance(nc, dict):
            try:
                nested_collections.append(NestedCollection(**nc))
            except Exception as exc:
                logger.warning("Could not parse nested_collection entry: %s", exc)

    # association_config
    association_config: AssociationConfig | None = None
    if assoc_config_dict:
        try:
            association_config = AssociationConfig(**assoc_config_dict)
        except Exception as exc:
            logger.warning("Could not parse association_config: %s", exc)

    return DatasetContext(
        dataset_id=dataset_id,
        source_file="",
        generated_at=str(datetime.date.today()),
        id_field=structural_config.get("id_field") or "uniqueId",
        type_field=structural_config.get("type_field") or "typeName",
        node_types=node_types,
        relationship_types=rel_types,
        implicit_relationships=implicit_rels,
        nested_collections=nested_collections,
        association_config=association_config,
        hierarchy_config=hierarchy_config,
        ambiguous_fields=ambiguous_fields,
    )


def validate_proposed_context(ctx: DatasetContext, sample: list[dict]) -> list[str]:
    """Validate a proposed DatasetContext against the sample. Returns warning strings."""
    warnings: list[str] = []

    sample_type_names = {r.get(ctx.type_field) for r in sample if ctx.type_field in r}
    proposed_canonical_labels = {nt.maps_to for nt in ctx.node_types}

    for nt in ctx.node_types:
        if nt.name not in sample_type_names:
            warnings.append(
                f"Node type '{nt.name}' not found in sample typeNames "
                f"(known: {sorted(sample_type_names)})"
            )

    for rt in ctx.relationship_types:
        if rt.from_type and rt.from_type not in proposed_canonical_labels and rt.from_type != "any":
            warnings.append(
                f"Relationship '{rt.name}' from_type '{rt.from_type}' "
                f"is not in proposed node labels {sorted(proposed_canonical_labels)}"
            )
        if rt.to_type and rt.to_type not in proposed_canonical_labels and rt.to_type != "any":
            warnings.append(
                f"Relationship '{rt.name}' to_type '{rt.to_type}' "
                f"is not in proposed node labels {sorted(proposed_canonical_labels)}"
            )

    return warnings
