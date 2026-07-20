from __future__ import annotations

import os
import re
import time
from pathlib import Path

from agent_poc.agent.types import RegisteredTool, ToolSource
from agent_poc.config.loader import AgentPocConfig

_PROMPT_PATH = Path(__file__).parent.parent / "agent" / "prompts" / "nlp_to_cypher.txt"

_CYPHER_NOISE_RE = re.compile(
    r"'[^'\\]*(?:\\.[^'\\]*)*'"   # single-quoted strings
    r'|"[^"\\]*(?:\\.[^"\\]*)*"'  # double-quoted strings
    r"|//[^\n]*"                   # line comments
    r"|/\*.*?\*/"                  # block comments
    r"|\[[^\]]*\]",                # relationship brackets (strips rel types like [:ReusableStep])
    re.DOTALL,
)

# Keyed by Neo4j URI → (schema_str, expires_at)
_SCHEMA_CACHE: dict[str, tuple[str, float]] = {}

_INPUT_SCHEMA = {
    "type": "object",
    "required": ["question"],
    "properties": {
        "question": {
            "type": "string",
            "description": "Natural language question to answer from the Neo4j graph",
        }
    },
}


def _fetch_rel_patterns(session) -> list[str]:
    """Return one representative pattern per relationship type, e.g. '(:TestCase)-[:ReusableStep]->(:ReuseableTestStepBlock)'."""
    try:
        rows = session.run("CALL db.schema.visualization()").data()
    except Exception:
        return []
    if not rows:
        return []
    # Collect all non-Entity patterns per rel type, then pick one per type
    by_rel: dict[str, list[tuple[str, str]]] = {}
    for rel in rows[0].get("relationships", []):
        from_name = rel[0].get("name", "")
        rel_type = rel[1]
        to_name = rel[2].get("name", "")
        if from_name == "Entity" or to_name == "Entity":
            continue
        by_rel.setdefault(rel_type, []).append((from_name, to_name))
    lines = []
    for rel_type in sorted(by_rel):
        from_name, to_name = by_rel[rel_type][0]
        lines.append(f"  (:{from_name})-[:{rel_type}]->(:{to_name})")
    return lines


def _fetch_schema(session, budget: int = 1900) -> str:
    result = session.run("CALL db.schema.nodeTypeProperties()")
    rows = result.data()

    by_label: dict[str, list[str]] = {}
    for row in rows:
        label = row.get("nodeType", "Unknown")
        prop = row.get("propertyName")
        if prop:
            by_label.setdefault(label, []).append(prop)

    rel_patterns = _fetch_rel_patterns(session)

    # Tier 1: node label names only
    tier1_lines = ["Node labels:"] + [f"  {label}" for label in sorted(by_label)]
    tier1 = "\n".join(tier1_lines)

    # Tier 2: relationship patterns (endpoint-aware, replaces the bare rel-types list)
    if rel_patterns:
        tier2 = tier1 + "\nRelationship patterns:\n" + "\n".join(rel_patterns)
    else:
        tier2 = tier1

    if len(tier2) >= budget:
        return tier2[:budget]

    # Tier 3: per-label property details, one label at a time
    prop_lines: list[str] = []
    truncated = False
    for label in sorted(by_label):
        props = sorted(by_label[label])
        line = f"  {label}: {', '.join(props)}"
        candidate = tier2 + "\nNode label properties:\n" + "\n".join(prop_lines + [line])
        if len(candidate) > budget:
            truncated = True
            break
        prop_lines.append(line)

    if not prop_lines:
        return tier2

    schema_str = tier2 + "\nNode label properties:\n" + "\n".join(prop_lines)
    if truncated:
        schema_str += "\n  [property details truncated]"
    return schema_str


def _get_cached_schema(session, uri: str, budget: int, ttl: float) -> str:
    entry = _SCHEMA_CACHE.get(uri)
    if entry is not None and time.monotonic() < entry[1]:
        return entry[0]
    schema_str = _fetch_schema(session, budget=budget)
    _SCHEMA_CACHE[uri] = (schema_str, time.monotonic() + ttl)
    return schema_str


def _strip_cypher_noise(cypher: str) -> str:
    return _CYPHER_NOISE_RE.sub("", cypher)


def _extract_labels(cypher: str) -> set[str]:
    """Extract node labels used in a Cypher query."""
    return set(re.findall(r':([A-Z][A-Za-z0-9_]*)', _strip_cypher_noise(cypher)))


_NEO4J_PREFIX_RE = re.compile(r'^(?:Neo\.\S+|org\.neo4j\.\S+):\s*')


def _trim_neo4j_error(exc: Exception, max_chars: int = 300) -> str:
    first_line = next(
        (ln for ln in str(exc).splitlines() if ln.strip()),
        str(exc),
    )
    first_line = _NEO4J_PREFIX_RE.sub("", first_line.strip())
    return first_line[:max_chars]


def _make_undirected(cypher: str) -> str:
    """Strip relationship direction arrows so queries don't fail due to wrong direction."""
    cypher = re.sub(r'\]->', ']-', cypher)   # (a)-[:T]->(b) → (a)-[:T]-(b)
    cypher = re.sub(r'<-\[', '-[', cypher)   # (a)<-[:T]-(b) → (a)-[:T]-(b)
    return cypher


def _known_labels(schema_str: str) -> set[str]:
    """Extract valid labels from the schema string."""
    labels = set()
    in_labels_section = False
    for line in schema_str.splitlines():
        if line.startswith("Node labels:"):
            in_labels_section = True
            continue
        if in_labels_section:
            # Label lines are indented; unindented lines are section headers → stop
            if not line.startswith(" "):
                in_labels_section = False
                continue
            # Labels may be compound (e.g. :`Entity`:`ReuseableTestStepBlock`); extract each
            for m in re.finditer(r'`([A-Za-z][A-Za-z0-9_]*)`', line):
                labels.add(m.group(1))
    return labels


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text).rstrip("`").strip()
    return text


def _format_results(records: list[dict]) -> str:
    if not records:
        return "No results found."
    lines = []
    for i, record in enumerate(records, 1):
        pairs = ", ".join(f"{k}={v}" for k, v in record.items())
        lines.append(f"Row {i}: {pairs}")
    lines.append(f"({len(records)} result{'s' if len(records) != 1 else ''})")
    return "\n".join(lines)


def make_cypher_tool(config: AgentPocConfig) -> RegisteredTool:
    def _query_graph(args: dict) -> str:
        question: str = args["question"]
        print(f"[cypher_tool] question received: {question!r}", flush=True)
        driver = None
        try:
            from neo4j import GraphDatabase

            uri = os.environ["NEO4J_URI"]
            username = os.environ["NEO4J_USERNAME"]
            password = os.environ["NEO4J_PASSWORD"]
            driver = GraphDatabase.driver(uri, auth=(username, password))

            with driver.session() as session:
                schema_str = _get_cached_schema(
                    session, uri,
                    budget=config.cypher_tool.schema_budget,
                    ttl=config.cypher_tool.schema_ttl_seconds,
                )

            prompt_template = _PROMPT_PATH.read_text()
            prompt = prompt_template.replace("{schema}", schema_str).replace("{question}", question)

            tool_config = config.model_copy(deep=True)
            raw_model = config.cypher_tool.model
            resolved_model = (raw_model if raw_model and "${" not in raw_model else "") or config.model.model_name
            from agent_poc.models.factory import make_backend
            backend = make_backend(
                tool_config,
                provider=config.cypher_tool.provider,
                model_override=resolved_model,
            )
            if config.cypher_tool.provider == "tricentis":
                user_content = [{"type": "text", "text": prompt, "cache_control": {"type": "ephemeral"}}]
            else:
                user_content = prompt
            messages = [{"role": "user", "content": user_content}]
            response = backend.complete(messages, tools=[])
            cypher = _make_undirected(_strip_fences(response.content or ""))

            if not cypher:
                return "Error: model returned an empty response."

            # Retry once if the generated Cypher uses labels not in the schema
            unknown = _extract_labels(cypher) - _known_labels(schema_str)
            if unknown:
                print(f"[cypher_tool] unknown labels {unknown}, retrying", flush=True)
                correction = (
                    f"\n\nCORRECTION: The labels {sorted(unknown)} do not exist in the schema. "
                    f"Valid labels are: {sorted(_known_labels(schema_str))}. "
                    f"Rewrite the query using only labels from the schema."
                )
                retry_content = prompt + correction
                messages = [{"role": "user", "content": retry_content}]
                response = backend.complete(messages, tools=[])
                cypher = _make_undirected(_strip_fences(response.content or ""))
                if not cypher:
                    return "Error: model returned an empty response on retry."

            if re.search(r'\$[a-zA-Z_]\w*', cypher):
                return f"Error: generated Cypher contains query parameters which are not supported. Generated query was: {cypher}"

            with driver.session() as session:
                try:
                    result = session.run(cypher)
                    records = [dict(record) for record in result]
                except Exception as cypher_exc:
                    err = _trim_neo4j_error(cypher_exc)
                    print(f"[cypher_tool] Cypher error, retrying: {err}", flush=True)
                    messages = messages + [
                        {"role": "assistant", "content": response.content},
                        {"role": "user", "content": (
                            f"The Cypher query you generated produced an error: {err}. "
                            f"Rewrite the query to fix this error."
                        )},
                    ]
                    response = backend.complete(messages, tools=[])
                    cypher = _make_undirected(_strip_fences(response.content or ""))
                    if not cypher:
                        return "Error: model returned an empty response on retry."
                    with driver.session() as session2:
                        result = session2.run(cypher)
                        records = [dict(record) for record in result]

            return _format_results(records)

        except KeyError as exc:
            return f"Error: missing required environment variable {exc}."
        except Exception as exc:
            return f"Error: {exc}"
        finally:
            if driver is not None:
                driver.close()

    return RegisteredTool(
        name="query_graph",
        description=(
            "Answer a natural language question by querying the Neo4j knowledge graph. "
            "Generates and executes a Cypher query internally — do not write Cypher yourself."
        ),
        input_schema=_INPUT_SCHEMA,
        callable=_query_graph,
        source=ToolSource.GENERATED,
        timeout_seconds=config.cypher_tool.timeout_seconds,
    )
