from __future__ import annotations

import os
import re
from pathlib import Path

from agent_poc.agent.types import RegisteredTool, ToolSource
from agent_poc.config.loader import AgentPocConfig

_PROMPT_PATH = Path(__file__).parent.parent / "agent" / "prompts" / "nlp_to_cypher.txt"

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


def _fetch_schema(session) -> str:
    result = session.run("CALL db.schema.nodeTypeProperties()")
    rows = result.data()

    # Group properties by label
    by_label: dict[str, list[str]] = {}
    for row in rows:
        label = row.get("nodeType", "Unknown")
        prop = row.get("propertyName")
        if prop:
            by_label.setdefault(label, []).append(prop)

    # Collect relationship types
    rel_result = session.run(
        "CALL db.schema.relTypeProperties() YIELD relType RETURN DISTINCT relType"
    )
    rel_types = [r["relType"] for r in rel_result.data()]

    lines = ["Node labels and properties:"]
    for label, props in sorted(by_label.items()):
        lines.append(f"  {label}: {', '.join(sorted(props))}")
    if rel_types:
        lines.append("Relationship types:")
        lines.append("  " + ", ".join(sorted(rel_types)))

    schema_str = "\n".join(lines)
    # Truncate to stay well under 2000 chars
    if len(schema_str) > 1900:
        schema_str = schema_str[:1900] + "\n  [truncated]"
    return schema_str


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
        driver = None
        try:
            from neo4j import GraphDatabase

            uri = os.environ["NEO4J_URI"]
            username = os.environ["NEO4J_USERNAME"]
            password = os.environ["NEO4J_PASSWORD"]
            driver = GraphDatabase.driver(uri, auth=(username, password))

            with driver.session() as session:
                schema_str = _fetch_schema(session)

            prompt_template = _PROMPT_PATH.read_text()
            prompt = prompt_template.format(schema=schema_str, question=question)

            tool_config = config.model_copy(deep=True)
            raw_model = config.cypher_tool.model
            resolved_model = (raw_model if raw_model and "${" not in raw_model else "") or config.model.model_name
            from agent_poc.models.factory import make_backend
            backend = make_backend(
                tool_config,
                provider=config.cypher_tool.provider,
                model_override=resolved_model,
            )
            response = backend.complete([{"role": "user", "content": prompt}], tools=[])
            cypher = _strip_fences(response.content or "")

            if not cypher:
                return "Error: model returned an empty response."

            with driver.session() as session:
                result = session.run(cypher)
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
