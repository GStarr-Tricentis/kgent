# open-weight-agent

A synchronous tool-using agent that runs against any OpenAI-compatible local model server (Ollama, llama.cpp, vLLM, etc.).

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai) installed and running (`ollama serve`)
- A pulled model — `ollama pull qwen2.5:7b` is the default

## Install

```bash
# Core only
pip install -e .

# With Streamlit UI
pip install -e ".[ui]"

# With MCP server support
pip install -e ".[mcp]"

# Development (includes pytest)
pip install -e ".[dev,mcp]"

# With Tricentis cloud backend (requires submodule — run after git clone --recurse-submodules)
uv pip install --override-requires-python ">=3.11" -e "./tricentis-ai-client[openai]"
pip install -e ".[tricentis]"
```

> **Note:** The `tricentis-ai-client` submodule declares `requires-python = ">=3.13"` but runs
> fine on 3.11. The `--override-requires-python` flag bypasses the metadata check. If you clone
> without `--recurse-submodules`, run `git submodule update --init` first.

## Streamlit UI

A web UI with two tabs: **Chat** and **Benchmark**.

```bash
pip install -e ".[ui]"
streamlit run ui/app.py
```

### Chat tab
- **Left panel (70%)**: chat history and prompt input
- **Right panel (30%)**: tool call cards with args, output, and elapsed time, plus a run summary (iterations, finish reason, wall time, token counts)
- Model dropdown populated from `ollama list`

### Benchmark tab
Upload a queries CSV, select one or more models, set repetitions, and click **Run Benchmark**. Results appear as a table when the run completes, with a **Download CSV** button.

## Benchmark CLI

Run a set of queries across multiple models from the command line:

```bash
python scripts/benchmark.py \
  --queries queries.csv \
  --models qwen3:8b,mistral-small:latest \
  --output results.csv \
  --reps 3
```

Results are flushed to the output CSV after every run, so you can `Ctrl+C` at any point and keep what's been collected so far.

| Flag | Default | Description |
|---|---|---|
| `--queries` | required | Path to input CSV |
| `--models` | required | Comma-separated model names |
| `--output` | `benchmark_results.csv` | Output CSV path |
| `--reps` | `3` | Repetitions per query × model |
| `--config` | `agent_poc/config/config.yaml` | Agent config path |

### Queries CSV format

The input CSV must have these columns:

| Column | Required | Description |
|---|---|---|
| `id` | no | Stable identifier; row index used if absent |
| `use_case` | yes | Category label (e.g. `graph_query`, `file_ops`) |
| `query` | yes | Prompt text sent to the agent |

Example `queries.csv`:

```csv
id,use_case,query
1,graph_query,Which UI modules are invoked by the most reusable step blocks?
2,graph_query,List all nodes connected to the Entity label
3,file_ops,Read the contents of README.md and summarize it
4,reasoning,What tools do you have available?
```

### Output CSV columns

One row per `query × model × rep`:

`run_id`, `model`, `use_case`, `query_id`, `query`, `rep`, `finish_reason`, `iterations`, `wall_time_s`, `prompt_tokens`, `response_tokens`, `total_tokens`, `num_tool_calls`, `tool_names`, `tool_latencies_ms`, `mean_tool_latency_ms`, `response`, `error`

## Quick start

```bash
ollama pull qwen2.5:7b
python main.py --prompt "list the files in the current directory"
```

Interactive mode (no `--prompt`):
```bash
python main.py
> what is 2 + 2?
```

Override model without editing config:
```bash
python main.py --model qwen3:8b --prompt "hello"
```

## The four demos

### Demo 1 — File Q&A
```bash
echo -e "Paris\nTokyo\nNairobi\nSydney" > cities.txt
python main.py --prompt "Read cities.txt and tell me how many cities are listed"
```
Agent calls `read_file`, then answers from the content.

### Demo 2 — Python computation
```bash
python main.py --prompt "Use python_exec to compute the sum of the first 100 natural numbers"
```
Agent writes and runs Python code, reads the output (`5050`), reports the answer.

### Demo 3 — MCP server tools
Add a server to `agent_poc/config/config.yaml`:
```yaml
mcp:
  servers:
    - name: fs
      command: uvx
      args: ["mcp-server-filesystem", "/tmp"]
```
```bash
python main.py --prompt "list tools available"
```
MCP tools appear alongside static tools; the agent can invoke them.

### Demo 4 — Custom format parser
```bash
cat > sample.dat << 'EOF'
##name=Alice;age=30;city=NYC
##name=Bob;age=25;city=LA
EOF
python main.py --prompt "Parse sample.dat — each line starts with ## and fields are separated by ; in key=value format. How many records are there and what are the names?"
```
Agent reads the file, recognises the format, writes a parser with `python_exec`, iterates on errors, and reports the result.

## Config reference (`agent_poc/config/config.yaml`)

```yaml
model:
  provider: local           # "local" for Ollama/llama.cpp/vLLM; "tricentis" coming in PR 2
  base_url: http://localhost:11434/v1
  api_key: ollama           # required by OpenAI SDK; value ignored by Ollama
  model_name: qwen2.5:7b
  temperature: 0.0

agent:
  max_iterations: 20        # hard cap on tool-call rounds
  tool_timeout_seconds: 30  # per-tool execution timeout
  repeated_call_window: 3   # identical consecutive calls before error injection

tools:
  static:                   # which static tool groups to register
    - filesystem            # read_file, write_file, list_dir
    - shell                 # shell (subprocess, no pipes)
    - python_exec           # sandboxed Python subprocess

mcp:
  servers: []               # list of {name, command, args}

sandbox:
  timeout_seconds: 15       # python_exec subprocess timeout
  max_output_bytes: 65536   # truncate output above this size
  allow_network: false      # informational (not enforced on macOS)
```

## Graph Pipeline

Ingest any structured dataset (CSV, JSON, JSONL, SQLite) into a Neo4j knowledge graph — no code changes required for new data sources. The pipeline uses a local LLM to discover the schema, then extracts nodes and relationships according to a config-driven `DatasetContext`.

### Prerequisites

- Neo4j running locally (or set `NEO4J_URI` / `NEO4J_USERNAME` / `NEO4J_PASSWORD` in `.env`)
- Ollama running with a model pulled (default: `qwen3:8b`)

### Ingest a dataset

```bash
python scripts/ingest.py --file path/to/data.jsonl
```

On first run the pipeline will:
1. Sample records and detect structure
2. Call the LLM to propose node types, relationship types, and structural config
3. Save a `DatasetContext` YAML to `context/datasets/<dataset-id>.yaml`
4. Pause for human review of the proposed schema
5. Extract nodes and relationships
6. Validate referential integrity (dangling edges are skipped with a warning)
7. Write to Neo4j
8. Merge the dataset's types into a shared context for cross-dataset consistency

| Flag | Default | Description |
|---|---|---|
| `--file` | required | Path to the data file (CSV, JSON, JSONL, SQLite) |
| `--dataset-id` | file stem | Identifier for this dataset |
| `--model` | `qwen3:8b` | Override the LLM used for schema discovery |
| `--dry-run` | off | Run extraction and validation without writing to Neo4j |
| `--skip-review` | off | Skip the human review step if canonical names are unchanged |
| `--sample-size` | `50` | Number of records to sample for schema discovery |
| `--batch-size` | `500` | Neo4j write batch size |
| `--config` | `agent_poc/config/config.yaml` | Path to config file |

### Config (`agent_poc/config/config.yaml`)

```yaml
graph_pipeline:
  context_dir: context/         # where DatasetContext YAMLs are stored
  default_sample_size: 50
  default_batch_size: 500
  default_model: qwen3:8b
```

### How it works

The pipeline is fully config-driven via `DatasetContext` (a Pydantic model stored as YAML). Key fields:

- `id_field` / `type_field` — which record fields hold the unique ID and entity type (auto-detected by the LLM)
- `node_types` — entity labels with canonical Neo4j label mappings
- `relationship_types` — explicit FK-based edges
- `implicit_relationships` — edges inferred from matching field values across records
- `nested_collections` — child objects embedded inside parent records
- `hierarchy_config` — folder/path fields that generate phantom ancestor nodes
- `association_config` — structured association arrays (e.g. `[{edgeName, partnerId, direction}]`)

Generated context files (`context/datasets/`, `context/shared_context.yaml`) are gitignored — they are runtime artifacts, not source.

## Running tests

```bash
# Unit tests (no Ollama required)
pytest agent_poc/tests/ -v --ignore=agent_poc/tests/integration

# Integration tests (Ollama must be running)
pytest agent_poc/tests/integration/ -v -m integration

# Override model for integration tests
AGENT_MODEL=llama3.1:8b pytest agent_poc/tests/integration/ -v -m integration
```

## Known limitations

- **No true network isolation on macOS** — the sandbox subprocess runs with the same network access as the parent. Blocking outbound connections requires a firewall rule or container.
- **No MCP reconnect on failure** — MCP adapters connect once at startup and are reused for the life of the registry. If an MCP subprocess dies mid-run, there is no automatic reconnect; subsequent calls to that tool will fail.
- **Shell tool has no safelist** — `shell` runs arbitrary commands as the current user. Intended for local/trusted use only.
- **Generated tool code runs in sandbox** — only stdlib is available; third-party packages installed in the venv are not accessible from inside `python_exec`.
