# kgent

A tool-using agent that runs against any OpenAI-compatible local model server (Ollama, llama.cpp, vLLM, etc.), the Tricentis AI Service (TAIS) cloud, or AWS Bedrock.

## Prerequisites

**Local provider (default):**
- Python 3.11+
- [Ollama](https://ollama.ai) installed and running (`ollama serve`)
- A pulled model — `ollama pull qwen3:8b` is the default

**Tricentis cloud provider:**
- Python 3.11+
- Access to a TAIS tenant — set `TAIS_GATEWAY_URL`, `TAIS_TENANT_NAME`, `TAIS_PRODUCT_NAME`, `KB_NODE_ID`, and `TAIS_LLM_DEPLOYMENT` in `.env`
- First run triggers a browser device flow login; subsequent runs use the cached token at `./data/tokens.json`

**AWS Bedrock provider:**
- Python 3.11+
- AWS credentials with Bedrock access — set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and optionally `AWS_SESSION_TOKEN` (for temporary credentials) in `.env`
- Models must be enabled in your AWS account via the Bedrock console under **Model access**

## Install

```bash
# Core only
pip install -e .

# With MCP server support
pip install -e ".[mcp]"

# Development (includes pytest)
pip install -e ".[dev,mcp]"

# With Tricentis cloud backend
git submodule update --init                                              # pull tricentis-ai-client
uv pip install --override-requires-python ">=3.11" -e "./tricentis-ai-client[openai,anthropic]"
pip install -e ".[tricentis]"

# With AWS Bedrock backend
pip install -e ".[bedrock]"
```

> **Note:** `tricentis-ai-client` declares `requires-python = ">=3.13"` but runs fine on 3.11.
> The `--override-requires-python` flag bypasses that metadata check.

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
| `--models` | required | Comma-separated model names (Ollama tags, TAIS deployment names, or Bedrock model IDs) |
| `--output` | `benchmark_results.csv` | Output CSV path |
| `--reps` | `3` | Repetitions per query × model |
| `--config` | `kgent/config/config.yaml` | Agent config path |
| `--provider` | `local` | `local`, `tricentis`, or `bedrock` |
| `--cypher-tool` | off | Use NLP-to-Cypher tool instead of raw Neo4j MCP tools |
| `--no-graph` | off | Disable all graph access (straight-RAG baseline) |
| `--raw-data` | — | Path to a JSONL source data dump; enables `save_as_tool` |

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

`run_id`, `model`, `use_case`, `query_id`, `query`, `rep`, `graph_mode`, `finish_reason`, `iterations`, `wall_time_s`, `prompt_tokens`, `response_tokens`, `total_tokens`, `num_tool_calls`, `tool_names`, `tool_latencies_ms`, `mean_tool_latency_ms`, `response`, `error`, `provider`, `raw_data_file`

## Quick start

```bash
ollama pull qwen3:8b
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

Use Tricentis cloud instead of local Ollama (set `TAIS_LLM_DEPLOYMENT` in `.env` first):
```bash
python main.py --provider tricentis --prompt "list the files in the current directory"
python scripts/query.py --provider tricentis --question "how many test cases are in the graph?"
python scripts/ingest.py --file data.jsonl --provider tricentis
```

Use AWS Bedrock (set `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in `.env` first):
```bash
python main.py --provider bedrock --model qwen.qwen3-coder-next --prompt "list the files in the current directory"
python main.py --provider bedrock --model deepseek.r1-v1:0 --prompt "hello"
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
Add a server to `kgent/config/config.yaml`:
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

## Config reference (`kgent/config/config.yaml`)

```yaml
model:
  provider: local           # "local" | "tricentis" | "bedrock"
  base_url: http://localhost:11434/v1
  api_key: ollama           # required by OpenAI SDK; value ignored by Ollama
  model_name: qwen3:8b
  temperature: 0.0

agent:
  max_iterations: 20        # hard cap on tool-call rounds
  tool_timeout_seconds: 30  # per-tool execution timeout

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

bedrock:
  region: "${AWS_REGION}"   # falls back to us-east-1 if unset
  model_id: ""              # default model; overridden by --model or UI selector
```

### Bedrock available models

| Model | Bedrock ID |
|---|---|
| Qwen3 Coder | `qwen.qwen3-coder-next` |
| DeepSeek R1 | `deepseek.r1-v1:0` |

Models must be enabled in your AWS account under **Bedrock → Model access** before use.

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
1. Pre-scan: reservoir-sample records, compute a dataset fingerprint, and diff per-record hashes
2. Load the shared context (cross-dataset canonical type registry)
3. Call the LLM to propose node types, relationship types, and structural config; save the result as a `DatasetContext` YAML to `context/datasets/<dataset-id>.yaml` (skipped automatically when the dataset fingerprint is unchanged — use `--force-rediscover` to override)
4. Pause for human review of the proposed schema (always required on the first ingest; skippable on re-ingestion when canonical names are unchanged)
5. Stream-extract and write: nodes first (Pass 3a), then relationships (Pass 3b) — only records whose content has changed since the last run are processed
6. Validate label coverage against the shared context
7. Soft-delete records removed from the source file (sets `deleted_at`; nodes remain in the graph) and save per-record hashes for the next incremental run
8. Merge the dataset's types into the shared context for cross-dataset consistency

| Flag | Default | Description |
|---|---|---|
| `--file` | required | Path to the data file (CSV, JSON, JSONL, SQLite) |
| `--dataset-id` | file stem | Identifier for this dataset |
| `--model` | `qwen3:8b` | Override the LLM used for schema discovery |
| `--dry-run` | off | Run extraction and validation without writing to Neo4j |
| `--skip-review` | off | Skip human review when canonical names are unchanged (review is always required on the first ingest) |
| `--force-rediscover` | off | Re-run LLM schema discovery even if the dataset fingerprint is unchanged |
| `--full-ingest` | off | Bypass the per-record hash cache and process all records |
| `--prune-deleted` | off | Soft-delete nodes for records no longer present in the source file |
| `--sample-size` | `650` | Number of records to sample for schema discovery |
| `--batch-size` | `500` | Neo4j write batch size |
| `--config` | `kgent/config/config.yaml` | Path to config file |
| `--provider` | `local` | `local`, `tricentis`, or `bedrock` |

### Config (`kgent/config/config.yaml`)

```yaml
graph_pipeline:
  context_dir: context/         # where DatasetContext YAMLs are stored
  default_sample_size: 650
  default_batch_size: 500
  default_model: qwen3:8b
```

### How it works

The pipeline is fully config-driven via `DatasetContext` (a Pydantic model stored as YAML). Key fields:

- `id_field` / `type_field` — which record fields hold the unique ID and entity type (auto-detected by the LLM)
- `node_types` — entity labels with canonical Neo4j label mappings
- `relationship_types` — explicit FK-based edges
- `implicit_relationships` — edges inferred from matching field values across records
- `path_fk_relationships` — edges where FK values are path strings matched via a pre-built path index
- `nested_collections` — child objects embedded inside parent records
- `hierarchy_config` — folder/path fields that generate phantom ancestor nodes
- `association_config` — structured association arrays (e.g. `[{edgeName, partnerId, direction}]`)
- `ambiguous_fields` / `ambiguous_field_rules` — fields whose string values may contain implicit UID references (delimiter-split, statistically validated, resolved by a dedicated LLM call)
- `property_paths` — dot-path mappings for extracting values from nested fields

Generated context files (`context/datasets/`, `context/shared_context.yaml`) are gitignored — they are runtime artifacts, not source.

## Running tests

```bash
# Unit tests (no Ollama required)
pytest kgent/tests/ -v --ignore=kgent/tests/integration

# Integration tests (Ollama must be running)
pytest kgent/tests/integration/ -v -m integration

# Override model for integration tests
AGENT_MODEL=llama3.1:8b pytest kgent/tests/integration/ -v -m integration
```

## Known limitations

- **No true network isolation on macOS** — the sandbox subprocess runs with the same network access as the parent. Blocking outbound connections requires a firewall rule or container.
- **Shell tool has no safelist** — `shell` runs arbitrary commands as the current user. Intended for local/trusted use only.
- **Generated tool code runs in sandbox** — only stdlib is available; third-party packages installed in the venv are not accessible from inside `python_exec`.
- **Graph pipeline is Unix-only** — `context_store.py` uses `fcntl` for file locking, which is not available on Windows. `scripts/ingest.py` will not run on Windows.
- **Phantom node ID collision in folder hierarchies** — `_build_hierarchy_structures` in `extractor.py` identifies phantom folder nodes by segment name alone. Two folders with the same name under different roots (e.g. `Root1/Setup` and `Root2/Setup`) will collide into a single Neo4j node. Fix tracked: use the full cumulative path as the node ID.
- **Filesystem tools have no path sandboxing** — `read_file`, `write_file`, and `list_dir` resolve paths but impose no restrictions. The agent can read or overwrite any file accessible to the current user, including `.env` and config files. This is the same class of risk as the shell tool above; both are intended for local/trusted use only.
- **Referential integrity is not checked before writing** — `check_referential_integrity()` exists in `graph_pipeline/validator.py` but is not called during ingest. Dangling relationships (edges whose endpoint nodes don't exist in the graph) are silently skipped at write time and reported as warnings in the run summary, but there is no pre-write pass that surfaces them upfront.
- **`--sample-size` must be tuned to the model's context window** — the default of 650 records is calibrated for frontier cloud models with large context windows. Relationship type discovery sends the full sample to the LLM, so running with a small local model (~32K-token context) and the default sample size will silently exceed the context window and cause truncation or errors. Reduce `--sample-size` when using a smaller model — a rough formula: `(context_tokens × 4 - 15000) / avg_record_size_chars`. For qwen3:8b (~32K tokens) with typical multi-KB records, 35–50 is a safe ceiling.
