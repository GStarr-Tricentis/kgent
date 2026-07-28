# Plan: Make kgent Async End-to-End

## Context

The codebase is currently synchronous throughout, but three subsystems require async internally:
- `MCPAdapter` uses the `mcp` library's async `stdio_client` / `ClientSession` context managers, and currently reconnects (spawns a subprocess + runs a full MCP handshake) on **every tool call** via `asyncio.run()`.
- `TricentisBackend` uses an `AsyncAnthropic` client and calls `asyncio.run()` on every LLM turn, plus on construction.
- Both call `asyncio.run()` inside library code, which is a deployment blocker — any async server (FastAPI, etc.) will raise `RuntimeError: This event loop is already running`.

The fix: make the entire agent stack async end-to-end. One `asyncio.run()` lives at the process boundary (the CLI entry point), nowhere else.

---

## Approach

Work bottom-up: fix the leaves first (backends, MCP adapter, tool callables), then fix the registry and runner, then fix instrumentation, then update entry points and tests. Each step is independently testable.

---

## Step-by-Step Implementation

### Step 1 — Update the `ModelBackend` protocol

**File:** `agent_poc/agent/types.py`

Change `complete()` signature in the `ModelBackend` Protocol from a sync method to an async method:

```python
class ModelBackend(Protocol):
    async def complete(
        self,
        messages: list[dict],
        tools: list[RegisteredTool],
        response_format: dict | None = None,
    ) -> ModelResponse: ...
```

This is the anchor point. Every backend must implement `async def complete()`.

---

### Step 2 — Make `OpenAICompatibleBackend` async

**File:** `agent_poc/models/openai_compatible.py`

- Replace `openai.OpenAI` with `openai.AsyncOpenAI`.
- Change `complete()` to `async def complete()`.
- `await self._client.chat.completions.create(...)`.
- No `asyncio.run()` anywhere.

---

### Step 3 — Make `BedrockBackend` async

**File:** `agent_poc/models/bedrock_backend.py`

`boto3` has no native async client. Wrap the blocking `converse()` call with `asyncio.to_thread()`:

```python
async def complete(self, messages, tools, response_format=None) -> ModelResponse:
    response = await asyncio.to_thread(self._client.converse, **kwargs)
    ...
```

`__init__` remains sync (boto3 client construction is cheap and non-blocking).

---

### Step 4 — Make `TricentisBackend` async

**File:** `agent_poc/models/tricentis_backend.py`

- Remove `asyncio.run(self._setup())` from `__init__`. Instead, add `async def setup(self)` that is the same body as the current `_setup()`.
- `__init__` should do nothing async — only store config values.
- Add a classmethod factory: `@classmethod async def create(cls, deployment, temperature) -> TricentisBackend` that constructs the instance and calls `await instance.setup()`. All callers use `await TricentisBackend.create(...)`.
- `complete()` → `async def complete()`.
- `_complete_anthropic()` → `async def _complete_anthropic()`. Remove the inner `async def _call()` wrapper and `asyncio.run(_call())`. Just `await self._async_anthropic_client.messages.create(...)` directly.
- `_complete_openai()` → `async def _complete_openai()`. The `openai.OpenAI` sync client call gets wrapped: `await asyncio.to_thread(self._client.chat.completions.create, ...)`. Or switch to `AsyncOpenAI` (preferred — consistent with Step 2).
- `_reauthenticate()` → `async def _reauthenticate()`. `await self._tais_client.authenticate()` directly — no `asyncio.run()`.
- **Auth race lock:** add `self._auth_lock = asyncio.Lock()` in `__init__`. Wrap the entire reauthentication flow (detect expired token → refresh → update client) inside `async with self._auth_lock`. Without this, two concurrent calls hitting `AuthenticationError` simultaneously will both try to reauthenticate and clobber each other's token store.
- **`AsyncOpenAI` token refresh:** do not mutate `client.api_key` between calls — `AsyncOpenAI` does not support that pattern safely under concurrency. Instead, recreate the `AsyncOpenAI` client with a fresh token after a successful `_reauthenticate()` call. This is safe because re-auth only happens while the auth lock is held, so no concurrent call can observe a partially updated client.

---

### Step 5 — Make `MCPAdapter` a persistent async context manager

**File:** `agent_poc/tools/mcp_adapter.py`

This is the most significant change. The adapter must hold the subprocess and session open for its lifetime, expose itself as an async context manager, and handle four known failure modes explicitly.

#### 5a — Async context manager interface

Make `MCPAdapter` an async context manager so resource lifetime is explicit and guaranteed:

```python
async def __aenter__(self):
    await self.connect()
    return self

async def __aexit__(self, *exc_info):
    await self.disconnect()
```

Entry points use it as:
```python
async with MCPAdapter(name, command, args, env) as adapter:
    registry.register_tools(adapter.list_tools())
    state = await runner.run(prompt)
# subprocess cleaned up here, guaranteed
```

#### 5b — Persistent session via `AsyncExitStack`

Use `contextlib.AsyncExitStack` to hold the `stdio_client` and `ClientSession` context managers open without manual `__aenter__`/`__aexit__` calls:

```python
async def connect(self):
    self._stack = contextlib.AsyncExitStack()
    read, write = await self._stack.enter_async_context(stdio_client(params))
    self._session = await self._stack.enter_async_context(ClientSession(read, write))
    await self._session.initialize()
    # fetch and register tools — also serves as a health check
    result = await self._session.list_tools()
    if not result.tools:
        raise RuntimeError(f"MCP server '{self._name}' connected but returned no tools — check server config")
    self._tools = [...]  # register as async callables

async def disconnect(self):
    await self._stack.aclose()
    self._session = None
    self._stack = None
```

`AsyncExitStack.aclose()` handles cleanup order and exception safety automatically — no manual `__aenter__`/`__aexit__` calls anywhere.

#### 5c — Async tool callables

Tool lambdas registered during `connect()` become `async def`:
```python
async def _call(args, name=t.name):
    return await self.call_tool(name, args)
```

Remove both `asyncio.run()` calls entirely.

#### 5d — Reconnect on subprocess death (Downfall 1)

Wrap `call_tool` with a reconnect-on-failure guard:

```python
async def call_tool(self, name: str, arguments: dict) -> str:
    try:
        result = await self._session.call_tool(name, arguments)
        self._call_count += 1
        return str(result.content)
    except (BrokenPipeError, ConnectionError, EOFError) as exc:
        logger.warning("MCP session error (%s), attempting reconnect", exc)
        await self._stack.aclose()
        self._call_count = 0
        await self.connect()
        result = await self._session.call_tool(name, arguments)
        self._call_count += 1
        return str(result.content)
```

Catch only transport-level errors (`BrokenPipeError`, `ConnectionError`, `EOFError`) — these are the signals that the subprocess died or the pipe was broken. Do **not** catch bare `Exception`: errors like `McpError`, `ValueError`, or tool-logic failures should propagate immediately as tool errors without triggering a reconnect, since a reconnect cannot fix them.

One reconnect attempt. If the retry also fails, the exception propagates as a tool error — the agent sees it and can decide how to proceed.

#### 5e — Concurrent call safety via lock (Downfall 2)

Add an `asyncio.Lock()` to serialize tool calls through the shared session:

```python
def __init__(self, ...):
    ...
    self._lock = asyncio.Lock()
    self._call_count = 0

async def call_tool(self, name, arguments):
    async with self._lock:
        ...
```

Zero cost when the agent loop is sequential (never contended). Safe by construction if parallel tool execution is added later.

#### 5f — Startup failure behavior (Downfall 3)

`connect()` should propagate exceptions — do not swallow them. A failed MCP server startup should prevent the agent from starting (fail fast), not silently produce an agent missing tools.

In `build_registry()` / `instrumentation.py`, callers that currently catch and warn on `connect()` failure should re-raise. If graceful degradation is explicitly desired for a specific server, that decision belongs at the call site with a clear comment, not as a default.

**Preserve the `skip_servers` distinction.** `skip_servers` means "do not attempt to connect at all" — it is not a failure, it is intentional omission (e.g. skipping the `neo4j` MCP server when graph mode is `cypher_tool` or `none`). The fail-fast rule applies only to servers that are *attempted*. The `if srv.name in skip_servers: continue` guard runs before `connect()` is called and must be kept as-is.

#### 5g — Proactive reconnect for long sessions (Downfall 4)

Add an optional `max_calls_before_reconnect: int | None` config parameter (default `None` = disabled). After that many successful calls, proactively reconnect to clear accumulated MCP server state:

```python
if self._max_calls and self._call_count >= self._max_calls:
    logger.info("MCP '%s': proactive reconnect after %d calls", self._name, self._call_count)
    await self._stack.aclose()
    await self.connect()
```

Wire this into `MCPServerConfig` in `agent_poc/config/loader.py` as an optional field. Default to `100` — conservative enough to prevent Neo4j MCP state accumulation, but high enough to avoid unnecessary reconnects in short benchmark runs. Raise via config for cheaper stateless MCP servers.

---

### Step 6 — Make tool callables async

**Files:** `agent_poc/tools/static/filesystem.py`, `shell.py`, `python_exec.py`, `agent_poc/tools/cypher_tool.py`, `agent_poc/tools/generated.py`

All static tool callables are currently sync functions. They do blocking I/O (file reads, `subprocess.run`, Neo4j). Wrap each with `asyncio.to_thread()` at the registry level (Step 7) rather than changing each tool's internal implementation — this keeps the tool code readable.

**Exception: `cypher_tool.py`** — this one is complex enough that it should become a proper `async def` internally:
- `make_cypher_tool()` → `async def make_cypher_tool()`. It must `await make_backend(tool_config, ...)` **once at construction time** and close over the resulting backend. The backend must **not** be constructed inside `_query_graph` — doing so would trigger the full auth flow (including the TAIS device-flow handshake for Tricentis) on every tool call.
- **Schema cache:** move `_SCHEMA_CACHE` from a module-level dict to an instance variable (a dict closed over inside `make_cypher_tool()`). The module-level dict is a hidden global that breaks in multi-process benchmark runs (stale cross-process state). An instance variable is garbage-collected with the tool and avoids the problem entirely.
- `GraphDatabase.driver()` creation: wrap `session.run()` calls with `asyncio.to_thread()` so Neo4j I/O does not block the event loop.
- The internal `backend.complete()` calls become `await backend.complete(...)`.
- **Correction loops stay as-is:** the tool's two "retry" branches are not retries of the same operation — they are prompt-correction loops that send different content each time (corrected label list; execution error for LLM rewrite). Do not replace them with `retry_async`; the sequential correction logic is correct as written.
- `_get_cached_schema()` → `async def _get_cached_schema()`.
- `_query_graph` → `async def _query_graph(args)`, closing over the pre-built backend.
- Callers of `make_cypher_tool()` in `scripts/benchmark.py` must `await` it.

For all other static tools, keep the callable sync — the registry will wrap them.

#### 6a — Add a shared retry utility

**File:** `agent_poc/tools/retry.py` (new file)

Add a small `retry_async` helper:

```python
async def retry_async(fn, attempts: int = 2, backoff: float = 1.0):
    for i in range(attempts):
        try:
            return await fn()
        except Exception:
            if i == attempts - 1:
                raise
            await asyncio.sleep(backoff * (2 ** i))
```

Use it in `TricentisBackend._complete_openai()` for transient HTTP errors (network failures, 5xx responses). Do **not** apply it to `cypher_tool.py`'s correction branches — those are prompt-engineering loops that send different content on each attempt, not retries of the same operation.

---

### Step 7 — Make `ToolRegistry` async

**File:** `agent_poc/tools/registry.py`

- `execute()` → `async def execute()`.
- Remove `ThreadPoolExecutor`. Replace with `asyncio.wait_for()` for the timeout.
- Distinguish async vs sync callables:
  ```python
  import asyncio, inspect

  async def execute(self, call: ToolCall, timeout_override=None) -> ToolResult:
      tool = self._tools.get(call.name)
      timeout = timeout_override or tool.timeout_seconds
      try:
          if inspect.iscoroutinefunction(tool.callable):
              coro = tool.callable(call.arguments)
          else:
              coro = asyncio.to_thread(tool.callable, call.arguments)
          output = await asyncio.wait_for(coro, timeout=timeout)
      except asyncio.TimeoutError:
          ...
      except Exception as exc:
          ...
  ```

This is the single place that bridges sync callables into the async world via `asyncio.to_thread()`. No changes needed to individual sync tool callables.

**Timeout caveat:** when `asyncio.wait_for` cancels a `to_thread` task, the underlying OS thread is not killed — it continues running until the blocking call returns naturally. For `python_exec` and `shell`, the subprocess keeps running after the timeout exception is raised. This is the same limitation as the original `ThreadPoolExecutor` implementation. True subprocess kill-on-timeout would require explicit `os.kill()` logic inside the tool callable; that is out of scope for this refactor but worth noting for future work.

---

### Step 8 — Make `AgentRunner` async

**File:** `agent_poc/agent/runner.py`

- `run()` → `async def run()`.
- `await self._backend.complete(...)`.
- **Concurrent tool dispatch:** when a response contains multiple tool calls, execute them concurrently with `asyncio.gather` instead of sequentially:

```python
results = await asyncio.gather(
    *[self._registry.execute(tc) for tc in response.tool_calls]
)
```

  This is a free win once everything is async — tool calls that hit different backends (e.g. Neo4j + filesystem) run in parallel. The repeated-call detection logic must be updated for batched dispatch: collect all `call_key` tuples for the current batch **before** dispatch, then after `asyncio.gather` completes, check whether the entire batch is identical to the previous batch (same ordered list of call keys), and append all keys to `state.recent_calls` as a unit. A "repeated call" in the concurrent model means the whole batch is identical to the previous batch — a stricter definition that avoids false positives when multiple different tools fire in the same round.

- Everything else (message building, loop logic, repeated-call detection) stays identical.

---

### Step 9 — Make instrumentation async

**File:** `agent_poc/agent/instrumentation.py`

- `TrackingBackend.complete()` → `async def complete()`. `await self._backend.complete(...)`.
- `TimingRegistry.execute()` → `async def execute()`. `await super().execute(...)` with `time.perf_counter()` wrapping.
- `build_registry()` → `async def build_registry()`. `await adapter.connect()` for each MCP server.
- **Adapter lifetime:** `build_registry()` must own the adapters it connects and ensure they are disconnected at shutdown. Make `ToolRegistry` (and by inheritance `TimingRegistry`) an async context manager. Store each connected `MCPAdapter` in a list on the registry; `__aexit__` calls `await adapter.disconnect()` for each. Entry points use the registry as:

  ```python
  registry = await build_registry(config)
  async with registry:
      state = await runner.run(prompt)
  # all MCP subprocesses cleaned up here
  ```

  This keeps adapter lifetime exactly as long as registry lifetime — no leaks, no manual cleanup at call sites. The `ToolRegistry` changes belong in `registry.py`, not `instrumentation.py`.

---

### Step 10 — Update `make_backend` factory

**File:** `agent_poc/models/factory.py`

- `make_backend()` → `async def make_backend()`.
- For `tricentis` provider: `return await TricentisBackend.create(deployment, temperature)`.
- For `local` and `bedrock`: construction is still sync, but return the instance. Can remain sync actually — only `TricentisBackend.create()` needs awaiting. Consider splitting into `make_backend_sync()` for local/bedrock and `await make_backend()` for tricentis, or just always make it async for uniformity.

Recommended: make `make_backend()` async throughout for a uniform API.

---

### Step 11 — Update CLI entry points

**Files:** `main.py`, `scripts/query.py`, `scripts/benchmark.py`, `scripts/ingest.py`

**Note:** The Streamlit UI (`ui/`) is being removed entirely. Do not port it.

Each CLI entry point becomes:
```python
import asyncio

async def main():
    registry = await build_registry(config)
    async with registry:
        backend = await make_backend(config, provider=...)
        runner = AgentRunner(backend, registry, config, system_prompt=...)
        state = await runner.run(prompt)
        ...

if __name__ == "__main__":
    asyncio.run(main())
```

One `asyncio.run()` per process, at the top level. Nowhere else. The `async with registry:` block guarantees MCP subprocess cleanup even if an exception is raised mid-run.

For `scripts/ingest.py`: it uses `backend.complete()` directly (no runner). Same pattern — `async def main()`, `await backend.complete(...)`, `asyncio.run(main())`.

---

### Step 12 — Update tests

**Files:** `agent_poc/tests/conftest.py`, `test_runner.py`, `test_registry.py`, `test_mcp_adapter.py`, `test_static_tools.py`, `agent_poc/tests/integration/test_live_agent.py`

- Add `pytest-asyncio` to dev dependencies in `pyproject.toml`.
- Add `asyncio_mode = "auto"` to `[tool.pytest.ini_options]` in `pyproject.toml` — eliminates per-test `@pytest.mark.asyncio` decoration.
- `MockBackend.complete()` in `conftest.py` → `async def complete()`.
- All `test_runner.py` test functions → `async def test_*()`.
- `test_registry.py`: replace `time.sleep(60)` timeout test with `asyncio.sleep(60)` inside an async tool callable. `execute()` → `await registry.execute(...)`.
- `test_mcp_adapter.py`: remove the `asyncio.run` patch approach. Instead mock `stdio_client` and `ClientSession` as async context managers using `AsyncMock` / `asynctest` patterns. Test `connect()` and `call_tool()` via `await`.
- `test_static_tools.py`: tool callables remain sync; tests can stay sync or go async — no change required since callables themselves don't change.
- Integration test: `async def test_*`, `await runner.run(...)`.

---

## Files Changed Summary

| File | Change |
|---|---|
| `agent_poc/agent/types.py` | `ModelBackend.complete()` → async |
| `agent_poc/agent/runner.py` | `run()` → async; concurrent tool dispatch via `asyncio.gather`; batched repeated-call detection |
| `agent_poc/agent/instrumentation.py` | `complete()`, `execute()`, `build_registry()` → async; registry owns adapter lifetime |
| `agent_poc/models/factory.py` | `make_backend()` → async |
| `agent_poc/models/openai_compatible.py` | `AsyncOpenAI`, `complete()` → async |
| `agent_poc/models/bedrock_backend.py` | `complete()` → async via `to_thread` |
| `agent_poc/models/tricentis_backend.py` | factory classmethod, all methods → async; auth lock; recreate `AsyncOpenAI` client after re-auth |
| `agent_poc/tools/registry.py` | `execute()` → async, `asyncio.wait_for` + `to_thread`; async context manager for adapter cleanup; timeout-does-not-kill-thread caveat documented |
| `agent_poc/tools/mcp_adapter.py` | async context manager, `AsyncExitStack`, persistent session, lock, reconnect, health check, proactive cycle (default 100 calls) |
| `agent_poc/tools/cypher_tool.py` | `make_cypher_tool()` → async, backend built once at construction, instance-level schema cache, correction loops unchanged, `_query_graph` → async, internal `to_thread` for Neo4j |
| `agent_poc/tools/retry.py` | **new** — `retry_async(fn, attempts, backoff)` utility |
| `agent_poc/config/loader.py` | add `max_calls_before_reconnect: int \| None = 100` to `MCPServerConfig`; fix duplicate `integration` marker in `pyproject.toml` |
| `agent_poc/tools/static/filesystem.py` | no change (registry wraps via `to_thread`) |
| `agent_poc/tools/static/shell.py` | no change |
| `agent_poc/tools/static/python_exec.py` | no change |
| `ui/` | **deleted** |
| `main.py` | `async def main()`, `asyncio.run(main())` |
| `scripts/query.py` | same pattern |
| `scripts/benchmark.py` | same pattern |
| `scripts/ingest.py` | same pattern |
| `pyproject.toml` | add `pytest-asyncio`, `asyncio_mode = "auto"`; fix duplicate `integration` marker |
| `agent_poc/tests/conftest.py` | `MockBackend.complete()` → async |
| `agent_poc/tests/test_runner.py` | all tests → async |
| `agent_poc/tests/test_registry.py` | all tests → async |
| `agent_poc/tests/test_mcp_adapter.py` | rewrite with AsyncMock |
| `agent_poc/tests/integration/test_live_agent.py` | `await runner.run()` |

---

## Verification

1. **Unit tests:** `pytest agent_poc/tests/` — all pass without live services.
2. **MCP persistent session:** run `main.py` with Neo4j MCP configured. Confirm in logs that the subprocess is spawned once at startup, not per tool call. Time a multi-tool run before and after.
3. **MCP health check:** start with a misconfigured MCP server (e.g. wrong command). Confirm the process exits immediately with a clear error rather than failing silently at first tool call.
4. **Concurrent tool dispatch:** send a prompt that elicits two tool calls in one response. Confirm via logs that both start before either completes (i.e. they overlap in time).
5. **Tricentis auth:** run with `--provider tricentis`. Auth should happen once at startup via `create()`. Multiple LLM calls in one run should reuse the same client. Simulate a token expiry mid-run and confirm only one reauthentication fires (lock prevents concurrent re-auth) and that the `AsyncOpenAI` client is recreated with a fresh token afterward.
6. **Benchmark:** run `scripts/benchmark.py` with a query set. Confirm it completes and produces the same CSV output as before.
7. **Ingest pipeline:** `scripts/ingest.py` on a sample CSV — confirm Neo4j write succeeds.
8. **Error path:** kill the Neo4j MCP server mid-run and confirm the adapter reconnects once and returns an error gracefully rather than hanging.
9. **Schema cache isolation:** run two benchmark queries back-to-back and confirm schema is fetched once (cached), not twice.
