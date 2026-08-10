# Intent Bridge

An OpenAI-compatible voice proxy that recognizes official Open Home Foundation
intents locally with HassIL, executes them through Home Assistant, and falls
back to a local model for requests outside the deterministic grammar.

## Development

Python 3.12 or newer is required. `pyproject.toml` is the authoritative project
and development configuration; `requirements.txt` is retained for existing
deployments.

```powershell
$env:UV_CACHE_DIR = ".uv-cache"
uv sync --dev
uv run pytest -q
uv run python scripts/check_branch_coverage.py --minimum 85
uv run ruff check .
```

Pytest collects line and branch data and enforces the combined coverage gate.
The following command checks the raw branch ratio independently, preventing a
high line score from concealing inadequate branch coverage.

Start the existing deployment entry point with:

```powershell
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

## Architecture

`main.py` is an explicit compatibility entry point exporting only `app` for
existing `uvicorn main:app` deployments.

The core request path is an ordered, extensible voice-to-action pipeline:

```text
OpenAI-compatible request
        |
        v
VoiceActionPipeline
        |
        +-- 1. local OHF/HassIL deterministic route
        |
        +-- 2. LLM fallback route
                    |
                    +-- Home Assistant tools
                    +-- optional advanced HA plugin
                    +-- optional Music Assistant plugin
```

Each route implements the small `VoiceRoute` protocol. The pipeline stops at
the first successful spoken result and records failed attempts for diagnostics.
Agent capabilities are contributed through `AgentToolPlugin`, so a new tool
integration does not require changes to the core pipeline or agent factory.

Key packages:

```text
intent_bridge/
├── intent_engine/            # OHF grammar, HassIL adapter, resolution, orchestration
├── application.py           # composition root, lifecycle, HTTP API
├── bootstrap.py             # executable process setup
├── config/                  # typed models and canonical environment loader
├── core/                    # voice pipeline and pure text/tool policies
├── runtime/                 # process dependencies, request state, stores
├── agents/                  # plugin contracts, composition, result policy
├── api/                     # conversation and voice-origin HTTP support
├── home_assistant/          # HA policy, catalog, WebSocket, tools, MCP
├── music_assistant/         # MA client, search, playback, policy, tools
├── indicators/              # indicator policy, topology, controller
└── transports/              # deterministic/fallback transport
```

Modules inside an integration package may depend on `core` and `runtime`.
`core` must not import concrete integrations. The root `application.py` is the
composition boundary where transports, integrations, and the voice pipeline are
wired together.
The policy modules (`core/voice.py`, `core/text.py`,
`home_assistant/policy.py`, `home_assistant/catalog.py`,
`indicators/policy.py`, `indicators/topology.py`, and `runtime/context.py`) are
side-effect free. Importing the package never connects
to an external service;
lifecycle-owned connections are created by `application.lifespan`. The legacy
ASGI entry point calls `bootstrap.configure_process()` before importing the
application, so dotenv, SDK tracing and logging setup remain executable concerns.

## Configuration

Copy [`.env.example`](.env.example) to `.env` and set the required integration
addresses and credentials. Intent Bridge accepts only canonical variables with
the `INTENT_BRIDGE_` prefix; there are intentionally no legacy aliases while the
project remains alpha. Environment parsing and validation are isolated in
`config/environment.py`, and consumers use typed fields from `BridgeSettings`.

### Custom deterministic wording

The deterministic engine loads the pinned `home-assistant-intents` grammar
first, then recursively loads every `*.yaml` file beneath:

```text
INTENT_BRIDGE_DETERMINISTIC_CUSTOM_SENTENCES_PATH=./custom_sentences/en
```

The default path is `custom_sentences/en`. If it does not exist, startup logs a
warning and continues with the packaged OHF grammar. A configured path that
exists but is not a directory, malformed YAML, or an invalid extension fails
startup rather than partially publishing a grammar.

Custom files use normal HassIL YAML and may only add wording to an existing OHF
intent:

```yaml
language: en

intents:
  HassTurnOff:
    data:
      - sentences:
          - "(kill|shut off) [all] [the] {area} light[s]"
        slots:
          domain: light
```

Files are loaded recursively in deterministic relative-path order. Custom
intent names, global rule/list replacements, response overrides, duplicate
templates, and changes to official intent semantics are rejected. Intent-local
lists and expansion rules are supported.

## Extension points

To add another fallback action route, implement `VoiceRoute` and supply it to
`VoiceActionPipeline`; inject the pipeline with `create_app(pipeline)`. Custom
household wording normally belongs in the custom sentence directory instead of
another route, so every deterministic candidate participates in the same
ambiguity and topology-resolution pass.

Every factory-created app owns the complete API router and resolves requests
through `ApplicationDependencies`. Per-request tool execution data is
context-local, so concurrent voice requests cannot overwrite each other's
origin, selected entity, or terminal action data.

To expose another integration to the LLM route, construct an `AgentToolPlugin`
with its tools and optional instructions, then pass the plugin tuple to
`make_fallback_agent`. Plugins should depend on narrow integration adapters and
must not mutate `main.py` or copy runtime instances into module globals.

Mutable connections and agents belong in the shared `RuntimeState`. The
application lifecycle owns writes to it; adapters consume it. This avoids stale
copies and makes integration replacement explicit in tests.

## TDD workflow

1. Add a characterization test for legacy behavior being changed.
2. Run it against the current implementation.
3. Add or update an architecture test when changing an extension boundary.
4. Make the smallest refactor or feature change.
5. Run the focused test, then the full suite and static checks.

Tests must mock network boundaries. No test should require a live MQTT broker,
Home Assistant instance, Music Assistant server, or language model.

## Acceptance benchmark

The exhaustive corpus now lives in `benchmark/datasets/` and is collected
dynamically. Run all 14,243 sentence/dialogue examples with:

```powershell
uv run pytest benchmark/test_benchmark.py -o addopts="" -q
```

The runner loads `.env` and automatically exercises the production voice/LLM/tool
pipeline when the LLM is enabled and its base URL and model are configured.
Runs are exhaustive unless `BENCHMARK_LIMIT` is explicitly set; see
[`benchmark/README.md`](benchmark/README.md) for filtering, limiting, and
force-LLM controls.

To run only one benchmark home:

```powershell
$env:BENCHMARK_HOME='studio'; uv run pytest benchmark/test_benchmark.py -o addopts="" -q
```

Every example must produce the exact expected entity operations; the benchmark
contains no skip, xfail, sampling, or allow-list path. See
[`benchmark/README.md`](benchmark/README.md) for the dataset contract, loader
checks, and collection commands.
