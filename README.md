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

`INTENT_BRIDGE_LLM_AMBIGUOUS_TARGET_FALLBACK_ENABLED` defaults to `true`.
When enabled, a deterministic target ambiguity is offered to the LLM route
before asking the user to clarify. If the LLM route is disabled or fails, the
original deterministic clarification is returned. Set it to `false` to ask for
clarification immediately without attempting the LLM.

`INTENT_BRIDGE_MA_PREFER_NATIVE_PLAYBACK` defaults to `true`. When the native
Music Assistant integration is configured, deterministic media search requests
such as "play Lana Del Rey" execute directly through the native `ma_play_query`
fast-start operation without an LLM turn or Home Assistant's media-search intent.
Set it to `false` to retain Home Assistant intent handling for media searches.
Playback controls such as pause and resume are unaffected.

`INTENT_BRIDGE_VOICE_FAILURE_RESPONSE` configures the final non-action response
used when every safe planning route declines or is unavailable. The default is
`Sorry, I couldn't handle that request.` This keeps ordinary no-match failures
inside the OpenAI-compatible response contract for HA Assist; failures after an
action may have started still return an HTTP error rather than claiming safety.

When `INTENT_BRIDGE_HA_WS_ENABLED=true`, ordinary single-step deterministic
voice commands use Home Assistant's persistent WebSocket
`conversation/process` command first. The direct `/api/intent/handle` HTTP
request remains the fallback when the socket is unavailable or rejects the
conversation request. Compound commands and deliberately materialised exact
entity targets stay on the named-intent route so an utterance is never replayed
or widened accidentally.

Assistant feedback is configured through the transport-neutral
`INTENT_BRIDGE_ASSISTANT_*` namespace. LED feedback uses
`ASSISTANT_LED_ENABLED`, `ASSISTANT_LED_DOMAINS`, `ASSISTANT_LED_COLOR`,
`ASSISTANT_LED_EFFECT`, `ASSISTANT_LED_SOFTWARE_PULSE_ENABLED`, and
`ASSISTANT_LED_PULSE_INTERVAL_SECONDS`. Any adapter can use the shared async
feedback lifecycle with LED feedback, sound feedback, or both; Music Assistant
currently uses LED feedback while waiting for asynchronous playback.

Optional assistant sounds replace spoken response text with short status tones
on a `media_player` associated with the calling Assist satellite. Set
`INTENT_BRIDGE_ASSISTANT_SOUNDS_ENABLED=true` and configure
`INTENT_BRIDGE_BASE_URL` to the externally reachable HTTP(S) root of this
server (for example, `http://intent-bridge.local:8000`). The bridge serves and
sends fully qualified URLs for `processing.mp3`, `success.mp3`, and `error.mp3`
from `/assistant/sounds/`. Sounds default to `false`. When enabled, only the
configured `INTENT_BRIDGE_VOICE_ACTION_CONFIRMATION` is replaced by the success
sound and empty assistant text. Informational answers, clarification questions,
and other custom responses remain text for Home Assistant to speak.

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

### Custom MCP tools

Copy `mcp.json.example` to `mcp.json` to expose additional MCP tools directly
to the LLM fallback. The file is optional and `mcp.json` is gitignored because
server headers or environment values may contain credentials. Its path can be
changed with `INTENT_BRIDGE_MCP_CONFIG_PATH`.

Each entry under `mcpServers` accepts `name`, `description`, and `isActive`.
Supported transports are `streamableHttp` and `sse` with `baseUrl` and optional
`headers`, plus `stdio` with `command`, optional `args`, `env`, and `cwd`.
Restart Intent Bridge after changing the file. Active servers are connected at
startup; failed custom servers are logged and omitted without disabling the
fallback agent.

After the deterministic Home Assistant route declines a request, a conservative
informational gate sends clear general questions and conversation to a separate
LLM agent. That agent has custom MCP tools but no Home Assistant or Music
Assistant tools, and must search before answering potentially current facts.
Requests containing household, device-control, or media intent continue to the
existing HA/MA fallback agent.

Set `INTENT_BRIDGE_LOCALE` and `INTENT_BRIDGE_LOCATION` to ground that agent in
local language conventions and geography. `INTENT_BRIDGE_TIMEZONE` supplies a
fresh local date and time on every request. Voice-origin areas are identified as
household rooms and are never treated as geographic locations.

When those values are not explicitly set, Intent Bridge bootstraps timezone,
language/locale, country, units, and currency from Home Assistant's authenticated
WebSocket `get_config` response. Explicit environment values take precedence.
Home Assistant latitude and longitude are never included in LLM context.

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
dynamically. Run all 17,815 sentence/dialogue examples with:

```powershell
uv run pytest benchmark/test_benchmark.py -o addopts="" -n auto -q
```

The runner always exercises the production voice pipeline. When the LLM is
enabled and configured, the normal fallback route is available after the
deterministic route exactly as it is in the application.
Runs are exhaustive unless `BENCHMARK_LIMIT` is explicitly set. Independent
examples run in parallel using one worker per CPU core by default; use
`-n 1` for sequential execution. See
[`benchmark/README.md`](benchmark/README.md) for filtering, limiting, and
LLM configuration. Benchmark scratch files stay under `.cache` so stale
Windows `%TEMP%` permissions do not prevent collection.

To run only one benchmark home:

```powershell
$env:BENCHMARK_HOME='studio'; uv run pytest benchmark/test_benchmark.py -o addopts="" -n auto -q
```

Every example must produce the exact expected entity operations; the benchmark
contains no skip, xfail, sampling, or allow-list path. See
[`benchmark/README.md`](benchmark/README.md) for the dataset contract, loader
checks, and collection commands.
