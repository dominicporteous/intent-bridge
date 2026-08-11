# Intent benchmark

This folder contains the exhaustive, data-driven acceptance benchmark for the
local voice-to-action matcher. It is intentionally separate from the fast unit
suite in `tests/`.

The checked-in corpus currently contains:

- 5 independent Home Assistant home topologies
- 908 scenario YAML files
- 2,470 scenarios
- 17,815 independently executed sentence or dialogue examples
- 19,943 total conversation turns

The `2_floor_2_bed` corpus models a two-storey British home with two bedrooms,
a study, utility cupboard, garden and shed. In addition to the standard
device, area, query, clarification, persistence, automation, list and
multi-intent coverage, it includes an end-to-end UK household routine suite
covering central heating, commuting, damp-weather ventilation, off-peak
washing, cooking, bins, shopping, evening security and bedtime.

Every example must pass. The benchmark has no skip, xfail, sampling, or
allow-list mechanism, so a successful run means 100% corpus coverage.

## Run it

Install the development environment, then run:

```shell
uv sync --all-groups
uv run pytest benchmark/test_benchmark.py -o addopts="" -q
```

Run the loader and corpus-integrity contract tests separately with:

```shell
uv run pytest benchmark/test_loader.py -o addopts="" -q
```

To filter the benchmark to a single home or scenario path, set one or both of:

- `BENCHMARK_HOME` — comma-separated benchmark home IDs (dataset directory names)
- `BENCHMARK_SOURCE` — substring(s) of the scenario source path

Example:

```shell
BENCHMARK_HOME=2_floor_2_bed uv run pytest benchmark/test_benchmark.py -o addopts="" -q
```

Or, to select a specific scenario file:

```shell
BENCHMARK_SOURCE=switches.yaml uv run pytest benchmark/test_benchmark.py -o addopts="" -q
```

To see every dynamically collected example without executing the matcher:

```shell
uv run pytest benchmark/test_benchmark.py -o addopts="" --collect-only -q
```

The `-o addopts=""` override keeps the application unit-suite coverage options
out of this acceptance run. It does not skip benchmark examples.

## Dataset layout

Each home is isolated below `benchmark/datasets/<home_id>/`:

```text
benchmark/datasets/
  studio/
    home_config.yaml
    devices/
    area/
    clarifications/
    state_persistance/
    multiple_intents/
    lists/
    timers/
    automations/
```

`home_config.yaml` defines floors, areas, devices, scenes, scripts, timers, and
lists. Every other YAML file is discovered recursively at collection time, so
new files and new sentences automatically become benchmark cases.

A normal scenario contains a name, expected final conditions, and alternative
wordings:

```yaml
- name: kitchen_light_on
  conditions:
    - type: action
      entity_id: light.kitchen_ceiling
      state: "on"
  sentences:
    - Turn on the kitchen ceiling light
    - Please switch the kitchen ceiling light on
```

A sentence may instead be a list of turns. Each such list is one stateful
dialogue and is executed in order with its own isolated setup:

```yaml
sentences:
  - - What is the brightness of the kitchen lights?
    - Set it to 62.
```

## Gold-data rules

- Each sentence or dialogue is an independent example; state never leaks to
  another example.
- `setup` is input state, not an expected action.
- Area/domain conditions expand to the concrete entities in that home's
  catalog.
- Combined-device source cases may contain alternative sentences naming only
  one device. The loader projects the case-level conditions onto the target
  explicitly named by that example, and requires the union of all examples to
  cover every case-level condition.
- Clarification cases treat the first condition as the selected action. Later
  conditions are invariants documenting similarly named entities that must not
  be changed.
- Identical conversation, home, setup, and origin context must have one
  compatible answer key. Corpus collection fails before matching if it does
  not.
- Corrections to imported fixture data retain an inline `# Changed from: ...`
  comment so the original wording remains reviewable beside its replacement.

The production adapter receives only conversation turns, the home/catalog,
setup state, and origin context. Fixture paths, scenario names, and expected
operations are deliberately excluded from matcher input, preventing an
implementation from passing by looking up fixture answers.

## Production pipeline benchmark

The pytest runner always uses the production voice pipeline constructed by
`intent_bridge.application.build_voice_pipeline`. LLM configuration never
selects a different benchmark matcher or planner stack. The configured LLM/tool
fallback becomes available when these application options are all set:

- `INTENT_BRIDGE_LLM_ENABLED=true`
- `INTENT_BRIDGE_LLM_BASE_URL`
- `INTENT_BRIDGE_LLM_MODEL`

The runner loads the repository `.env` before importing application settings.
Every turn is passed directly to that pipeline. This preserves production route
order: deterministic planning runs first and `llm-ha-ws` runs only when the
deterministic route declines. Each corpus home is presented to the normal
LLM HA tools as an isolated in-memory HA cache; service calls are recorded and
compared with the fixture's expected operations. It never controls the HA
instance configured in `.env`.

Start with a small filtered sample because LLM cases make real model requests:

```powershell
$env:BENCHMARK_HOME = "studio"
$env:BENCHMARK_LIMIT = "25"
uv run pytest benchmark/test_benchmark.py -o addopts="" -q
```

Supported controls are `BENCHMARK_HOME`, `BENCHMARK_SOURCE`,
and `BENCHMARK_LIMIT`. Runs remain exhaustive by default, including when
LLM-backed; set `BENCHMARK_LIMIT` explicitly to control model usage and cost.
Execution is sequential because the agent-facing HA transport is process-global.

The normal fallback agent and its `ha_search`, `ha_get_state`,
`ha_list_services`, and `ha_call_service` tools are exercised. The advanced
administrative `ha-mcp` subprocess and Music Assistant are deliberately not
attached: they cannot safely operate against the synthetic fixture home and
would otherwise reach external systems. Those integrations retain their own
integration tests and should be evaluated separately against a disposable HA
test instance.
