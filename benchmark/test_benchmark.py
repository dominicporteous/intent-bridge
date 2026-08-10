"""The exhaustive deterministic matcher benchmark (14,243 gold examples)."""

from __future__ import annotations

import os

import pytest

from benchmark.loader import load_corpus
from benchmark.production import ProductionBenchmarkMatcher
from benchmark.runner import compare_operations
from benchmark.validation import assert_valid_corpus, inventory

CORPUS = load_corpus()
BENCHMARK_HOME = tuple(
    value.strip()
    for value in os.environ.get("BENCHMARK_HOME", "").split(",")
    if value.strip()
)
BENCHMARK_SOURCE = tuple(
    value.strip()
    for value in os.environ.get("BENCHMARK_SOURCE", "").split(",")
    if value.strip()
)


def _filter_examples(examples):
    if not BENCHMARK_HOME and not BENCHMARK_SOURCE:
        return examples
    filtered = []
    for example in examples:
        if BENCHMARK_HOME and example.request.home.home_id not in BENCHMARK_HOME:
            continue
        if BENCHMARK_SOURCE and not any(source in example.diagnostic_id for source in BENCHMARK_SOURCE):
            continue
        filtered.append(example)
    return tuple(filtered)


EXAMPLES = _filter_examples(CORPUS.examples)


@pytest.fixture(scope="session")
def matcher() -> ProductionBenchmarkMatcher:
    return ProductionBenchmarkMatcher()


def test_benchmark_corpus_is_complete_and_unambiguous():
    corpus_inventory = inventory(CORPUS)
    assert corpus_inventory.scenario_files == 730
    assert corpus_inventory.scenarios == 1_971
    assert corpus_inventory.examples == 14_219
    assert corpus_inventory.turns == 15_883
    assert_valid_corpus(CORPUS)


def test_filtered_benchmark_selection_matches_examples():
    if not BENCHMARK_HOME and not BENCHMARK_SOURCE:
        pytest.skip("benchmark selection not enabled")
    assert EXAMPLES, (
        "BENCHMARK_HOME or BENCHMARK_SOURCE matched no benchmark examples. "
        f"BENCHMARK_HOME={BENCHMARK_HOME!r} BENCHMARK_SOURCE={BENCHMARK_SOURCE!r}"
    )


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda example: example.diagnostic_id)
async def test_intent_benchmark(example, matcher: ProductionBenchmarkMatcher):
    result = await matcher.match(example.request)
    missing, unexpected = compare_operations(example.expected, result.operations)

    assert not missing and not unexpected, (
        f"{example.diagnostic_id}\n"
        f"turns={example.request.turns!r}\n"
        f"missing={missing!r}\n"
        f"unexpected={unexpected!r}\n"
        f"response={result.response!r}"
    )
