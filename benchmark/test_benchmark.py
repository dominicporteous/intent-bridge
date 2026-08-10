"""The exhaustive intent benchmark, with optional production-pipeline execution."""

from __future__ import annotations

import pytest

from benchmark.loader import load_corpus
from benchmark.models import BenchmarkMatcher
from benchmark.runner import (
    BenchmarkOptions,
    compare_operations,
    make_benchmark_matcher,
    select_examples,
)
from benchmark.validation import assert_valid_corpus, inventory

CORPUS = load_corpus()
OPTIONS = BenchmarkOptions.from_environment()
EXAMPLES = select_examples(CORPUS.examples, OPTIONS)

@pytest.fixture(scope="session")
def matcher() -> BenchmarkMatcher:
    return make_benchmark_matcher(OPTIONS)

@pytest.mark.parametrize("example", EXAMPLES, ids=lambda example: example.diagnostic_id)
async def test_intent_benchmark(example, matcher: BenchmarkMatcher):
    result = await matcher.match(example.request)
    missing, unexpected = compare_operations(example.expected, result.operations)

    assert not missing and not unexpected, (
        f"{example.diagnostic_id}\n"
        f"turns={example.request.turns!r}\n"
        f"missing={missing!r}\n"
        f"unexpected={unexpected!r}\n"
        f"routes={getattr(matcher, 'last_routes', ())!r}\n"
        f"response={result.response!r}"
    )
