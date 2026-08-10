"""Reusable comparison and corpus execution helpers."""

from __future__ import annotations

import asyncio
from collections import Counter
from dataclasses import dataclass
from typing import Any

from benchmark.models import BenchmarkCorpus, BenchmarkExample, BenchmarkMatcher, Operation


def _keys(operations: tuple[Operation, ...]) -> Counter[tuple[Any, ...]]:
    return Counter(key for operation in operations for key in operation.atomic_semantic_keys())


@dataclass(frozen=True, slots=True)
class ExampleFailure:
    diagnostic_id: str
    missing: tuple[tuple[Any, ...], ...] = ()
    unexpected: tuple[tuple[Any, ...], ...] = ()
    error: str | None = None


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    total: int
    passed: int
    failures: tuple[ExampleFailure, ...]

    @property
    def coverage_percent(self) -> float:
        return 100.0 if not self.total else (self.passed / self.total) * 100.0


def compare_operations(
    expected: tuple[Operation, ...],
    actual: tuple[Operation, ...],
) -> tuple[tuple[tuple[Any, ...], ...], tuple[tuple[Any, ...], ...]]:
    expected_keys = _keys(expected)
    actual_keys = _keys(actual)
    missing = tuple((expected_keys - actual_keys).elements())
    unexpected = tuple((actual_keys - expected_keys).elements())
    return missing, unexpected


async def evaluate_example(
    matcher: BenchmarkMatcher,
    example: BenchmarkExample,
) -> ExampleFailure | None:
    try:
        result = await matcher.match(example.request)
    except Exception as exc:  # benchmark reports failures instead of aborting a full run
        return ExampleFailure(example.diagnostic_id, error=f"{type(exc).__name__}: {exc}")
    missing, unexpected = compare_operations(example.expected, result.operations)
    if missing or unexpected:
        return ExampleFailure(example.diagnostic_id, missing, unexpected)
    return None


async def run_corpus(
    matcher: BenchmarkMatcher,
    corpus: BenchmarkCorpus,
    *,
    concurrency: int = 1,
) -> BenchmarkSummary:
    """Evaluate every example; no sampling, skipping, or xfail path exists."""

    examples = corpus.examples
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def evaluate(example: BenchmarkExample) -> ExampleFailure | None:
        async with semaphore:
            return await evaluate_example(matcher, example)

    results = await asyncio.gather(*(evaluate(example) for example in examples))
    failures = tuple(result for result in results if result is not None)
    return BenchmarkSummary(
        total=len(examples),
        passed=len(examples) - len(failures),
        failures=failures,
    )


__all__ = [
    "BenchmarkSummary",
    "ExampleFailure",
    "compare_operations",
    "evaluate_example",
    "run_corpus",
]
