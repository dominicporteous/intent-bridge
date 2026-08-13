"""Reusable comparison and corpus execution helpers."""

from __future__ import annotations

import asyncio
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from benchmark.models import BenchmarkCorpus, BenchmarkExample, BenchmarkMatcher, Operation


def _default_worker_count() -> int:
    return os.cpu_count() or 1


@dataclass(frozen=True, slots=True)
class BenchmarkOptions:
    """Environment-controlled selection and matcher options for pytest runs."""

    homes: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    limit: int | None = None
    workers: int = field(default_factory=_default_worker_count)

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
    ) -> BenchmarkOptions:
        environ = os.environ if environ is None else environ
        homes = _csv(environ.get("BENCHMARK_HOME", ""))
        sources = _csv(environ.get("BENCHMARK_SOURCE", ""))

        raw_limit = environ.get("BENCHMARK_LIMIT")
        limit = max(1, int(raw_limit)) if raw_limit else None
        raw_workers = environ.get("BENCHMARK_WORKERS")
        workers = max(1, int(raw_workers)) if raw_workers else _default_worker_count()
        return cls(homes, sources, limit, workers)


def _csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def select_examples(
    examples: Sequence[BenchmarkExample],
    options: BenchmarkOptions,
) -> tuple[BenchmarkExample, ...]:
    """Apply the common home, source, and safety-limit selection controls."""

    selected = tuple(
        example
        for example in examples
        if (not options.homes or example.request.home.home_id in options.homes)
        and (
            not options.sources
            or any(source in example.diagnostic_id for source in options.sources)
        )
    )
    return selected if options.limit is None else selected[: options.limit]


def make_benchmark_matcher(options: BenchmarkOptions) -> BenchmarkMatcher:
    """Always exercise the application production pipeline."""
    del options
    from benchmark.full_pipeline import FullPipelineBenchmarkMatcher

    return FullPipelineBenchmarkMatcher()


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


async def run_examples(
    matcher: BenchmarkMatcher,
    examples: Sequence[BenchmarkExample],
    *,
    workers: int | None = None,
    concurrency: int | None = None,
) -> BenchmarkSummary:
    """Evaluate examples concurrently, preserving input order in failures.

    ``concurrency`` is retained as a compatibility alias for callers of the
    original runner API. New callers should use ``workers``.
    """

    if workers is not None and concurrency is not None:
        raise ValueError("pass either workers or concurrency, not both")
    worker_count = max(
        1,
        workers
        if workers is not None
        else concurrency
        if concurrency is not None
        else _default_worker_count(),
    )
    selected = tuple(examples)
    semaphore = asyncio.Semaphore(worker_count)

    async def evaluate(example: BenchmarkExample) -> ExampleFailure | None:
        async with semaphore:
            return await evaluate_example(matcher, example)

    results = await asyncio.gather(*(evaluate(example) for example in selected))
    failures = tuple(result for result in results if result is not None)
    return BenchmarkSummary(
        total=len(selected),
        passed=len(selected) - len(failures),
        failures=failures,
    )


async def run_corpus(
    matcher: BenchmarkMatcher,
    corpus: BenchmarkCorpus,
    *,
    workers: int | None = None,
    concurrency: int | None = None,
) -> BenchmarkSummary:
    """Evaluate every example; no sampling, skipping, or xfail path exists."""

    return await run_examples(
        matcher,
        corpus.examples,
        workers=workers,
        concurrency=concurrency,
    )


__all__ = [
    "BenchmarkOptions",
    "BenchmarkSummary",
    "ExampleFailure",
    "compare_operations",
    "evaluate_example",
    "make_benchmark_matcher",
    "run_examples",
    "run_corpus",
    "select_examples",
]
