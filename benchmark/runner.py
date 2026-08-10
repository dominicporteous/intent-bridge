"""Reusable comparison and corpus execution helpers."""

from __future__ import annotations

import asyncio
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from benchmark.models import BenchmarkCorpus, BenchmarkExample, BenchmarkMatcher, Operation

_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True, slots=True)
class BenchmarkOptions:
    """Environment-controlled selection and matcher options for pytest runs."""

    homes: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    limit: int | None = None
    force_llm: bool = False
    use_full_pipeline: bool = False

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        *,
        llm_settings: Any | None = None,
    ) -> BenchmarkOptions:
        environ = os.environ if environ is None else environ
        if llm_settings is None:
            # Imported lazily so an executable can load .env before application
            # settings are constructed.
            from intent_bridge.config import settings

            llm_settings = settings.llm

        homes = _csv(environ.get("BENCHMARK_HOME", ""))
        sources = _csv(environ.get("BENCHMARK_SOURCE", ""))
        force_llm = environ.get("BENCHMARK_FORCE_LLM", "").strip().casefold() in _TRUE_VALUES
        configured = bool(
            llm_settings.enabled and llm_settings.base_url and llm_settings.model
        )
        use_full_pipeline = configured or force_llm

        raw_limit = environ.get("BENCHMARK_LIMIT")
        limit = max(1, int(raw_limit)) if raw_limit else None
        return cls(homes, sources, limit, force_llm, use_full_pipeline)


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
    """Choose the production pipeline adapter when configured, otherwise deterministic."""

    if options.use_full_pipeline:
        from benchmark.full_pipeline import FullPipelineBenchmarkMatcher

        return FullPipelineBenchmarkMatcher(force_llm=options.force_llm)

    from benchmark.production import ProductionBenchmarkMatcher

    return ProductionBenchmarkMatcher()


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
    "BenchmarkOptions",
    "BenchmarkSummary",
    "ExampleFailure",
    "compare_operations",
    "evaluate_example",
    "make_benchmark_matcher",
    "run_corpus",
    "select_examples",
]
