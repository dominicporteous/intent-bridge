"""Inventory and gold-data consistency checks for the benchmark corpus."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from benchmark.models import BenchmarkCorpus, BenchmarkExample, Operation


@dataclass(frozen=True, slots=True)
class CorpusInventory:
    homes: int
    scenario_files: int
    scenarios: int
    examples: int
    single_turn_examples: int
    dialogue_examples: int
    turns: int


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    message: str
    diagnostic_ids: tuple[str, ...] = ()


class CorpusValidationError(AssertionError):
    pass


def inventory(corpus: BenchmarkCorpus) -> CorpusInventory:
    examples = corpus.examples
    return CorpusInventory(
        homes=len(corpus.homes),
        scenario_files=len({scenario.source for scenario in corpus.scenarios}),
        scenarios=len(corpus.scenarios),
        examples=len(examples),
        single_turn_examples=sum(len(example.request.turns) == 1 for example in examples),
        dialogue_examples=sum(len(example.request.turns) > 1 for example in examples),
        turns=sum(len(example.request.turns) for example in examples),
    )


def _normal_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _operation_keys(operations: tuple[Operation, ...]) -> frozenset[tuple[Any, ...]]:
    return frozenset(key for operation in operations for key in operation.atomic_semantic_keys())


def input_key(example: BenchmarkExample) -> tuple[Any, ...]:
    """Fingerprint only information legitimately supplied to the matcher."""

    return (
        example.request.home.home_id,
        tuple(_normal_text(turn) for turn in example.request.turns),
        tuple(
            sorted(
                (operation.semantic_key() for operation in example.request.setup),
                key=repr,
            )
        ),
        tuple(
            sorted((str(key), repr(value)) for key, value in example.request.origin_context.items())
        ),
    )


def validate_corpus(corpus: BenchmarkCorpus) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    example_ids = [example.diagnostic_id for example in corpus.examples]
    if len(example_ids) != len(set(example_ids)):
        issues.append(
            ValidationIssue("duplicate-example-id", "example diagnostic IDs are not unique")
        )

    scenario_ids = [scenario.diagnostic_id for scenario in corpus.scenarios]
    if len(scenario_ids) != len(set(scenario_ids)):
        issues.append(
            ValidationIssue("duplicate-scenario-id", "scenario diagnostic IDs are not unique")
        )

    for scenario in corpus.scenarios:
        if not scenario.examples:
            issues.append(
                ValidationIssue(
                    "empty-scenario",
                    "scenario has no executable examples",
                    (scenario.diagnostic_id,),
                )
            )
        scenario_keys = _operation_keys(scenario.expected)
        covered = frozenset(
            key for example in scenario.examples for key in _operation_keys(example.expected)
        )
        missing = scenario_keys - covered
        if missing:
            issues.append(
                ValidationIssue(
                    "uncovered-case-operation",
                    f"no wording exercises {len(missing)} case-level operation(s)",
                    (scenario.diagnostic_id,),
                )
            )
        for example in scenario.examples:
            if not example.expected:
                issues.append(
                    ValidationIssue(
                        "empty-expected",
                        "example has no expected operation",
                        (example.diagnostic_id,),
                    )
                )

    grouped: dict[tuple[Any, ...], list[BenchmarkExample]] = defaultdict(list)
    for example in corpus.examples:
        grouped[input_key(example)].append(example)
    for examples in grouped.values():
        expected_variants = {_operation_keys(example.expected) for example in examples}
        if len(expected_variants) <= 1:
            continue
        common = set.intersection(*(set(variant) for variant in expected_variants))
        if not common:
            issues.append(
                ValidationIssue(
                    "incompatible-gold",
                    "identical matcher input has mutually incompatible expected operations",
                    tuple(example.diagnostic_id for example in examples),
                )
            )
        else:
            issues.append(
                ValidationIssue(
                    "unprojected-gold",
                    "identical matcher input still has subset/superset expected operations",
                    tuple(example.diagnostic_id for example in examples),
                )
            )
    return tuple(issues)


def assert_valid_corpus(corpus: BenchmarkCorpus) -> None:
    issues = validate_corpus(corpus)
    if not issues:
        return
    details = "\n".join(
        f"- [{issue.code}] {issue.message}: {', '.join(issue.diagnostic_ids[:5])}"
        for issue in issues[:25]
    )
    remainder = len(issues) - min(25, len(issues))
    if remainder:
        details += f"\n- ... and {remainder} more issue(s)"
    raise CorpusValidationError(f"benchmark corpus is inconsistent:\n{details}")


__all__ = [
    "CorpusInventory",
    "CorpusValidationError",
    "ValidationIssue",
    "assert_valid_corpus",
    "input_key",
    "inventory",
    "validate_corpus",
]
