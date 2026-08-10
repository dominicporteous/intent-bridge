"""Fast contract tests for the corpus loader itself."""

from __future__ import annotations

from dataclasses import fields

from benchmark.loader import load_corpus
from benchmark.models import BenchmarkRequest
from benchmark.validation import assert_valid_corpus, inventory


def test_complete_corpus_inventory_and_consistency():
    corpus = load_corpus()

    assert inventory(corpus) == (
        inventory(corpus).__class__(
            homes=5,
            scenario_files=731,
            scenarios=1_979,
            examples=14_243,
            single_turn_examples=12_573,
            dialogue_examples=1_670,
            turns=15_907,
        )
    )
    assert_valid_corpus(corpus)


def test_matcher_request_cannot_see_fixture_identity_or_answer_key():
    assert {field.name for field in fields(BenchmarkRequest)} == {
        "turns",
        "home",
        "setup",
        "origin_context",
    }


def test_clarification_uses_only_the_selected_action_as_expected():
    corpus = load_corpus()
    scenario = next(
        scenario
        for scenario in corpus.scenarios
        if scenario.source
        == "studio/clarifications/kitchen_light_ceiling_vs_under_cabinet_off.yaml"
    )

    assert len(scenario.expected) == 1
    assert scenario.expected[0].entity_ids == ("light.kitchen_ceiling_lights",)
    assert scenario.expected[0].state == "off"


def test_timer_files_normalize_untyped_conditions_as_timer_operations():
    corpus = load_corpus()
    scenario = next(
        scenario
        for scenario in corpus.scenarios
        if scenario.source == "studio/timers/timers.yaml" and scenario.name == "timer_oven_start"
    )

    assert scenario.expected[0].kind == "timer"
    assert scenario.expected[0].entity_ids == ("timer.oven",)
    assert scenario.expected[0].payload == {"minutes": 2, "seconds": 30}


def test_projection_ignores_optional_articles_in_device_references():
    corpus = load_corpus()
    example = next(
        example
        for example in corpus.examples
        if example.request.turns == ("Close the blinds in the master bedroom",)
    )

    assert [operation.entity_ids for operation in example.expected] == [
        ("cover.master_blinds",)
    ]


def test_collective_wording_retains_every_explicit_target():
    corpus = load_corpus()
    example = next(
        example
        for example in corpus.examples
        if example.request.turns == ("Unlock both the front door and garage entry",)
    )

    assert {operation.entity_ids for operation in example.expected} == {
        ("lock.front_door",),
        ("lock.garage_entry",),
    }


def test_coordinated_area_wording_retains_elided_area_targets():
    corpus = load_corpus()
    example = next(
        example
        for example in corpus.examples
        if example.request.turns
        == (
            "What is the status of the lights in bedroom 3 and the kitchen?",
            "Turn them on.",
        )
    )

    assert {operation.entity_ids for operation in example.expected} == {
        ("light.bedroom3_ceiling",),
        ("light.kitchen_bench",),
        ("light.kitchen_ceiling",),
    }
