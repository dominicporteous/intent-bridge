"""Deterministic natural-language intent recognition."""

from intent_bridge.intent_engine.grammar import (
    GrammarDependencyError,
    GrammarLoadError,
    IntentGrammarLoader,
    LoadedIntentGrammar,
    SentenceProvenance,
    load_intent_grammar,
)
from intent_bridge.intent_engine.natural_language import (
    NaturalLanguageIntentPlanner,
    NaturalLanguageIntentRecognizer,
    split_compound_request,
)
from intent_bridge.intent_engine.planning import IntentPlannerChain
from intent_bridge.intent_engine.supplemental import (
    DialogueState,
    PendingClarification,
    PlanningSession,
    PlanningTurn,
    SupplementalIntentPlanner,
)

__all__ = [
    "DialogueState",
    "GrammarDependencyError",
    "GrammarLoadError",
    "IntentGrammarLoader",
    "LoadedIntentGrammar",
    "NaturalLanguageIntentPlanner",
    "NaturalLanguageIntentRecognizer",
    "IntentPlannerChain",
    "PendingClarification",
    "PlanningSession",
    "PlanningTurn",
    "SentenceProvenance",
    "SupplementalIntentPlanner",
    "load_intent_grammar",
    "split_compound_request",
]
