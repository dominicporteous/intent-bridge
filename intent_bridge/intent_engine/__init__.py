"""Deterministic natural-language intent recognition."""

from intent_bridge.intent_engine.grammar import (
    GrammarDependencyError,
    GrammarLoadError,
    IntentGrammarLoader,
    LoadedIntentGrammar,
    SentenceProvenance,
    load_intent_grammar,
)
from intent_bridge.intent_engine.measurement import MeasurementIntentPlanner
from intent_bridge.intent_engine.natural_language import (
    NaturalLanguageIntentPlanner,
    NaturalLanguageIntentRecognizer,
    split_compound_request,
)
from intent_bridge.intent_engine.outcomes import (
    AmbiguousTarget,
    CapabilityMismatch,
    IncompleteCompound,
    NoTarget,
    Resolved,
    UnsupportedOperation,
)
from intent_bridge.intent_engine.planning import IntentPlannerChain
from intent_bridge.intent_engine.supplemental import (
    ClauseReferent,
    DialogueState,
    DiscourseCondition,
    DiscourseOperationFrame,
    EntityFocus,
    PendingClarification,
    PlanningSession,
    PlanningTurn,
    PropertyFocus,
    ReferentCardinality,
    SupplementalIntentPlanner,
    UnresolvedDiscourseFrame,
)

__all__ = [
    "AmbiguousTarget",
    "CapabilityMismatch",
    "ClauseReferent",
    "DialogueState",
    "DiscourseCondition",
    "DiscourseOperationFrame",
    "EntityFocus",
    "GrammarDependencyError",
    "GrammarLoadError",
    "IntentGrammarLoader",
    "LoadedIntentGrammar",
    "NaturalLanguageIntentPlanner",
    "NaturalLanguageIntentRecognizer",
    "MeasurementIntentPlanner",
    "IntentPlannerChain",
    "IncompleteCompound",
    "NoTarget",
    "PendingClarification",
    "PlanningSession",
    "PlanningTurn",
    "PropertyFocus",
    "ReferentCardinality",
    "Resolved",
    "SentenceProvenance",
    "SupplementalIntentPlanner",
    "UnresolvedDiscourseFrame",
    "UnsupportedOperation",
    "load_intent_grammar",
    "split_compound_request",
]
