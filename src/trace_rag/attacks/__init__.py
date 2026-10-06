"""Attack transformations and chronological schedules for Person C."""

from .adaptive import AttackEvent, framing, hit_and_run, signal_evasion, slow_burn
from .poisoned_rag import (
    PoisonedDocument,
    entity_swap,
    instruction_injection,
    make_poisoned_document,
    negation,
)
from .stream import (
    StreamIngestion,
    StreamQuery,
    StreamResult,
    build_ingestion_schedule,
    build_query_stream,
    validate_target_order,
    zipf_queries,
)

ATTACK_GENERATORS = {
    "entity_swap": entity_swap,
    "negation": negation,
    "instruction_injection": instruction_injection,
    "hit_and_run": hit_and_run,
    "slow_burn": slow_burn,
    "framing": framing,
    "signal_evasion": signal_evasion,
}

ATTACK_NAMES = tuple(ATTACK_GENERATORS.keys())

__all__ = [
    "PoisonedDocument", "AttackEvent", "StreamQuery", "StreamIngestion", "StreamResult",
    "entity_swap", "negation", "instruction_injection", "make_poisoned_document",
    "hit_and_run", "slow_burn", "framing", "signal_evasion", "zipf_queries",
    "build_query_stream", "build_ingestion_schedule", "validate_target_order",
    "ATTACK_GENERATORS", "ATTACK_NAMES",
]
