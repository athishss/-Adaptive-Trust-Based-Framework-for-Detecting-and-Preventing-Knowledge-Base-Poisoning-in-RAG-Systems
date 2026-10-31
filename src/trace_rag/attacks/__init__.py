"""Attack registry for Person C experiments.

Every name in :data:`ATTACK_GENERATORS` maps to a function that performs the
transformation it is named after; :data:`ATTACK_SPECS` documents the
assumptions, transformation, insertion point and ground-truth label for each.
"""

from trace_rag.attacks.adaptive import (
    DEFAULT_WARMUP,
    AttackEvent,
    framing,
    hit_and_run,
    poison_events,
    schedule_bounds,
    signal_evasion,
    slow_burn,
)
from trace_rag.attacks.poisoned_rag import (
    ATTACK_SPECS,
    PoisonedDocument,
    entity_swap,
    instruction_injection,
    make_poisoned_document,
    negation,
)
from trace_rag.attacks.stream import (
    INGESTION_ROW_KEYS,
    QUERY_ROW_KEYS,
    StreamContractError,
    StreamIngestion,
    StreamPlan,
    StreamQuery,
    as_ingestion_row,
    as_query_row,
    build_ingestion_schedule,
    build_query_stream,
    build_stream_plan,
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
    "ATTACK_GENERATORS", "ATTACK_NAMES", "ATTACK_SPECS",
    "PoisonedDocument", "entity_swap", "negation", "instruction_injection",
    "make_poisoned_document",
    "AttackEvent", "hit_and_run", "slow_burn", "framing", "signal_evasion",
    "DEFAULT_WARMUP", "poison_events", "schedule_bounds",
    "StreamQuery", "StreamIngestion", "StreamPlan", "StreamContractError",
    "build_query_stream", "build_ingestion_schedule", "build_stream_plan",
    "zipf_queries", "as_query_row", "as_ingestion_row",
    "QUERY_ROW_KEYS", "INGESTION_ROW_KEYS",
]
