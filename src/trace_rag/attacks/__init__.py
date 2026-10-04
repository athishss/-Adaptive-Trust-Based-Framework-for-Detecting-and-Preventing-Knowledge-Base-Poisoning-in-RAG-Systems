"""Attack registry for Person C experiments."""

from trace_rag.attacks.adaptive import (
    framing,
    hit_and_run,
    signal_evasion,
    slow_burn,
)
from trace_rag.attacks.poisoned_rag import (
    entity_swap,
    instruction_injection,
    negation,
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
