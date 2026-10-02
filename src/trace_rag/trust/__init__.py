"""Person B: trust ledger, corroboration verifier, and escalation policy.

This package implements the three protocols that Person A's pipeline expects
from ``contracts.py``:

  * :class:`TrustLedger`              — ``TrustProvider`` (read + write)
  * :class:`CorroborationVerifier`    — ``Verifier``
  * :class:`TrustPolicy`             — ``Policy``

All three work offline (hashing embedder + stub LLM) and require no GPU.
"""

from .ledger import TrustLedger, TrustConfig
from .verifier import CorroborationVerifier
from .policy import TrustPolicy

__all__ = ["TrustLedger", "TrustConfig", "CorroborationVerifier", "TrustPolicy"]
