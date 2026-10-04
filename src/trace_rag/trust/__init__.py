"""Person B: trust ledger, corroboration verifier, and escalation policy.

This package implements the three protocols that Person A's pipeline expects
from ``contracts.py``:

  * :class:`TrustLedger`              — ``TrustProvider`` (read + write)
  * :class:`CorroborationVerifier`    — ``Verifier``
  * :class:`TrustPolicy`             — ``Policy``

All three work offline (hashing embedder + stub LLM) and require no GPU.
"""

from . import plots
from .ledger import TrustLedger, TrustConfig
from .verifier import CorroborationVerifier
from .policy import TrustPolicy
from .queue import VerificationQueue, PendingVerification

__all__ = ["TrustLedger", "TrustConfig", "CorroborationVerifier", "TrustPolicy",
           "VerificationQueue", "PendingVerification", "plots"]
