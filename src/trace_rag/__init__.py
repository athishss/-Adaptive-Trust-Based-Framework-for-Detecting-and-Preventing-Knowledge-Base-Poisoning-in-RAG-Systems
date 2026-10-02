"""TRACE-RAG - Person A: retrieval, detection signals and grounded generation.

Quick start::

    from trace_rag import Config, PersonAPipeline
    pipeline = PersonAPipeline.from_config(Config())
    result = pipeline.answer("Who designed the Eiffel Tower?")
    print(result.answer, result.record.abstained)

With Person B (trust framework)::

    from trace_rag import Config, PersonAPipeline
    from trace_rag.trust import TrustLedger, CorroborationVerifier, TrustPolicy

    pipeline = PersonAPipeline.from_config(Config.load("config/default.yaml"))
    ledger = TrustLedger("runs/default/trust.sqlite3", store=pipeline.store)
    pipeline.trust_provider = ledger
    pipeline.retriever.trust = ledger
    pipeline.verifier = CorroborationVerifier(llm=pipeline.generator.llm)
    pipeline.policy = TrustPolicy(ledger=ledger, on_quarantine=pipeline.on_quarantine)
"""

from .config import Config
from .contracts import (Action, AnswerRecord, Band, Citation, DefaultPolicy, FeatureSnapshot,
                        NullTrustProvider, NullVerifier, PolicyDecision, RetrievedDocument,
                        SecurityAssessment, SignalVector, TrustSnapshot, TrustStatus,
                        VerificationOutcome, VerificationResult, CONTRACT_VERSION)
from .pipeline import PersonAPipeline, QueryResult

__version__ = "0.1.0"
__all__ = ["Config", "PersonAPipeline", "QueryResult", "CONTRACT_VERSION", "__version__",
           "RetrievedDocument", "SecurityAssessment", "AnswerRecord", "TrustSnapshot",
           "TrustStatus", "Band", "Action", "SignalVector", "FeatureSnapshot", "Citation",
           "PolicyDecision", "VerificationResult", "VerificationOutcome",
           "NullTrustProvider", "NullVerifier", "DefaultPolicy"]

