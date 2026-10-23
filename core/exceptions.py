"""
FinSight AI Custom Exceptions
Structured hierarchy for financial, retrieval, agent, and ingestion domain errors.
"""

class FinSightException(Exception):
    """Base exception for all FinSight AI errors."""
    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


# Financial Analytics Exceptions
class FinancialCalculationError(FinSightException):
    """Raised when a financial calculation cannot be performed."""
    pass


class ZeroDenominatorError(FinancialCalculationError):
    """Raised when a ratio calculation encounters a zero or undefined denominator."""
    pass


class UnitMismatchError(FinancialCalculationError):
    """Raised when comparing or operating on figures with incompatible units or currencies."""
    pass


class MissingFactError(FinancialCalculationError):
    """Raised when a required financial fact is missing for a specified period or metric."""
    pass


class InconsistentPeriodError(FinancialCalculationError):
    """Raised when attempting multi-period analysis across misaligned reporting intervals."""
    pass


# Retrieval & Document Ingestion Exceptions
class DocumentIngestionError(FinSightException):
    """Raised during document parsing, chunking, or indexing failures."""
    pass


class UnsupportedFileTypeError(DocumentIngestionError):
    """Raised when an uploaded file format is not supported."""
    pass


class RetrievalError(FinSightException):
    """Raised when vector or hybrid search execution fails."""
    pass


# Agent & Security Exceptions
class PromptInjectionDetected(FinSightException):
    """Raised when untrusted text contains adversarial prompt injection sequences."""
    pass


class EvidenceVerificationFailed(FinSightException):
    """Raised when citations or claims cannot be grounded in verified source passages."""
    pass


class QueryRefinementLimitExceeded(FinSightException):
    """Raised when bounded query retries are exhausted without finding sufficient evidence."""
    pass
