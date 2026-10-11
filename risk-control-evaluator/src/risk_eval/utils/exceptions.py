"""Custom exception hierarchy for risk-eval."""


class RiskEvalError(Exception):
    """Base exception for all risk-eval errors."""

    def __init__(self, message: str, *, context: dict | None = None) -> None:
        """Initialize with message and optional context.

        Args:
            message: Human-readable error description.
            context: Additional structured context for debugging.
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        if self.context:
            ctx = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} ({ctx})"
        return self.message


class DataLoadError(RiskEvalError):
    """Raised when data loading fails."""


class DataValidationError(RiskEvalError):
    """Raised when data fails validation checks."""


class RuleConfigError(RiskEvalError):
    """Raised when rule configuration is invalid."""


class RuleExecutionError(RiskEvalError):
    """Raised when rule execution encounters an error."""


class EvaluationError(RiskEvalError):
    """Raised when evaluation calculation fails."""


class ReportGenerationError(RiskEvalError):
    """Raised when report generation fails."""
