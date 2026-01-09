class WorkerError(Exception):
    """Base worker error."""


class JobValidationError(WorkerError):
    """Raised when job payload is invalid."""


class ProcessingError(WorkerError):
    """Raised when processing fails."""


class InfrastructureError(WorkerError):
    """Raised when external services fail."""
