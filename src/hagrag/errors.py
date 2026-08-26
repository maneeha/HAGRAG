class HAGRAGError(RuntimeError):
    """Base error for expected HAGRAG runtime failures."""


class ConfigurationError(HAGRAGError):
    """Raised when the runtime configuration is incomplete or invalid."""


class DataError(HAGRAGError):
    """Raised when required experiment data is missing or malformed."""
