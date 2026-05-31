from __future__ import annotations


class PyHFOAgentError(Exception):
    """Structured error for agent-facing PyHFO entrypoints."""

    def __init__(self, code: str, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.details = dict(details or {})

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


def error_response(error: Exception) -> dict:
    if isinstance(error, PyHFOAgentError):
        return error.to_dict()
    return {
        "code": error.__class__.__name__,
        "message": str(error),
        "details": {},
    }
