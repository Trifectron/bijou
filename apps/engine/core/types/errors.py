"""Every error the engine raises. One base, one subclass per kind of failure."""

from __future__ import annotations


class EngineError(Exception):
    """Base for every error the engine raises."""


class ConfigError(EngineError):
    """A setting is out of range or two settings contradict each other."""


class AdapterError(EngineError):
    """An adapter could not be attached, loaded or activated."""


class BackendError(EngineError):
    """A backend could not build a model, train, or generate."""


class ArtifactError(EngineError):
    """A trained adapter or checkpoint that a run needs is not on disk."""


class ModelError(EngineError):
    """The chat model could not be reached or answered with something unusable."""

    def __init__(self, message: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class ToolError(EngineError):
    """A tool failed. The loop feeds the message back to the model rather than stopping."""


class SkillRuntimeError(EngineError):
    """The skill bank could not be loaded or reached, or refused a request."""


class StoreError(EngineError):
    """The session store could not read or write."""


class PlanError(EngineError):
    """A structured reply from the model did not match what was asked for."""


class ConfirmationError(EngineError):
    """A confirmation names no pending action, the wrong one, or one that expired."""
