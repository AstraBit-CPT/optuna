from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna.samplers import BaseSampler
    from optuna.storages import BaseStorage
    from optuna.trial import FrozenTrial
    from optuna.trial import Trial


class BatchCapabilityMode(Enum):
    """How a batch path was served for one capability axis."""

    NATIVE = "native"
    FALLBACK = "fallback"


class BatchFallbackMode(Enum):
    """Fallback path used to preserve correctness before native batching exists."""

    NONE = "none"
    REPEATED_SINGLE_TRIAL = "repeated_single_trial"
    REPEATED_SINGLE_SUGGESTION = "repeated_single_suggestion"
    REPEATED_SINGLE_COMPLETION = "repeated_single_completion"


@dataclass(frozen=True)
class BatchCapability:
    """Capability labels for storage reservation and sampler suggestion paths."""

    storage_batch_reservation: BatchCapabilityMode
    sampler_batch_suggestion: BatchCapabilityMode


@dataclass(frozen=True)
class BatchAskMetadata:
    """Provenance and capability labels for a batch ask result."""

    batch_id: str
    requested_count: int
    returned_count: int
    capability: BatchCapability
    fallback_mode: BatchFallbackMode


@dataclass(frozen=True)
class BatchTrialHandle:
    """Reserved trial handle returned by batch ask."""

    trial: Trial
    number: int


@dataclass(frozen=True)
class BatchAskResult:
    """Trials reserved by batch ask plus provenance metadata."""

    trial_handles: list[BatchTrialHandle]
    metadata: BatchAskMetadata

    @property
    def trials(self) -> list[Trial]:
        return [handle.trial for handle in self.trial_handles]


@dataclass(frozen=True)
class BatchTellInput:
    """Single trial completion request used by batch tell."""

    trial: Trial | int
    values: float | Sequence[float] | None = None
    state: TrialState | None = None


@dataclass(frozen=True)
class BatchTellMetadata:
    """Provenance and capability labels for a batch tell result."""

    batch_id: str
    requested_count: int
    completed_count: int
    capability: BatchCapability
    fallback_mode: BatchFallbackMode


@dataclass(frozen=True)
class BatchTellOutcome:
    """Per-trial completion result returned by batch tell."""

    trial_number: int
    state: TrialState
    values: list[float] | None
    frozen_trial: FrozenTrial
    warning_message: str | None


@dataclass(frozen=True)
class BatchTellResult:
    """Per-trial outcomes plus provenance metadata for batch tell."""

    outcomes: list[BatchTellOutcome]
    metadata: BatchTellMetadata


def fallback_batch_capability() -> BatchCapability:
    return BatchCapability(
        storage_batch_reservation=BatchCapabilityMode.FALLBACK,
        sampler_batch_suggestion=BatchCapabilityMode.FALLBACK,
    )


def get_batch_capability(storage: BaseStorage, sampler: BaseSampler) -> BatchCapability:
    sample_batch = getattr(sampler, "sample_batch", None)
    return BatchCapability(
        storage_batch_reservation=(
            BatchCapabilityMode.NATIVE
            if storage._supports_native_batch_trial_creation()
            else BatchCapabilityMode.FALLBACK
        ),
        sampler_batch_suggestion=(
            BatchCapabilityMode.NATIVE
            if callable(sample_batch)
            else BatchCapabilityMode.FALLBACK
        ),
    )
