from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from enum import Enum
from typing import Any
from typing import TYPE_CHECKING
import uuid

from optuna.trial import TrialState


if TYPE_CHECKING:
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


class BatchTellStatus(Enum):
    """Result status for one batch tell completion request."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SKIPPED = "skipped"


_BATCH_TRIAL_LEASE_ATTR = "batch:trial_lease"


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
class BatchTrialLease:
    """Ownership token for one pre-reserved batch trial."""

    owner: str
    token: str
    deadline: datetime
    renewal_count: int = 0

    def to_system_attrs(self) -> dict[str, Any]:
        return {
            "owner": self.owner,
            "token": self.token,
            "deadline": self.deadline.isoformat(),
            "renewal_count": self.renewal_count,
        }


def create_batch_trial_lease(
    owner: str, lease_timeout: timedelta, now: datetime | None = None
) -> BatchTrialLease:
    now = now or datetime.now(timezone.utc)
    return BatchTrialLease(
        owner=owner,
        token=uuid.uuid4().hex,
        deadline=now + lease_timeout,
    )


@dataclass(frozen=True)
class BatchTrialHandle:
    """Reserved trial handle returned by batch ask."""

    trial: Trial
    number: int
    lease: BatchTrialLease | None = None


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
    lease_token: str | None = None


@dataclass(frozen=True)
class BatchTellMetadata:
    """Provenance and capability labels for a batch tell result."""

    batch_id: str
    requested_count: int
    completed_count: int
    rejected_count: int
    skipped_count: int
    capability: BatchCapability
    fallback_mode: BatchFallbackMode


@dataclass(frozen=True)
class BatchTellOutcome:
    """Per-trial completion result returned by batch tell."""

    trial_number: int | None
    state: TrialState | None
    values: list[float] | None
    frozen_trial: FrozenTrial | None
    warning_message: str | None
    status: BatchTellStatus
    error_message: str | None = None
    lease: BatchTrialLease | None = None
    lease_token_valid: bool | None = None


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


def get_batch_trial_lease(system_attrs: Mapping[str, Any]) -> BatchTrialLease | None:
    raw_lease = system_attrs.get(_BATCH_TRIAL_LEASE_ATTR)
    if not isinstance(raw_lease, Mapping):
        return None

    owner = raw_lease.get("owner")
    token = raw_lease.get("token")
    deadline = raw_lease.get("deadline")
    renewal_count = raw_lease.get("renewal_count", 0)
    if not isinstance(owner, str) or not isinstance(token, str):
        return None
    if not isinstance(deadline, str):
        return None
    if not isinstance(renewal_count, int):
        return None

    try:
        parsed_deadline = datetime.fromisoformat(deadline)
    except ValueError:
        return None

    return BatchTrialLease(
        owner=owner,
        token=token,
        deadline=parsed_deadline,
        renewal_count=renewal_count,
    )


def get_batch_capability(storage: BaseStorage, native_sampler_batch_used: bool) -> BatchCapability:
    return BatchCapability(
        storage_batch_reservation=(
            BatchCapabilityMode.NATIVE
            if storage._supports_native_batch_trial_creation()
            else BatchCapabilityMode.FALLBACK
        ),
        sampler_batch_suggestion=(
            BatchCapabilityMode.NATIVE
            if native_sampler_batch_used
            else BatchCapabilityMode.FALLBACK
        ),
    )
