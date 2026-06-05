from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
import datetime
from enum import Enum
from typing import TYPE_CHECKING

from optuna.distributions import BaseDistribution
from optuna.study._batch import _BATCH_TRIAL_LEASE_ATTR
from optuna.study._batch import BatchAskMetadata
from optuna.study._batch import BatchFallbackMode
from optuna.study._batch import BatchTellInput
from optuna.study._batch import BatchTellResult
from optuna.study._batch import BatchTellStatus
from optuna.study._batch import BatchTrialHandle
from optuna.study._batch import BatchTrialLease
from optuna.study._batch import create_batch_trial_lease
from optuna.study._batch import get_batch_trial_lease
from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna.study import Study


class BatchQueueAcquireStatus(Enum):
    """Status returned by one candidate queue acquisition attempt."""

    READY = "ready"
    EMPTY = "empty"
    BACKPRESSURE = "backpressure"
    STALE = "stale"


class BatchQueueRefillStatus(Enum):
    """Status returned by one candidate queue refill attempt."""

    RESERVED = "reserved"
    BACKPRESSURE = "backpressure"


class BatchQueueRenewStatus(Enum):
    """Status returned by one queue lease renewal attempt."""

    RENEWED = "renewed"
    EXPIRED = "expired"
    NOT_FOUND = "not_found"
    TOKEN_MISMATCH = "token_mismatch"


@dataclass(frozen=True)
class BatchQueueRefillResult:
    """Result of filling a bounded candidate queue from batch reservation."""

    status: BatchQueueRefillStatus
    requested_count: int
    reserved_count: int
    ready_count: int
    inflight_count: int
    ask_metadata: BatchAskMetadata | None


@dataclass(frozen=True)
class BatchQueueAcquireResult:
    """Result of acquiring one ready candidate from a bounded queue."""

    status: BatchQueueAcquireStatus
    trial_handle: BatchTrialHandle | None
    worker_id: str | None
    batch_id: str | None
    reservation_order: int | None
    queue_age: datetime.timedelta | None
    ready_count: int
    inflight_count: int
    fallback_mode: BatchFallbackMode | None


@dataclass(frozen=True)
class BatchQueueRenewResult:
    """Result of renewing one active queue lease."""

    status: BatchQueueRenewStatus
    trial_handle: BatchTrialHandle | None
    lease: BatchTrialLease | None
    ready_count: int
    inflight_count: int
    error_message: str | None = None


@dataclass(frozen=True)
class BatchQueueReclaimResult:
    """Result of moving expired inflight candidates back to the ready queue."""

    reclaimed_count: int
    blocked_count: int
    ready_count: int
    inflight_count: int
    reclaimed_trial_numbers: list[int]


@dataclass(frozen=True)
class _QueuedCandidate:
    trial_handle: BatchTrialHandle
    batch_id: str
    reservation_order: int
    queued_at: datetime.datetime
    fallback_mode: BatchFallbackMode


@dataclass(frozen=True)
class _InflightCandidate:
    trial_handle: BatchTrialHandle
    batch_id: str
    reservation_order: int
    queued_at: datetime.datetime
    acquired_at: datetime.datetime
    fallback_mode: BatchFallbackMode


class BatchCandidateQueue:
    """Bounded in-memory queue for opt-in high-throughput candidate acquisition."""

    def __init__(
        self,
        *,
        study: Study,
        batch_size: int,
        max_queue_size: int,
        max_inflight: int,
        fixed_distributions: dict[str, BaseDistribution] | None = None,
        lease_timeout: datetime.timedelta | None = None,
        max_snapshot_age: datetime.timedelta | None = None,
    ) -> None:
        if not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer.")
        if not isinstance(max_queue_size, int):
            raise TypeError("max_queue_size must be an integer.")
        if not isinstance(max_inflight, int):
            raise TypeError("max_inflight must be an integer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if max_queue_size <= 0:
            raise ValueError("max_queue_size must be positive.")
        if max_inflight <= 0:
            raise ValueError("max_inflight must be positive.")
        if lease_timeout is not None and not isinstance(
            lease_timeout, datetime.timedelta
        ):
            raise TypeError("lease_timeout must be a datetime.timedelta when provided.")
        if lease_timeout is not None and lease_timeout <= datetime.timedelta(0):
            raise ValueError("lease_timeout must be positive.")
        if max_snapshot_age is not None and not isinstance(
            max_snapshot_age, datetime.timedelta
        ):
            raise TypeError("max_snapshot_age must be a datetime.timedelta when provided.")
        if max_snapshot_age is not None and max_snapshot_age <= datetime.timedelta(0):
            raise ValueError("max_snapshot_age must be positive.")

        self._study = study
        self._batch_size = batch_size
        self._max_queue_size = max_queue_size
        self._max_inflight = max_inflight
        self._fixed_distributions = fixed_distributions
        self._lease_timeout = lease_timeout or datetime.timedelta(minutes=5)
        self._max_snapshot_age = max_snapshot_age
        self._ready: deque[_QueuedCandidate] = deque()
        self._inflight: dict[int, _InflightCandidate] = {}

    @property
    def ready_count(self) -> int:
        return len(self._ready)

    @property
    def inflight_count(self) -> int:
        return len(self._inflight)

    def refill(self) -> BatchQueueRefillResult:
        """Fill the ready queue without exceeding queue or inflight bounds."""

        capacity = min(
            self._batch_size,
            self._max_queue_size - len(self._ready),
            self._max_inflight - len(self._ready) - len(self._inflight),
        )
        if capacity <= 0:
            return BatchQueueRefillResult(
                status=BatchQueueRefillStatus.BACKPRESSURE,
                requested_count=0,
                reserved_count=0,
                ready_count=len(self._ready),
                inflight_count=len(self._inflight),
                ask_metadata=None,
            )

        ask_result = self._study.ask_batch(
            capacity, fixed_distributions=self._fixed_distributions
        )
        queued_at = datetime.datetime.now(datetime.timezone.utc)
        for reservation_order, trial_handle in enumerate(ask_result.trial_handles):
            self._ready.append(
                _QueuedCandidate(
                    trial_handle=trial_handle,
                    batch_id=ask_result.metadata.batch_id,
                    reservation_order=reservation_order,
                    queued_at=queued_at,
                    fallback_mode=ask_result.metadata.fallback_mode,
                )
            )

        return BatchQueueRefillResult(
            status=BatchQueueRefillStatus.RESERVED,
            requested_count=capacity,
            reserved_count=len(ask_result.trial_handles),
            ready_count=len(self._ready),
            inflight_count=len(self._inflight),
            ask_metadata=ask_result.metadata,
        )

    def acquire(self, worker_id: str) -> BatchQueueAcquireResult:
        """Lease one ready queued candidate to a worker without reserving more trials."""

        if not isinstance(worker_id, str):
            raise TypeError("worker_id must be a string.")
        if not worker_id:
            raise ValueError("worker_id must be a non-empty string.")

        if len(self._inflight) >= self._max_inflight:
            return self._empty_acquire_result(BatchQueueAcquireStatus.BACKPRESSURE, worker_id)
        if not self._ready:
            return self._empty_acquire_result(BatchQueueAcquireStatus.EMPTY, worker_id)

        now = datetime.datetime.now(datetime.timezone.utc)
        candidate = self._ready[0]
        queue_age = now - candidate.queued_at
        if self._max_snapshot_age is not None and queue_age > self._max_snapshot_age:
            return BatchQueueAcquireResult(
                status=BatchQueueAcquireStatus.STALE,
                trial_handle=None,
                worker_id=worker_id,
                batch_id=candidate.batch_id,
                reservation_order=candidate.reservation_order,
                queue_age=queue_age,
                ready_count=len(self._ready),
                inflight_count=len(self._inflight),
                fallback_mode=candidate.fallback_mode,
        )

        candidate = self._ready.popleft()
        leased_handle = self._attach_lease(candidate.trial_handle, worker_id)
        self._inflight[leased_handle.number] = _InflightCandidate(
            trial_handle=leased_handle,
            batch_id=candidate.batch_id,
            reservation_order=candidate.reservation_order,
            queued_at=candidate.queued_at,
            acquired_at=now,
            fallback_mode=candidate.fallback_mode,
        )
        return BatchQueueAcquireResult(
            status=BatchQueueAcquireStatus.READY,
            trial_handle=leased_handle,
            worker_id=worker_id,
            batch_id=candidate.batch_id,
            reservation_order=candidate.reservation_order,
            queue_age=queue_age,
            ready_count=len(self._ready),
            inflight_count=len(self._inflight),
            fallback_mode=candidate.fallback_mode,
        )

    def renew(self, trial_handle: BatchTrialHandle) -> BatchQueueRenewResult:
        """Renew an active queue lease if its token still owns the inflight trial."""

        if trial_handle.lease is None:
            raise ValueError("trial_handle must include a lease.")

        inflight_candidate = self._inflight.get(trial_handle.number)
        if inflight_candidate is None:
            return self._renew_result(
                BatchQueueRenewStatus.NOT_FOUND,
                None,
                None,
                "Trial is not currently inflight in this queue.",
            )

        current_lease = self._get_current_lease(trial_handle)
        if current_lease is None or current_lease.token != trial_handle.lease.token:
            return self._renew_result(
                BatchQueueRenewStatus.TOKEN_MISMATCH,
                inflight_candidate.trial_handle,
                current_lease,
                "Batch lease token does not match the active trial lease.",
            )

        now = datetime.datetime.now(datetime.timezone.utc)
        if current_lease.deadline <= now:
            return self._renew_result(
                BatchQueueRenewStatus.EXPIRED,
                inflight_candidate.trial_handle,
                current_lease,
                "Batch lease has expired.",
            )

        renewed_lease = BatchTrialLease(
            owner=current_lease.owner,
            token=current_lease.token,
            deadline=max(now, current_lease.deadline) + self._lease_timeout,
            renewal_count=current_lease.renewal_count + 1,
        )
        renewed_handle = BatchTrialHandle(
            trial=trial_handle.trial, number=trial_handle.number, lease=renewed_lease
        )
        self._write_lease(renewed_handle, renewed_lease)
        self._inflight[renewed_handle.number] = replace(
            inflight_candidate, trial_handle=renewed_handle
        )
        return self._renew_result(
            BatchQueueRenewStatus.RENEWED, renewed_handle, renewed_lease, None
        )

    def reclaim_expired(self) -> BatchQueueReclaimResult:
        """Move expired inflight leases back to the ready queue when capacity allows."""

        now = datetime.datetime.now(datetime.timezone.utc)
        reclaimed_trial_numbers: list[int] = []
        blocked_count = 0

        for trial_number, inflight_candidate in list(self._inflight.items()):
            current_lease = self._get_current_lease(inflight_candidate.trial_handle)
            if current_lease is None or current_lease.deadline > now:
                continue
            if len(self._ready) >= self._max_queue_size:
                blocked_count += 1
                continue

            self._inflight.pop(trial_number)
            ready_handle = BatchTrialHandle(
                trial=inflight_candidate.trial_handle.trial,
                number=inflight_candidate.trial_handle.number,
                lease=None,
            )
            self._ready.append(
                _QueuedCandidate(
                    trial_handle=ready_handle,
                    batch_id=inflight_candidate.batch_id,
                    reservation_order=inflight_candidate.reservation_order,
                    queued_at=inflight_candidate.queued_at,
                    fallback_mode=inflight_candidate.fallback_mode,
                )
            )
            reclaimed_trial_numbers.append(trial_number)

        return BatchQueueReclaimResult(
            reclaimed_count=len(reclaimed_trial_numbers),
            blocked_count=blocked_count,
            ready_count=len(self._ready),
            inflight_count=len(self._inflight),
            reclaimed_trial_numbers=reclaimed_trial_numbers,
        )

    def tell(
        self,
        trial_handle: BatchTrialHandle,
        values: float | Sequence[float] | None = None,
        state: TrialState | None = None,
    ) -> BatchTellResult:
        """Complete one leased candidate and release inflight capacity on acceptance."""

        if trial_handle.lease is None:
            raise ValueError("trial_handle must include a lease.")

        result = self._study.tell_batch(
            [
                BatchTellInput(
                    trial=trial_handle.trial,
                    values=values,
                    state=state,
                    lease_token=trial_handle.lease.token,
                )
            ]
        )
        releasable_statuses = {BatchTellStatus.ACCEPTED, BatchTellStatus.SKIPPED}
        if result.outcomes and result.outcomes[0].status in releasable_statuses:
            self._inflight.pop(trial_handle.number, None)
        return result

    def _empty_acquire_result(
        self, status: BatchQueueAcquireStatus, worker_id: str
    ) -> BatchQueueAcquireResult:
        return BatchQueueAcquireResult(
            status=status,
            trial_handle=None,
            worker_id=worker_id,
            batch_id=None,
            reservation_order=None,
            queue_age=None,
            ready_count=len(self._ready),
            inflight_count=len(self._inflight),
            fallback_mode=None,
        )

    def _attach_lease(
        self, trial_handle: BatchTrialHandle, worker_id: str
    ) -> BatchTrialHandle:
        lease = create_batch_trial_lease(worker_id, self._lease_timeout)
        leased_handle = BatchTrialHandle(
            trial=trial_handle.trial, number=trial_handle.number, lease=lease
        )
        self._write_lease(leased_handle, lease)
        return leased_handle

    def _write_lease(
        self, trial_handle: BatchTrialHandle, lease: BatchTrialLease
    ) -> None:
        lease_attrs = lease.to_system_attrs()
        self._study._storage.set_trial_system_attr(
            trial_handle.trial._trial_id, _BATCH_TRIAL_LEASE_ATTR, lease_attrs
        )
        trial_handle.trial._cached_frozen_trial.system_attrs[
            _BATCH_TRIAL_LEASE_ATTR
        ] = lease_attrs

    def _get_current_lease(
        self, trial_handle: BatchTrialHandle
    ) -> BatchTrialLease | None:
        frozen_trial = self._study._storage.get_trial(trial_handle.trial._trial_id)
        return get_batch_trial_lease(frozen_trial.system_attrs)

    def _renew_result(
        self,
        status: BatchQueueRenewStatus,
        trial_handle: BatchTrialHandle | None,
        lease: BatchTrialLease | None,
        error_message: str | None,
    ) -> BatchQueueRenewResult:
        return BatchQueueRenewResult(
            status=status,
            trial_handle=trial_handle,
            lease=lease,
            ready_count=len(self._ready),
            inflight_count=len(self._inflight),
            error_message=error_message,
        )
