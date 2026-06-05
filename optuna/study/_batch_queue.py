from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import replace
import datetime
from enum import Enum
from typing import Any
from typing import TYPE_CHECKING

from optuna.distributions import BaseDistribution
from optuna.study._batch import _BATCH_RESOURCE_ASSIGNMENT_ATTR
from optuna.study._batch import _BATCH_TRIAL_LEASE_ATTR
from optuna.study._batch import BatchAskMetadata
from optuna.study._batch import BatchFallbackMode
from optuna.study._batch import BatchGeneratorMode
from optuna.study._batch import BatchResourceAssignment
from optuna.study._batch import BatchTellInput
from optuna.study._batch import BatchTellResult
from optuna.study._batch import BatchTellStatus
from optuna.study._batch import BatchTrialHandle
from optuna.study._batch import BatchTrialLease
from optuna.study._batch import create_batch_resource_assignment
from optuna.study._batch import create_batch_trial_lease
from optuna.study._batch import get_batch_trial_lease
from optuna.study._batch import normalize_batch_generator_mode
from optuna.trial import Trial
from optuna.trial import TrialState


if TYPE_CHECKING:
    from optuna.study import Study
    from optuna.trial import FrozenTrial


_BATCH_QUEUE_ENTRY_ATTR = "batch:queue_entry"


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
    sampler_snapshot_id: str | None
    reservation_order: int | None
    queue_age: datetime.timedelta | None
    ready_count: int
    inflight_count: int
    fallback_mode: BatchFallbackMode | None
    generator_mode: BatchGeneratorMode | None
    resource_assignment: BatchResourceAssignment | None


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
class BatchQueueRecoveryResult:
    """Result of reconstructing queue state from durable trial metadata."""

    recovered_ready_count: int
    recovered_inflight_count: int
    skipped_count: int
    ready_count: int
    inflight_count: int
    recovered_trial_numbers: list[int]
    skipped_trial_numbers: list[int]


@dataclass(frozen=True)
class _QueueEntry:
    queue_id: str
    batch_id: str
    sampler_snapshot_id: str | None
    generator_mode: BatchGeneratorMode
    reservation_order: int
    queued_at: datetime.datetime
    fallback_mode: BatchFallbackMode
    acquired_at: datetime.datetime | None


@dataclass(frozen=True)
class _QueuedCandidate:
    trial_handle: BatchTrialHandle
    batch_id: str
    sampler_snapshot_id: str | None
    generator_mode: BatchGeneratorMode
    reservation_order: int
    queued_at: datetime.datetime
    fallback_mode: BatchFallbackMode


@dataclass(frozen=True)
class _InflightCandidate:
    trial_handle: BatchTrialHandle
    batch_id: str
    sampler_snapshot_id: str | None
    generator_mode: BatchGeneratorMode
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
        queue_id: str = "default",
        generator_mode: BatchGeneratorMode | str | None = None,
        generator_seed: int | None = None,
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
        if not isinstance(queue_id, str):
            raise TypeError("queue_id must be a string.")
        if not queue_id:
            raise ValueError("queue_id must be a non-empty string.")
        normalized_generator_mode = normalize_batch_generator_mode(generator_mode)
        if generator_seed is not None and (
            not isinstance(generator_seed, int) or isinstance(generator_seed, bool)
        ):
            raise TypeError("generator_seed must be an integer when provided.")
        if (
            generator_seed is not None
            and normalized_generator_mode is BatchGeneratorMode.ADAPTIVE
        ):
            raise ValueError("generator_seed requires a non-adaptive generator_mode.")
        if (
            normalized_generator_mode is BatchGeneratorMode.RANDOM
            and not fixed_distributions
        ):
            raise ValueError("generator_mode='random' requires fixed_distributions.")

        self._study = study
        self._batch_size = batch_size
        self._max_queue_size = max_queue_size
        self._max_inflight = max_inflight
        self._fixed_distributions = fixed_distributions
        self._lease_timeout = lease_timeout or datetime.timedelta(minutes=5)
        self._max_snapshot_age = max_snapshot_age
        self._queue_id = queue_id
        self._generator_mode = normalized_generator_mode
        self._generator_seed = generator_seed
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
            capacity,
            fixed_distributions=self._fixed_distributions,
            generator_mode=self._generator_mode,
            generator_seed=self._generator_seed,
        )
        queued_at = datetime.datetime.now(datetime.timezone.utc)
        for reservation_order, trial_handle in enumerate(ask_result.trial_handles):
            queue_entry = _QueueEntry(
                queue_id=self._queue_id,
                batch_id=ask_result.metadata.batch_id,
                sampler_snapshot_id=ask_result.metadata.sampler_snapshot_id,
                generator_mode=ask_result.metadata.generator_mode,
                reservation_order=reservation_order,
                queued_at=queued_at,
                fallback_mode=ask_result.metadata.fallback_mode,
                acquired_at=None,
            )
            self._write_queue_entry(trial_handle, queue_entry)
            self._ready.append(
                _QueuedCandidate(
                    trial_handle=trial_handle,
                    batch_id=queue_entry.batch_id,
                    sampler_snapshot_id=queue_entry.sampler_snapshot_id,
                    generator_mode=queue_entry.generator_mode,
                    reservation_order=reservation_order,
                    queued_at=queued_at,
                    fallback_mode=queue_entry.fallback_mode,
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

    def acquire(
        self,
        worker_id: str,
        resource_profile: Mapping[str, Any] | None = None,
    ) -> BatchQueueAcquireResult:
        """Lease one ready queued candidate to a worker without reserving more trials."""

        if not isinstance(worker_id, str):
            raise TypeError("worker_id must be a string.")
        if not worker_id:
            raise ValueError("worker_id must be a non-empty string.")
        if resource_profile is not None and not isinstance(resource_profile, Mapping):
            raise TypeError("resource_profile must be a mapping when provided.")

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
                sampler_snapshot_id=candidate.sampler_snapshot_id,
                reservation_order=candidate.reservation_order,
                queue_age=queue_age,
                ready_count=len(self._ready),
                inflight_count=len(self._inflight),
                fallback_mode=candidate.fallback_mode,
                generator_mode=candidate.generator_mode,
                resource_assignment=None,
            )

        resource_assignment = (
            create_batch_resource_assignment(worker_id, now, resource_profile)
            if resource_profile is not None
            else None
        )
        candidate = self._ready.popleft()
        leased_handle = self._attach_lease(candidate.trial_handle, worker_id)
        if resource_assignment is not None:
            self._write_resource_assignment(leased_handle, resource_assignment)
        self._write_queue_entry(
            leased_handle,
            _QueueEntry(
                queue_id=self._queue_id,
                batch_id=candidate.batch_id,
                sampler_snapshot_id=candidate.sampler_snapshot_id,
                generator_mode=candidate.generator_mode,
                reservation_order=candidate.reservation_order,
                queued_at=candidate.queued_at,
                fallback_mode=candidate.fallback_mode,
                acquired_at=now,
            ),
        )
        self._inflight[leased_handle.number] = _InflightCandidate(
            trial_handle=leased_handle,
            batch_id=candidate.batch_id,
            sampler_snapshot_id=candidate.sampler_snapshot_id,
            generator_mode=candidate.generator_mode,
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
            sampler_snapshot_id=candidate.sampler_snapshot_id,
            reservation_order=candidate.reservation_order,
            queue_age=queue_age,
            ready_count=len(self._ready),
            inflight_count=len(self._inflight),
            fallback_mode=candidate.fallback_mode,
            generator_mode=candidate.generator_mode,
            resource_assignment=resource_assignment,
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
                    sampler_snapshot_id=inflight_candidate.sampler_snapshot_id,
                    generator_mode=inflight_candidate.generator_mode,
                    reservation_order=inflight_candidate.reservation_order,
                    queued_at=inflight_candidate.queued_at,
                    fallback_mode=inflight_candidate.fallback_mode,
                )
            )
            self._write_queue_entry(
                ready_handle,
                _QueueEntry(
                    queue_id=self._queue_id,
                    batch_id=inflight_candidate.batch_id,
                    sampler_snapshot_id=inflight_candidate.sampler_snapshot_id,
                    generator_mode=inflight_candidate.generator_mode,
                    reservation_order=inflight_candidate.reservation_order,
                    queued_at=inflight_candidate.queued_at,
                    fallback_mode=inflight_candidate.fallback_mode,
                    acquired_at=None,
                ),
            )
            reclaimed_trial_numbers.append(trial_number)

        return BatchQueueReclaimResult(
            reclaimed_count=len(reclaimed_trial_numbers),
            blocked_count=blocked_count,
            ready_count=len(self._ready),
            inflight_count=len(self._inflight),
            reclaimed_trial_numbers=reclaimed_trial_numbers,
        )

    def recover(self) -> BatchQueueRecoveryResult:
        """Rebuild ready and inflight queue state from durable RUNNING-trial metadata."""

        self._ready.clear()
        self._inflight.clear()

        recovered_trial_numbers: list[int] = []
        skipped_trial_numbers: list[int] = []
        queue_entries: list[tuple[FrozenTrial, _QueueEntry, BatchTrialLease | None]] = []
        trials = self._study._storage.get_all_trials(
            self._study._study_id, deepcopy=False, states=(TrialState.RUNNING,)
        )

        for frozen_trial in trials:
            queue_entry = _get_batch_queue_entry(frozen_trial.system_attrs)
            if queue_entry is None or queue_entry.queue_id != self._queue_id:
                continue
            queue_entries.append(
                (
                    frozen_trial,
                    queue_entry,
                    get_batch_trial_lease(frozen_trial.system_attrs),
                )
            )

        queue_entries.sort(
            key=lambda entry: (
                entry[1].queued_at,
                entry[1].batch_id,
                entry[1].reservation_order,
                entry[0].number,
            )
        )

        for frozen_trial, queue_entry, lease in queue_entries:
            if len(self._ready) + len(self._inflight) >= self._max_inflight:
                skipped_trial_numbers.append(frozen_trial.number)
                continue

            trial_handle = BatchTrialHandle(
                trial=Trial(self._study, frozen_trial._trial_id),
                number=frozen_trial.number,
                lease=lease,
            )
            if lease is None:
                if len(self._ready) >= self._max_queue_size:
                    skipped_trial_numbers.append(frozen_trial.number)
                    continue
                self._ready.append(
                    _QueuedCandidate(
                        trial_handle=trial_handle,
                        batch_id=queue_entry.batch_id,
                        sampler_snapshot_id=queue_entry.sampler_snapshot_id,
                        generator_mode=queue_entry.generator_mode,
                        reservation_order=queue_entry.reservation_order,
                        queued_at=queue_entry.queued_at,
                        fallback_mode=queue_entry.fallback_mode,
                    )
                )
            else:
                self._inflight[trial_handle.number] = _InflightCandidate(
                    trial_handle=trial_handle,
                    batch_id=queue_entry.batch_id,
                    sampler_snapshot_id=queue_entry.sampler_snapshot_id,
                    generator_mode=queue_entry.generator_mode,
                    reservation_order=queue_entry.reservation_order,
                    queued_at=queue_entry.queued_at,
                    acquired_at=queue_entry.acquired_at or queue_entry.queued_at,
                    fallback_mode=queue_entry.fallback_mode,
                )
            recovered_trial_numbers.append(frozen_trial.number)

        return BatchQueueRecoveryResult(
            recovered_ready_count=len(self._ready),
            recovered_inflight_count=len(self._inflight),
            skipped_count=len(skipped_trial_numbers),
            ready_count=len(self._ready),
            inflight_count=len(self._inflight),
            recovered_trial_numbers=recovered_trial_numbers,
            skipped_trial_numbers=skipped_trial_numbers,
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
            sampler_snapshot_id=None,
            reservation_order=None,
            queue_age=None,
            ready_count=len(self._ready),
            inflight_count=len(self._inflight),
            fallback_mode=None,
            generator_mode=None,
            resource_assignment=None,
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

    def _write_queue_entry(
        self, trial_handle: BatchTrialHandle, queue_entry: _QueueEntry
    ) -> None:
        queue_entry_attrs = {
            "queue_id": queue_entry.queue_id,
            "batch_id": queue_entry.batch_id,
            "sampler_snapshot_id": queue_entry.sampler_snapshot_id,
            "generator_mode": queue_entry.generator_mode.value,
            "reservation_order": queue_entry.reservation_order,
            "queued_at": queue_entry.queued_at.isoformat(),
            "fallback_mode": queue_entry.fallback_mode.value,
            "acquired_at": (
                queue_entry.acquired_at.isoformat()
                if queue_entry.acquired_at is not None
                else None
            ),
        }
        self._study._storage.set_trial_system_attr(
            trial_handle.trial._trial_id, _BATCH_QUEUE_ENTRY_ATTR, queue_entry_attrs
        )
        trial_handle.trial._cached_frozen_trial.system_attrs[
            _BATCH_QUEUE_ENTRY_ATTR
        ] = queue_entry_attrs

    def _write_resource_assignment(
        self,
        trial_handle: BatchTrialHandle,
        resource_assignment: BatchResourceAssignment,
    ) -> None:
        resource_assignment_attrs = resource_assignment.to_system_attrs()
        self._study._storage.set_trial_system_attr(
            trial_handle.trial._trial_id,
            _BATCH_RESOURCE_ASSIGNMENT_ATTR,
            resource_assignment_attrs,
        )
        trial_handle.trial._cached_frozen_trial.system_attrs[
            _BATCH_RESOURCE_ASSIGNMENT_ATTR
        ] = resource_assignment_attrs

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


def _get_batch_queue_entry(system_attrs: dict[str, Any]) -> _QueueEntry | None:
    raw_queue_entry = system_attrs.get(_BATCH_QUEUE_ENTRY_ATTR)
    if not isinstance(raw_queue_entry, dict):
        return None

    queue_id = raw_queue_entry.get("queue_id")
    batch_id = raw_queue_entry.get("batch_id")
    sampler_snapshot_id = raw_queue_entry.get("sampler_snapshot_id")
    generator_mode = raw_queue_entry.get(
        "generator_mode", BatchGeneratorMode.ADAPTIVE.value
    )
    reservation_order = raw_queue_entry.get("reservation_order")
    queued_at = raw_queue_entry.get("queued_at")
    fallback_mode = raw_queue_entry.get("fallback_mode")
    acquired_at = raw_queue_entry.get("acquired_at")
    if not isinstance(queue_id, str):
        return None
    if not isinstance(batch_id, str):
        return None
    if sampler_snapshot_id is not None and not isinstance(sampler_snapshot_id, str):
        return None
    if not isinstance(generator_mode, str):
        return None
    if not isinstance(reservation_order, int):
        return None
    if not isinstance(queued_at, str):
        return None
    if not isinstance(fallback_mode, str):
        return None
    if acquired_at is not None and not isinstance(acquired_at, str):
        return None

    try:
        parsed_queued_at = datetime.datetime.fromisoformat(queued_at)
        parsed_acquired_at = (
            datetime.datetime.fromisoformat(acquired_at)
            if acquired_at is not None
            else None
        )
        parsed_fallback_mode = BatchFallbackMode(fallback_mode)
        parsed_generator_mode = BatchGeneratorMode(generator_mode)
    except ValueError:
        return None

    return _QueueEntry(
        queue_id=queue_id,
        batch_id=batch_id,
        sampler_snapshot_id=sampler_snapshot_id,
        generator_mode=parsed_generator_mode,
        reservation_order=reservation_order,
        queued_at=parsed_queued_at,
        fallback_mode=parsed_fallback_mode,
        acquired_at=parsed_acquired_at,
    )
