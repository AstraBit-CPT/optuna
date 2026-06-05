from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from enum import Enum
import hashlib
import json
import math
from typing import Any
from typing import TYPE_CHECKING
import uuid

from optuna.distributions import BaseDistribution
from optuna.distributions import distribution_to_json
from optuna.distributions import FloatDistribution
from optuna.distributions import IntDistribution
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


class BatchGeneratorMode(Enum):
    """Candidate generation mode used for a high-throughput batch."""

    ADAPTIVE = "adaptive"
    RANDOM = "random"


class BatchTellStatus(Enum):
    """Result status for one batch tell completion request."""

    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SKIPPED = "skipped"


_BATCH_TRIAL_LEASE_ATTR = "batch:trial_lease"
_BATCH_TRIAL_COMPLETION_ATTR = "batch:trial_completion"
_BATCH_TRIAL_GENERATION_ATTR = "batch:trial_generation"
_BATCH_RESOURCE_ASSIGNMENT_ATTR = "batch:resource_assignment"


@dataclass(frozen=True)
class BatchCapability:
    """Capability labels for storage reservation and sampler suggestion paths."""

    storage_batch_reservation: BatchCapabilityMode
    sampler_batch_suggestion: BatchCapabilityMode


@dataclass(frozen=True)
class BatchSuggestionDiagnostics:
    """Duplicate diagnostics for parameter suggestions produced in one batch."""

    evaluated_count: int
    pair_count: int
    duplicate_pair_count: int
    duplicate_rate: float
    near_duplicate_pair_count: int
    near_duplicate_rate: float
    near_duplicate_threshold: float


@dataclass(frozen=True)
class BatchAskMetadata:
    """Provenance and capability labels for a batch ask result."""

    batch_id: str
    requested_count: int
    returned_count: int
    capability: BatchCapability
    fallback_mode: BatchFallbackMode
    sampler_snapshot_id: str | None = None
    suggestion_diagnostics: BatchSuggestionDiagnostics | None = None
    generator_mode: BatchGeneratorMode = BatchGeneratorMode.ADAPTIVE
    generator_seed: int | None = None
    storage_time: float = 0.0
    sampler_time: float = 0.0


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


@dataclass(frozen=True)
class BatchTrialCompletion:
    """Durable idempotency metadata for one accepted batch completion."""

    completion_key: str
    completed_at: datetime
    state: TrialState
    values: list[float] | None
    request_signature: str

    def to_system_attrs(self) -> dict[str, Any]:
        return {
            "completion_key": self.completion_key,
            "completed_at": self.completed_at.isoformat(),
            "state": self.state.name,
            "values": self.values,
            "request_signature": self.request_signature,
        }


@dataclass(frozen=True)
class BatchTrialGeneration:
    """Durable provenance for how one batch trial was generated."""

    batch_id: str
    generator_mode: BatchGeneratorMode
    generator_seed: int | None
    sampler_snapshot_id: str | None

    def to_system_attrs(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "generator_mode": self.generator_mode.value,
            "generator_seed": self.generator_seed,
            "sampler_snapshot_id": self.sampler_snapshot_id,
        }


@dataclass(frozen=True)
class BatchResourceAssignment:
    """Resource metadata assigned by a scheduler independently of sampler scoring."""

    worker_id: str
    assigned_at: datetime
    resource_profile: dict[str, Any]

    def to_system_attrs(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "assigned_at": self.assigned_at.isoformat(),
            "resource_profile": self.resource_profile,
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


def create_batch_trial_generation(
    batch_id: str,
    generator_mode: BatchGeneratorMode,
    generator_seed: int | None,
    sampler_snapshot_id: str | None,
) -> BatchTrialGeneration:
    return BatchTrialGeneration(
        batch_id=batch_id,
        generator_mode=generator_mode,
        generator_seed=generator_seed,
        sampler_snapshot_id=sampler_snapshot_id,
    )


def create_batch_resource_assignment(
    worker_id: str,
    assigned_at: datetime,
    resource_profile: Mapping[str, Any],
) -> BatchResourceAssignment:
    profile = _normalize_resource_profile(dict(resource_profile))
    return BatchResourceAssignment(
        worker_id=worker_id,
        assigned_at=assigned_at,
        resource_profile=profile,
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
    completion_key: str | None = None


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
    completion_key: str | None = None
    completed_at: datetime | None = None
    idempotent_replay: bool = False


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


def normalize_batch_generator_mode(
    generator_mode: BatchGeneratorMode | str | None,
) -> BatchGeneratorMode:
    if generator_mode is None:
        return BatchGeneratorMode.ADAPTIVE
    if isinstance(generator_mode, BatchGeneratorMode):
        return generator_mode
    if isinstance(generator_mode, str):
        try:
            return BatchGeneratorMode(generator_mode)
        except ValueError:
            valid_modes = ", ".join(mode.value for mode in BatchGeneratorMode)
            raise ValueError(
                f"generator_mode must be one of: {valid_modes}."
            ) from None
    raise TypeError("generator_mode must be a BatchGeneratorMode or string when provided.")


def calculate_batch_suggestion_diagnostics(
    params_batch: Sequence[Mapping[str, Any]],
    search_space: Mapping[str, BaseDistribution],
    *,
    near_duplicate_threshold: float = 0.01,
) -> BatchSuggestionDiagnostics | None:
    if not search_space:
        return None

    evaluated_params = [
        params
        for params in params_batch
        if all(param_name in params for param_name in search_space)
    ]
    pair_count = len(evaluated_params) * (len(evaluated_params) - 1) // 2
    if pair_count == 0:
        return BatchSuggestionDiagnostics(
            evaluated_count=len(evaluated_params),
            pair_count=0,
            duplicate_pair_count=0,
            duplicate_rate=0.0,
            near_duplicate_pair_count=0,
            near_duplicate_rate=0.0,
            near_duplicate_threshold=near_duplicate_threshold,
        )

    fingerprints = [
        _fingerprint_params(params, search_space) for params in evaluated_params
    ]
    duplicate_pair_count = 0
    near_duplicate_pair_count = 0

    for i, fingerprint in enumerate(fingerprints[:-1]):
        for j in range(i + 1, len(fingerprints)):
            if fingerprint == fingerprints[j]:
                duplicate_pair_count += 1
                continue
            if _is_near_duplicate(
                evaluated_params[i],
                evaluated_params[j],
                search_space,
                near_duplicate_threshold,
            ):
                near_duplicate_pair_count += 1

    return BatchSuggestionDiagnostics(
        evaluated_count=len(evaluated_params),
        pair_count=pair_count,
        duplicate_pair_count=duplicate_pair_count,
        duplicate_rate=duplicate_pair_count / pair_count,
        near_duplicate_pair_count=near_duplicate_pair_count,
        near_duplicate_rate=near_duplicate_pair_count / pair_count,
        near_duplicate_threshold=near_duplicate_threshold,
    )


def calculate_sampler_snapshot_id(
    sampler: object,
    search_space: Mapping[str, BaseDistribution],
    trials: Sequence[FrozenTrial],
) -> str:
    """Hash the visible sampler inputs used to produce one candidate batch."""

    payload = {
        "sampler_class": f"{sampler.__class__.__module__}.{sampler.__class__.__qualname__}",
        "search_space": [
            (name, json.loads(distribution_to_json(distribution)))
            for name, distribution in sorted(search_space.items())
        ],
        "trials": [
            _snapshot_trial(trial) for trial in sorted(trials, key=lambda trial: trial.number)
        ],
    }
    serialized_payload = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=_json_fallback
    )
    return hashlib.sha256(serialized_payload.encode()).hexdigest()


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


def create_batch_trial_completion(
    completion_key: str,
    completed_at: datetime,
    state: TrialState,
    values: list[float] | None,
    request_signature: str,
) -> BatchTrialCompletion:
    return BatchTrialCompletion(
        completion_key=completion_key,
        completed_at=completed_at,
        state=state,
        values=list(values) if values is not None else None,
        request_signature=request_signature,
    )


def get_batch_trial_completion(
    system_attrs: Mapping[str, Any]
) -> BatchTrialCompletion | None:
    raw_completion = system_attrs.get(_BATCH_TRIAL_COMPLETION_ATTR)
    if not isinstance(raw_completion, Mapping):
        return None

    completion_key = raw_completion.get("completion_key")
    completed_at = raw_completion.get("completed_at")
    state = raw_completion.get("state")
    values = raw_completion.get("values")
    request_signature = raw_completion.get("request_signature")
    if not isinstance(completion_key, str):
        return None
    if not isinstance(completed_at, str):
        return None
    if not isinstance(state, str):
        return None
    if values is not None and not isinstance(values, list):
        return None
    if not isinstance(request_signature, str):
        return None

    try:
        parsed_completed_at = datetime.fromisoformat(completed_at)
        parsed_state = TrialState[state]
    except (KeyError, ValueError):
        return None

    return BatchTrialCompletion(
        completion_key=completion_key,
        completed_at=parsed_completed_at,
        state=parsed_state,
        values=values,
        request_signature=request_signature,
    )


def calculate_batch_completion_request_signature(
    values: float | Sequence[float] | None,
    state: TrialState | None,
) -> str:
    payload = {
        "state": state.name if state is not None else None,
        "values": _normalize_completion_values(values),
    }
    serialized_payload = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), default=_json_fallback
    )
    return hashlib.sha256(serialized_payload.encode()).hexdigest()


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


def _fingerprint_params(
    params: Mapping[str, Any], search_space: Mapping[str, BaseDistribution]
) -> tuple[tuple[str, float], ...]:
    return tuple(
        sorted(
            (
                param_name,
                distribution.to_internal_repr(params[param_name]),
            )
            for param_name, distribution in search_space.items()
        )
    )


def _is_near_duplicate(
    params0: Mapping[str, Any],
    params1: Mapping[str, Any],
    search_space: Mapping[str, BaseDistribution],
    threshold: float,
) -> bool:
    squared_distance = 0.0
    comparable_dimension_count = 0
    for param_name, distribution in search_space.items():
        normalized0 = _normalize_numeric_param(params0[param_name], distribution)
        normalized1 = _normalize_numeric_param(params1[param_name], distribution)
        if normalized0 is None or normalized1 is None:
            continue

        comparable_dimension_count += 1
        squared_distance += (normalized0 - normalized1) ** 2

    if comparable_dimension_count == 0:
        return False

    distance = math.sqrt(squared_distance / comparable_dimension_count)
    return distance <= threshold


def _normalize_numeric_param(value: Any, distribution: BaseDistribution) -> float | None:
    if not isinstance(distribution, (FloatDistribution, IntDistribution)):
        return None

    normalized_value = float(value)
    low = float(distribution.low)
    high = float(distribution.high)
    if distribution.log:
        normalized_value = math.log(normalized_value)
        low = math.log(low)
        high = math.log(high)

    span = high - low
    if span <= 0 or not math.isfinite(span):
        return None

    return (normalized_value - low) / span


def _snapshot_trial(trial: FrozenTrial) -> dict[str, Any]:
    return {
        "number": trial.number,
        "state": trial.state.name,
        "values": trial.values,
        "params": _normalize_json(trial.params),
        "distributions": {
            name: json.loads(distribution_to_json(distribution))
            for name, distribution in sorted(trial.distributions.items())
        },
        "intermediate_values": _normalize_json(trial.intermediate_values),
        "system_attrs": _normalize_json(
            {
                key: value
                for key, value in trial.system_attrs.items()
                if not key.startswith("batch:")
            }
        ),
    }


def _normalize_json(value: Any) -> Any:
    try:
        json.dumps(value, sort_keys=True, default=_json_fallback)
    except TypeError:
        return _json_fallback(value)
    else:
        return value


def _normalize_resource_profile(resource_profile: Mapping[str, Any]) -> dict[str, Any]:
    for key in resource_profile:
        if not isinstance(key, str):
            raise TypeError("resource_profile keys must be strings.")

    serialized_profile = json.dumps(
        resource_profile, sort_keys=True, separators=(",", ":")
    )
    return json.loads(serialized_profile)


def _json_fallback(value: Any) -> str:
    return repr(value)


def _normalize_completion_values(values: float | Sequence[float] | None) -> Any:
    if values is None:
        return None
    if isinstance(values, Sequence) and not isinstance(values, (bytes, str)):
        return [_normalize_completion_value(value) for value in values]
    return [_normalize_completion_value(values)]


def _normalize_completion_value(value: Any) -> Any:
    try:
        float_value = float(value)
    except (TypeError, ValueError):
        return _json_fallback(value)
    if math.isnan(float_value):
        return "nan"
    if math.isinf(float_value):
        return "inf" if float_value > 0 else "-inf"
    return float_value
