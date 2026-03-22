from __future__ import annotations

import asyncio
import json
import os
import urllib.error
import urllib.request
import warnings
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from ._payload import CallIdResolver
from ._plugin import (
    CaptureConfigInput,
    HammingRuntime,
    LiveKitConfigInput,
    RecordingConfigInput,
    RecordingContext,
    RedactConfigInput,
    StreamingConfigInput,
    attach_session as _attach_session,
    build_config,
    configure_runtime,
    get_runtime,
)
from .version import __version__

if TYPE_CHECKING:
    from livekit.agents.job import JobContext
    from livekit.agents.voice.agent_session import AgentSession


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    ok: bool
    message: str


@dataclass(frozen=True)
class DoctorReport:
    ok: bool
    endpoint: str
    plugin_version: str
    checks: list[DoctorCheck]

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "endpoint": self.endpoint,
            "plugin_version": self.plugin_version,
            "checks": [asdict(check) for check in self.checks],
        }


def configure_hamming(
    api_key: str | None = None,
    *,
    base_url: str | None = None,
    call_review_enabled: bool = True,
    flush_on_shutdown: bool = True,
    auto_record_audio: bool = False,
    external_agent_id: str | None = None,
    payload_schema_version: str = "2026-03-02",
    call_id_strategy: str = "room_name",
    call_id_metadata_key: str = "call_id",
    resolve_call_id: CallIdResolver | None = None,
    recording_mode: str | None = None,
    sampling_rate: float = 1.0,
    sampling_key: str = "participant_identity",
    capture: CaptureConfigInput | None = None,
    redact: RedactConfigInput | None = None,
    livekit: LiveKitConfigInput | None = None,
    livekit_url: str | None = None,
    livekit_api_key: str | None = None,
    livekit_api_secret: str | None = None,
    include_interim_transcripts: bool | None = None,
    request_timeout_seconds: float = 10.0,
    max_retries: int = 3,
    retry_backoff_seconds: float = 0.5,
    recording: RecordingConfigInput | None = None,
    streaming: StreamingConfigInput | None = None,
) -> HammingRuntime:
    """Configure Hamming monitoring export for LiveKit AgentSession.

    After configuring, attach to each AgentSession via ``hamming.attach_session(session)``.
    """
    config = build_config(
        api_key=api_key,
        base_url=base_url,
        call_review_enabled=call_review_enabled,
        flush_on_shutdown=flush_on_shutdown,
        auto_record_audio=auto_record_audio,
        external_agent_id=external_agent_id,
        payload_schema_version=payload_schema_version,
        call_id_strategy=call_id_strategy,
        call_id_metadata_key=call_id_metadata_key,
        resolve_call_id=resolve_call_id,
        recording_mode=recording_mode,
        sampling_rate=sampling_rate,
        sampling_key=sampling_key,
        capture=capture,
        redact=redact,
        livekit=livekit,
        livekit_url=livekit_url,
        livekit_api_key=livekit_api_key,
        livekit_api_secret=livekit_api_secret,
        include_interim_transcripts=include_interim_transcripts,
        request_timeout_seconds=request_timeout_seconds,
        max_retries=max_retries,
        retry_backoff_seconds=retry_backoff_seconds,
        recording=recording,
        streaming=streaming,
    )
    return configure_runtime(config)


def attach_session(
    session: AgentSession[Any],
    *,
    job_ctx: JobContext | None = None,
    participant_identity: str | None = None,
    participant_metadata: str | None = None,
    external_agent_id: str | None = None,
    recording_context: RecordingContext | None = None,
) -> None:
    """Attach hamming monitoring to a specific AgentSession.

    This mirrors explicit plugin lifecycle patterns used by other plugins.
    """
    _attach_session(
        session,
        job_ctx=job_ctx,
        participant_identity=participant_identity,
        participant_metadata=participant_metadata,
        external_agent_id=external_agent_id,
        recording_context=recording_context,
    )


def doctor(
    api_key: str | None = None,
    *,
    base_url: str | None = None,
    timeout_seconds: float = 5.0,
) -> DoctorReport:
    """Run preflight checks for Hamming export integration."""
    checks: list[DoctorCheck] = []

    runtime = get_runtime()
    runtime_api_key = runtime.config.api_key if runtime is not None else None
    runtime_base_url = runtime.config.base_url if runtime is not None else None

    resolved_api_key = api_key or runtime_api_key
    resolved_base_url = (base_url or runtime_base_url or "https://app.hamming.ai").rstrip("/")
    endpoint = f"{resolved_base_url}/api/rest/v2/livekit-monitoring"

    checks.append(
        DoctorCheck(
            name="api_key_present",
            ok=bool(resolved_api_key),
            message="api key configured" if resolved_api_key else "missing API key",
        )
    )

    runtime_prereq_check = _build_runtime_prereq_check(runtime)
    if runtime_prereq_check is not None:
        checks.append(runtime_prereq_check)

    reachable = False
    reachability_message = "not checked"

    if resolved_api_key:
        if _running_in_event_loop():
            reachable = False
            reachability_message = (
                "skipped endpoint probe in a running event loop to avoid blocking; run doctor() "
                "outside async context for network validation"
            )
            warnings.warn(
                "doctor() was called in a running event loop. Endpoint reachability was skipped.",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            reachable, reachability_message = _check_endpoint_reachability(
                endpoint=endpoint,
                api_key=resolved_api_key,
                timeout_seconds=timeout_seconds,
            )

    checks.append(
        DoctorCheck(
            name="endpoint_reachable",
            ok=reachable,
            message=reachability_message,
        )
    )

    ok = all(check.ok for check in checks)
    return DoctorReport(ok=ok, endpoint=endpoint, plugin_version=__version__, checks=checks)


def doctor_json(
    api_key: str | None = None,
    *,
    base_url: str | None = None,
    timeout_seconds: float = 5.0,
) -> str:
    report = doctor(api_key=api_key, base_url=base_url, timeout_seconds=timeout_seconds)
    return json.dumps(report.to_dict(), indent=2)


def _running_in_event_loop() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def _check_endpoint_reachability(
    *,
    endpoint: str,
    api_key: str,
    timeout_seconds: float,
) -> tuple[bool, str]:
    request = urllib.request.Request(
        endpoint,
        method="OPTIONS",
        headers={
            "Authorization": f"Bearer {api_key}",
            "X-Workspace-Key": api_key,
            "X-API-Key": api_key,
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
            status = response.status
            return status < 500, f"endpoint responded with status={status}"
    except urllib.error.HTTPError as exc:
        return exc.code < 500, f"endpoint returned status={exc.code}"
    except urllib.error.URLError as exc:
        return False, f"endpoint unreachable: {exc.reason}"


def _build_runtime_prereq_check(runtime: HammingRuntime | None) -> DoctorCheck | None:
    if runtime is None:
        return None

    recording = runtime.config.recording
    if recording.mode != "room_composite" or recording.source != "auto_egress":
        return None

    required_envs = (
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "AWS_RECORDINGS_BUCKET",
        "AWS_REGION",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    )
    missing_envs = [name for name in required_envs if not _env_present(name)]
    if not missing_envs:
        return DoctorCheck(
            name="auto_egress_runtime_env",
            ok=True,
            message="auto_egress runtime environment is configured",
        )

    return DoctorCheck(
        name="auto_egress_runtime_env",
        ok=False,
        message=f"missing runtime env: {', '.join(missing_envs)}",
    )


def _env_present(name: str) -> bool:
    value = os.getenv(name)
    return isinstance(value, str) and bool(value.strip())
