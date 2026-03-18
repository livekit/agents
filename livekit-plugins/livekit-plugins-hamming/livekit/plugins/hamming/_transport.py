from __future__ import annotations

import asyncio
import os
from base64 import b64encode
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from urllib.parse import quote

import aiohttp

from livekit import api as livekit_api
from livekit.protocol import egress as proto_egress

from .log import logger


class _RetryableResponseError(RuntimeError):
    pass


_MAX_INLINE_RECORDING_BYTES = 20 * 1024 * 1024  # 20 MB raw audio
_ROOM_COMPOSITE_OPUS_SAMPLE_RATE_HZ = 48_000


@dataclass(frozen=True)
class ConnectionPolicy:
    timeout_seconds: float = 10.0
    max_retries: int = 3
    retry_backoff_seconds: float = 0.5


class HammingTransport:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        policy: ConnectionPolicy,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._policy = policy
        self._session: aiohttp.ClientSession | None = None

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "X-Workspace-Key": self._api_key,
            "X-API-Key": self._api_key,
        }

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def build_inline_recording_capture(
        self,
        *,
        recording_path: str | Path,
    ) -> dict[str, Any] | None:
        path = Path(recording_path)
        exists = await asyncio.to_thread(lambda: path.exists() and path.is_file())
        if not exists:
            return None

        try:
            stat = await asyncio.to_thread(path.stat)
        except OSError:
            return None
        if stat.st_size <= 0:
            return None
        if stat.st_size > _MAX_INLINE_RECORDING_BYTES:
            return None

        content_type = _guess_audio_content_type(path)
        file_name = path.name or "recording.ogg"
        try:
            audio_bytes = await asyncio.to_thread(path.read_bytes)
        except OSError:
            return None
        return {
            "file_name": file_name,
            "content_type": content_type,
            "content_base64": b64encode(audio_bytes).decode("ascii"),
        }

    async def send_capture(self, envelope: dict[str, Any]) -> None:
        session = await self._ensure_session()
        url = f"{self._base_url}/api/rest/v2/livekit-monitoring"
        timeout = aiohttp.ClientTimeout(total=self._policy.timeout_seconds)

        backoff = self._policy.retry_backoff_seconds
        for attempt in range(self._policy.max_retries + 1):
            try:
                async with session.post(
                    url,
                    json=envelope,
                    timeout=timeout,
                    headers={
                        **self._auth_headers(),
                        "Content-Type": "application/json",
                    },
                ) as resp:
                    body = (await resp.text())[:300]
                    if 200 <= resp.status < 300:
                        logger.info(
                            "hamming monitoring request accepted",
                            extra={
                                "endpoint": url,
                                "status": resp.status,
                                "attempt": attempt + 1,
                            },
                        )
                        return

                    if resp.status == 429 or resp.status >= 500:
                        raise _RetryableResponseError(
                            f"retryable response status={resp.status} body={body}"
                        )

                    raise ValueError(
                        f"hamming monitoring request rejected status={resp.status} body={body}"
                    )
            except (aiohttp.ClientError, asyncio.TimeoutError, _RetryableResponseError):
                if attempt >= self._policy.max_retries:
                    raise
                await asyncio.sleep(backoff)
                backoff *= 2

    async def fetch_test_case_run_recording_url(
        self,
        *,
        test_case_run_id: str,
    ) -> str | None:
        session = await self._ensure_session()
        url = f"{self._base_url}/api/rest/test-case-runs/{quote(test_case_run_id, safe='')}/details"
        timeout = aiohttp.ClientTimeout(total=self._policy.timeout_seconds)

        backoff = self._policy.retry_backoff_seconds
        for attempt in range(self._policy.max_retries + 1):
            try:
                async with session.get(
                    url,
                    timeout=timeout,
                    headers=self._auth_headers(),
                ) as resp:
                    if resp.status == 404:
                        return None

                    if resp.status == 429 or resp.status >= 500:
                        body = (await resp.text())[:300]
                        raise _RetryableResponseError(
                            f"retryable response status={resp.status} body={body}"
                        )

                    if resp.status >= 400:
                        return None

                    payload = await resp.json(content_type=None)
                    if not isinstance(payload, dict):
                        return None

                    recording_url = payload.get("recordingUrl")
                    if isinstance(recording_url, str) and recording_url.strip():
                        return recording_url.strip()

                    return None
            except (aiohttp.ClientError, asyncio.TimeoutError, _RetryableResponseError):
                if attempt >= self._policy.max_retries:
                    raise
                await asyncio.sleep(backoff)
                backoff *= 2

        return None

    async def start_plugin_managed_room_composite_egress(
        self,
        *,
        room_name: str,
        filepath: str,
    ) -> str:
        _assert_plugin_managed_room_composite_environment()
        logger.info(
            "starting plugin-managed room composite egress",
            extra={
                "room_name": room_name,
                "filepath": filepath,
                "bucket": os.getenv("AWS_RECORDINGS_BUCKET", "").strip() or None,
                "region": os.getenv("AWS_REGION", "").strip() or None,
            },
        )
        async with livekit_api.LiveKitAPI() as lkapi:
            output = livekit_api.EncodedFileOutput(
                filepath=filepath,
                s3=_build_s3_upload_config(),
                file_type=_resolve_encoded_file_type(filepath),
            )
            request = livekit_api.RoomCompositeEgressRequest(
                room_name=room_name,
                audio_only=True,
                audio_mixing=livekit_api.AudioMixing.DUAL_CHANNEL_AGENT,
                file_outputs=[output],
            )

            if filepath.lower().endswith(".ogg"):
                request.advanced.CopyFrom(
                    livekit_api.EncodingOptions(
                        audio_codec=livekit_api.AudioCodec.OPUS,
                        audio_frequency=_ROOM_COMPOSITE_OPUS_SAMPLE_RATE_HZ,
                    )
                )

            response = await lkapi.egress.start_room_composite_egress(request)
            egress_id = (response.egress_id or "").strip()
            if not egress_id:
                raise RuntimeError(f"LiveKit returned an empty egress_id for room={room_name}")
            logger.info(
                "started plugin-managed room composite egress",
                extra={
                    "room_name": room_name,
                    "filepath": filepath,
                    "egress_id": egress_id,
                },
            )
            return egress_id

    async def stop_plugin_managed_room_composite_egress_and_wait_for_url(
        self,
        *,
        egress_id: str,
        filepath: str,
        max_attempts: int,
        poll_interval_seconds: float,
    ) -> str | None:
        async with livekit_api.LiveKitAPI() as lkapi:
            self._log_plugin_managed_egress_finalization(
                egress_id=egress_id,
                filepath=filepath,
                max_attempts=max_attempts,
                poll_interval_seconds=poll_interval_seconds,
            )
            await self._request_plugin_managed_egress_stop(
                lkapi=lkapi,
                egress_id=egress_id,
                filepath=filepath,
            )
            last_status_name: str | None = None
            for attempt in range(1, max_attempts + 1):
                (
                    resolved_url,
                    last_status_name,
                    should_continue,
                ) = await self._poll_plugin_managed_egress(
                    lkapi=lkapi,
                    egress_id=egress_id,
                    filepath=filepath,
                    attempt=attempt,
                    max_attempts=max_attempts,
                    last_status_name=last_status_name,
                )
                if resolved_url is not None or not should_continue:
                    return resolved_url
                if attempt < max_attempts:
                    await asyncio.sleep(poll_interval_seconds)

        logger.warning(
            "plugin-managed room composite egress polling exhausted without recording URL",
            extra={"egress_id": egress_id, "filepath": filepath, "attempts": max_attempts},
        )
        return None

    def _log_plugin_managed_egress_finalization(
        self,
        *,
        egress_id: str,
        filepath: str,
        max_attempts: int,
        poll_interval_seconds: float,
    ) -> None:
        logger.info(
            "finalizing plugin-managed room composite egress",
            extra={
                "egress_id": egress_id,
                "filepath": filepath,
                "max_attempts": max_attempts,
                "poll_interval_seconds": poll_interval_seconds,
            },
        )

    async def _request_plugin_managed_egress_stop(
        self,
        *,
        lkapi: livekit_api.LiveKitAPI,
        egress_id: str,
        filepath: str,
    ) -> None:
        try:
            await lkapi.egress.stop_egress(livekit_api.StopEgressRequest(egress_id=egress_id))
            logger.info(
                "requested plugin-managed room composite egress stop",
                extra={"egress_id": egress_id, "filepath": filepath},
            )
        except Exception:
            logger.info(
                "plugin-managed room composite egress stop request was not accepted; polling final state",
                extra={"egress_id": egress_id, "filepath": filepath},
            )

    async def _poll_plugin_managed_egress(
        self,
        *,
        lkapi: livekit_api.LiveKitAPI,
        egress_id: str,
        filepath: str,
        attempt: int,
        max_attempts: int,
        last_status_name: str | None,
    ) -> tuple[str | None, str | None, bool]:
        response = await lkapi.egress.list_egress(
            livekit_api.ListEgressRequest(egress_id=egress_id)
        )
        egress_info = next(iter(response.items), None)
        if egress_info is None:
            result = _handle_missing_plugin_managed_egress(
                egress_id=egress_id,
                filepath=filepath,
                attempt=attempt,
                max_attempts=max_attempts,
            )
            return result, last_status_name, result is None

        resolved_location = _extract_egress_location(egress_info)
        if resolved_location:
            logger.info(
                "resolved plugin-managed room composite egress location",
                extra={
                    "egress_id": egress_id,
                    "filepath": filepath,
                    "attempt": attempt,
                    "resolved_location": resolved_location,
                },
            )
            return resolved_location, last_status_name, False

        status_name = _resolve_egress_status_name(getattr(egress_info, "status", None))
        if status_name != last_status_name:
            logger.info(
                "plugin-managed room composite egress status update",
                extra={
                    "egress_id": egress_id,
                    "filepath": filepath,
                    "attempt": attempt,
                    "status": status_name,
                },
            )

        resolved_url, should_continue = _resolve_plugin_managed_egress_terminal_result(
            egress_id=egress_id,
            filepath=filepath,
            attempt=attempt,
            status_name=status_name,
        )
        return resolved_url, status_name, should_continue

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session


def _guess_audio_content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".wav":
        return "audio/wav"
    if suffix in {".mp3", ".mpeg"}:
        return "audio/mpeg"
    if suffix == ".m4a":
        return "audio/m4a"
    if suffix == ".mp4":
        return "audio/mp4"
    if suffix == ".flac":
        return "audio/flac"
    if suffix == ".ogg":
        return "audio/ogg"
    return "audio/ogg"


def _build_s3_upload_config() -> livekit_api.S3Upload:
    bucket = _require_env("AWS_RECORDINGS_BUCKET")
    region = _require_env("AWS_REGION")
    access_key = _require_env("AWS_ACCESS_KEY_ID")
    secret_key = _require_env("AWS_SECRET_ACCESS_KEY")

    return livekit_api.S3Upload(
        access_key=access_key,
        secret=secret_key,
        region=region,
        bucket=bucket,
    )


def _resolve_encoded_file_type(filepath: str) -> livekit_api.EncodedFileType:
    if filepath.lower().endswith(".ogg"):
        encoded_file_type = getattr(livekit_api.EncodedFileType, "OGG", None)
        if encoded_file_type is not None:
            return cast(livekit_api.EncodedFileType, encoded_file_type)
        raise RuntimeError("LiveKit EncodedFileType.OGG is unavailable on this livekit-api version")

    encoded_file_type = getattr(livekit_api.EncodedFileType, "MP4", None)
    if encoded_file_type is not None:
        return cast(livekit_api.EncodedFileType, encoded_file_type)

    raise RuntimeError("LiveKit EncodedFileType.MP4 is unavailable")


def _handle_missing_plugin_managed_egress(
    *,
    egress_id: str,
    filepath: str,
    attempt: int,
    max_attempts: int,
) -> str | None:
    if attempt in {1, max_attempts}:
        logger.info(
            "plugin-managed room composite egress not visible in list response yet",
            extra={
                "egress_id": egress_id,
                "filepath": filepath,
                "attempt": attempt,
                "max_attempts": max_attempts,
            },
        )
    if attempt < max_attempts:
        return None

    logger.warning(
        "plugin-managed room composite egress never appeared in list response; falling back to expected S3 URL",
        extra={
            "egress_id": egress_id,
            "filepath": filepath,
            "attempts": max_attempts,
        },
    )
    return _build_public_s3_url(filepath)


def _resolve_plugin_managed_egress_terminal_result(
    *,
    egress_id: str,
    filepath: str,
    attempt: int,
    status_name: str | None,
) -> tuple[str | None, bool]:
    if status_name == "EGRESS_COMPLETE":
        logger.info(
            "plugin-managed room composite egress completed without explicit location; using expected S3 URL",
            extra={"egress_id": egress_id, "filepath": filepath, "attempt": attempt},
        )
        return _build_public_s3_url(filepath), False

    if status_name in {"EGRESS_FAILED", "EGRESS_ABORTED", "EGRESS_LIMIT_REACHED"}:
        logger.warning(
            "plugin-managed room composite egress ended without usable recording",
            extra={"egress_id": egress_id, "filepath": filepath, "status": status_name},
        )
        return None, False

    return None, True


def _extract_egress_location(egress_info: Any) -> str | None:
    file_results = getattr(egress_info, "file_results", None) or []
    for result in file_results:
        location = getattr(result, "location", None)
        if isinstance(location, str) and location.strip():
            return location.strip()

    file_info = getattr(egress_info, "file", None)
    location = getattr(file_info, "location", None) if file_info is not None else None
    if isinstance(location, str) and location.strip():
        return location.strip()

    return None


def _resolve_egress_status_name(status: Any) -> str | None:
    if isinstance(status, str) and status.strip():
        return status.strip()

    try:
        numeric_status = int(status)
    except (TypeError, ValueError):
        return None

    for name, value in proto_egress.EgressStatus.items():
        if int(value) == numeric_status:
            return str(name)
    return None


def _build_public_s3_url(filepath: str) -> str:
    key = filepath.lstrip("/")
    bucket = _require_env("AWS_RECORDINGS_BUCKET")
    region = _require_env("AWS_REGION")
    return f"https://{bucket}.s3.{region}.amazonaws.com/{key}"


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    raise RuntimeError(f"{name} is required for plugin-managed LiveKit room composite egress")


def _assert_plugin_managed_room_composite_environment() -> None:
    required_env_names = (
        "LIVEKIT_URL",
        "LIVEKIT_API_KEY",
        "LIVEKIT_API_SECRET",
        "AWS_RECORDINGS_BUCKET",
        "AWS_REGION",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
    )
    missing_env_names = [name for name in required_env_names if not os.getenv(name, "").strip()]
    if missing_env_names:
        joined_names = ", ".join(missing_env_names)
        raise RuntimeError(
            "Missing environment variables for plugin-managed LiveKit room composite egress: "
            f"{joined_names}"
        )
