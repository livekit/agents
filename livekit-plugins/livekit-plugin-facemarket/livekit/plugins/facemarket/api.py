from __future__ import annotations

import asyncio
import json
import ssl
from typing import Any

import aiohttp
import certifi

from .exceptions import FaceMarketPlatformError
from .log import logger
from .schemas import SessionInfo, StartSessionRequest

BASE_URL = "https://pre.facemarket.ai/vih"
AUTH_PATH = "/dispatcher/auth/session/token"
START_PATH = "/dispatcher/v1/session/start"
STOP_PATH = "/dispatcher/v1/session/stop"
DEFAULT_TIMEOUT = 10.0
DEFAULT_RETRIES = 2


def _redact_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "<redacted>"
            if key.lower().endswith("token")
            else _redact_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_payload(item) for item in value]
    return value


class FaceMarketAPI:
    def __init__(
        self,
        *,
        platform_api_key: str,
        base_url: str = BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_RETRIES,
    ) -> None:
        self._platform_api_key = platform_api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._max_retries = max_retries
        self._auth_token: str | None = None
        self._ssl_context = ssl.create_default_context(cafile=certifi.where())

    async def start_session(self, request: StartSessionRequest) -> SessionInfo:
        request_payload = request.to_payload()
        request_headers = {"Authorization": f"Bearer {self._platform_api_key}"}
        logger.info(
            "FaceMarket start request payload=%s headers=%s",
            _redact_payload(request_payload),
            _redact_payload(request_headers),
        )
        payload = await self._request_json(
            "POST",
            START_PATH,
            headers=request_headers,
            json_body=request_payload,
        )

        data = payload.get("data") or {}
        logger.info("FaceMarket start response payload=%s", _redact_payload(payload))
        if data.get("roomToken") or data.get("livekitUrl"):
            raise FaceMarketPlatformError(
                "FaceMarket start returned hosted-room response, not plugin-mode response. "
                "Expected data.sessionId/session_id and renderer/coordinator joining the provided room."
            )
        session_id = (
            payload.get("sessionId")
            or payload.get("session_id")
            or data.get("sessionId")
            or data.get("session_id")
        )
        if not session_id:
            raise FaceMarketPlatformError(
                "FaceMarket start response did not contain sessionId"
            )

        return SessionInfo(
            session_id=str(session_id),
            room_id=str(data["roomId"]) if data.get("roomId") else None,
            livekit_url=str(data["livekitUrl"]) if data.get("livekitUrl") else None,
            room_token=str(data["roomToken"]) if data.get("roomToken") else None,
            green_screen=data.get("greenScreen"),
        )

    async def stop_session(self, session_id: str) -> None:
        request_body = {"sessionId": session_id, "roomId": session_id}
        request_headers = {"Authorization": f"Bearer {self._platform_api_key}"}
        logger.info(
            "FaceMarket stop request payload=%s headers=%s",
            _redact_payload(request_body),
            _redact_payload(request_headers),
        )
        await self._request_json(
            "POST",
            STOP_PATH,
            headers=request_headers,
            json_body=request_body,
        )

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        attempts = self._max_retries + 1
        last_error: Exception | None = None

        for attempt in range(attempts):
            try:
                connector = aiohttp.TCPConnector(ssl=self._ssl_context)
                async with aiohttp.ClientSession(timeout=self._timeout, connector=connector) as http:
                    async with http.request(
                        method,
                        url,
                        params=params,
                        headers=headers,
                        json=json_body,
                    ) as response:
                        return await self._decode_response(response)
            except FaceMarketPlatformError:
                raise
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                last_error = exc
                if attempt >= attempts - 1:
                    break
                await asyncio.sleep(0.25 * (attempt + 1))

        raise FaceMarketPlatformError(f"FaceMarket API request failed: {last_error!r}")

    async def _decode_response(self, response: aiohttp.ClientResponse) -> dict[str, Any]:
        raw_text = await response.text()
        try:
            payload = json.loads(raw_text) if raw_text else {}
        except json.JSONDecodeError as exc:
            raise FaceMarketPlatformError(
                f"FaceMarket API returned non-JSON response: status={response.status}"
            ) from exc

        if response.status >= 500:
            logger.info(
                "FaceMarket API response status=%s payload=%s",
                response.status,
                _redact_payload(payload),
            )
            raise aiohttp.ClientResponseError(
                request_info=response.request_info,
                history=response.history,
                status=response.status,
                message=str(payload),
                headers=response.headers,
            )

        if response.status >= 400:
            logger.info(
                "FaceMarket API response status=%s payload=%s",
                response.status,
                _redact_payload(payload),
            )
            raise FaceMarketPlatformError(
                f"FaceMarket API request failed: status={response.status}, body={payload!r}"
            )

        code = payload.get("code")
        if code not in (None, 0):
            logger.info(
                "FaceMarket API response status=%s payload=%s",
                response.status,
                _redact_payload(payload),
            )
            message = payload.get("message") or payload.get("msg") or "unknown platform error"
            raise FaceMarketPlatformError(f"FaceMarket API error {code}: {message}")

        return payload
