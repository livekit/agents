from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Literal

import aiohttp

from livekit import rtc

from ....job import get_job_context
from ....log import logger
from ....utils.http_context import http_session
from ....voice import RunContext
from .config import SMSResponse, SMSToolRequest

SupportedProvider = Literal["twilio", "signalwire", "vonage"]


@dataclass
class ProviderInfo:
    """Provider metadata detected from the environment."""

    name: SupportedProvider
    credentials: dict[str, str]
    default_from_number: str | None


def detect_provider() -> ProviderInfo | None:
    """Auto-detect SMS provider based on environment variables."""
    if os.getenv("TWILIO_ACCOUNT_SID") and os.getenv("TWILIO_AUTH_TOKEN"):
        return ProviderInfo(
            name="twilio",
            credentials={
                "account_sid": os.getenv("TWILIO_ACCOUNT_SID", ""),
                "auth_token": os.getenv("TWILIO_AUTH_TOKEN", ""),
            },
            default_from_number=os.getenv("TWILIO_PHONE_NUMBER"),
        )

    if os.getenv("SIGNALWIRE_PROJECT_ID") and os.getenv("SIGNALWIRE_TOKEN"):
        return ProviderInfo(
            name="signalwire",
            credentials={
                "project_id": os.getenv("SIGNALWIRE_PROJECT_ID", ""),
                "token": os.getenv("SIGNALWIRE_TOKEN", ""),
                "space_url": os.getenv("SIGNALWIRE_SPACE_URL", ""),
            },
            default_from_number=os.getenv("SIGNALWIRE_PHONE_NUMBER"),
        )

    if os.getenv("VONAGE_API_KEY") and os.getenv("VONAGE_API_SECRET"):
        return ProviderInfo(
            name="vonage",
            credentials={
                "api_key": os.getenv("VONAGE_API_KEY", ""),
                "api_secret": os.getenv("VONAGE_API_SECRET", ""),
            },
            default_from_number=os.getenv("VONAGE_PHONE_NUMBER"),
        )

    return None


async def run_sms_request(
    request: SMSToolRequest,
    to_number: str,
    message: str,
) -> SMSResponse:
    """Send SMS through the configured provider."""
    timeout = aiohttp.ClientTimeout(total=request.timeout_ms / 1000.0)
    session = http_session()

    try:
        if request.provider_name == "twilio":
            return await _send_twilio(session, request, to_number, message, timeout)
        if request.provider_name == "signalwire":
            return await _send_signalwire(session, request, to_number, message, timeout)
        if request.provider_name == "vonage":
            return await _send_vonage(session, request, to_number, message, timeout)
        raise ValueError(f"Unsupported provider: {request.provider_name}")
    except asyncio.TimeoutError:
        logger.warning(
            "SMS request timeout",
            extra={"provider": request.provider_name, "to": to_number},
        )
        return SMSResponse(
            success=False,
            message_id=None,
            to=to_number,
            error="Request timeout",
        )
    except aiohttp.ClientError as e:
        logger.warning(
            "SMS request client error",
            extra={"provider": request.provider_name, "to": to_number, "error": str(e)},
        )
        return SMSResponse(
            success=False,
            message_id=None,
            to=to_number,
            error=str(e),
        )
    except Exception as e:
        logger.exception(
            "Unexpected SMS request failure",
            extra={"provider": request.provider_name},
        )
        return SMSResponse(
            success=False,
            message_id=None,
            to=to_number,
            error=str(e),
        )


async def _send_twilio(
    session: aiohttp.ClientSession,
    request: SMSToolRequest,
    to_number: str,
    message: str,
    timeout: aiohttp.ClientTimeout,
) -> SMSResponse:
    credentials = request.credentials
    url = f"https://api.twilio.com/2010-04-01/Accounts/{credentials['account_sid']}/Messages.json"
    auth = aiohttp.BasicAuth(credentials["account_sid"], credentials["auth_token"])
    payload = {"To": to_number, "From": request.from_number, "Body": message}
    async with session.post(url, auth=auth, data=payload, timeout=timeout) as resp:
        if resp.status in (200, 201):
            result = await resp.json()
            return SMSResponse(success=True, message_id=result.get("sid"), to=to_number)
        return SMSResponse(
            success=False,
            message_id=None,
            to=to_number,
            error=f"HTTP {resp.status}: {await resp.text()}",
        )


async def _send_signalwire(
    session: aiohttp.ClientSession,
    request: SMSToolRequest,
    to_number: str,
    message: str,
    timeout: aiohttp.ClientTimeout,
) -> SMSResponse:
    credentials = request.credentials
    space_url = credentials.get("space_url")
    if not space_url:
        return SMSResponse(
            success=False,
            message_id=None,
            to=to_number,
            error="SIGNALWIRE_SPACE_URL not configured",
        )

    account_id = credentials["project_id"]
    url = f"https://{space_url}/api/laml/2010-04-01/Accounts/{account_id}/Messages.json"
    auth = aiohttp.BasicAuth(account_id, credentials["token"])
    payload = {
        "To": to_number if to_number.startswith("+") else f"+{to_number}",
        "From": request.from_number
        if request.from_number.startswith("+")
        else f"+{request.from_number}",
        "Body": message,
    }

    async with session.post(url, auth=auth, data=payload, timeout=timeout) as resp:
        if resp.status in (200, 201):
            result = await resp.json()
            return SMSResponse(success=True, message_id=result.get("sid"), to=to_number)
        return SMSResponse(
            success=False,
            message_id=None,
            to=to_number,
            error=f"HTTP {resp.status}: {await resp.text()}",
        )


async def _send_vonage(
    session: aiohttp.ClientSession,
    request: SMSToolRequest,
    to_number: str,
    message: str,
    timeout: aiohttp.ClientTimeout,
) -> SMSResponse:
    credentials = request.credentials
    url = "https://rest.nexmo.com/sms/json"
    payload = {
        "api_key": credentials["api_key"],
        "api_secret": credentials["api_secret"],
        "to": to_number,
        "from": request.from_number,
        "text": message,
    }

    async with session.post(url, json=payload, timeout=timeout) as resp:
        body_text = await resp.text()
        if resp.status not in (200, 201):
            return SMSResponse(
                success=False,
                message_id=None,
                to=to_number,
                error=f"HTTP {resp.status}: {body_text}",
            )

        try:
            parsed = json.loads(body_text)
        except json.JSONDecodeError:
            return SMSResponse(
                success=False,
                message_id=None,
                to=to_number,
                error="Unable to parse Vonage response",
            )

        messages = parsed.get("messages", [])
        if messages and messages[0].get("status") == "0":
            return SMSResponse(
                success=True,
                message_id=messages[0].get("message-id"),
                to=to_number,
            )

        error_text = messages[0].get("error-text", "Unknown error") if messages else "Unknown error"
        return SMSResponse(success=False, message_id=None, to=to_number, error=error_text)


def get_caller_phone_number(context: RunContext) -> str | None:
    """Extract caller phone number from SIP participant via JobContext."""
    try:
        job_ctx = get_job_context()
        for participant in job_ctx.room.remote_participants.values():
            if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
                # remove sip_ prefix also from the identity
                return participant.identity.replace("sip_", "")
    except Exception:
        return None
    return None


def validate_phone_number(phone: str) -> bool:
    """Validate phone number format (international and local formats supported)."""
    if not phone or not isinstance(phone, str):
        return False
    digits_only = "".join(filter(str.isdigit, phone))
    return 7 <= len(digits_only) <= 15
