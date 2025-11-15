import asyncio
import json
import os
from collections.abc import Callable
from dataclasses import dataclass

import aiohttp

from livekit import rtc

from .... import FunctionTool
from ....job import get_job_context
from ....llm.tool_context import ToolError, function_tool


@dataclass
class SMSResponse:
    """Structured response from SMS send operation.

    Attributes:
        success: Whether the SMS was sent successfully
        message_id: Provider's message ID (if available)
        to: Recipient phone number
        error: Error message (if failed)
    """

    success: bool
    message_id: str | None
    to: str
    error: str | None = None


@dataclass
class SMSToolConfig:
    """Configuration for SMS tool customization.

    Attributes:
        name: Custom tool name (default: "send_sms")
        description: Custom tool description
        to_number: Recipient phone number (if set, disables auto_detect_caller)
        from_number: Sender phone number (if not set, uses PROVIDER_PHONE_NUMBER env var)
        auto_detect_caller: Auto-detect caller phone from SIP participant (default: True)
        output_normalizer: Optional function to transform SMSResponse to string
    """

    name: str | None = None
    description: str | None = None
    to_number: str | None = None
    from_number: str | None = None
    auto_detect_caller: bool = True
    output_normalizer: Callable[[SMSResponse], str] | None = None


def _detect_provider() -> tuple[str, dict] | None:
    """Auto-detect SMS provider based on environment variables.

    Returns:
        Tuple of (provider_name, credentials) or None if no provider configured
    """
    if os.getenv("TWILIO_ACCOUNT_SID") and os.getenv("TWILIO_AUTH_TOKEN"):
        return (
            "twilio",
            {
                "account_sid": os.getenv("TWILIO_ACCOUNT_SID"),
                "auth_token": os.getenv("TWILIO_AUTH_TOKEN"),
                "from": os.getenv("TWILIO_PHONE_NUMBER"),
            },
        )

    if os.getenv("SIGNALWIRE_PROJECT_ID") and os.getenv("SIGNALWIRE_TOKEN"):
        return (
            "signalwire",
            {
                "project_id": os.getenv("SIGNALWIRE_PROJECT_ID"),
                "token": os.getenv("SIGNALWIRE_TOKEN"),
                "space_url": os.getenv("SIGNALWIRE_SPACE_URL"),
                "from": os.getenv("SIGNALWIRE_PHONE_NUMBER"),
            },
        )

    if os.getenv("VONAGE_API_KEY") and os.getenv("VONAGE_API_SECRET"):
        return (
            "vonage",
            {
                "api_key": os.getenv("VONAGE_API_KEY"),
                "api_secret": os.getenv("VONAGE_API_SECRET"),
                "from": os.getenv("VONAGE_PHONE_NUMBER"),
            },
        )

    return None


async def _get_caller_phone_number() -> str | None:
    """Extract caller phone number from SIP participant.

    Returns:
        Phone number from SIP participant identity or None if not available
    """
    try:
        job_ctx = get_job_context()
        for participant in job_ctx.room.remote_participants.values():
            if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
                return participant.identity
    except Exception:
        return None


def _validate_phone_number(phone: str) -> bool:
    """Validate phone number format (international and local formats supported).

    Args:
        phone: Phone number to validate

    Returns:
        True if valid phone number (7-15 digits after cleaning)
    """
    if not phone or not isinstance(phone, str):
        return False

    digits_only = "".join(c for c in phone if c.isdigit())
    return 7 <= len(digits_only) <= 15


async def _send_sms_http(
    provider_name: str,
    url: str,
    to_number: str,
    message: str,
    from_number: str,
    *,
    auth: aiohttp.BasicAuth | None = None,
    json_body: dict | None = None,
    form_body: dict | None = None,
    parse_response: Callable[[dict, str], SMSResponse],
) -> SMSResponse:
    """Generic HTTP SMS sender with unified error handling."""
    async with aiohttp.ClientSession() as session:
        try:
            kwargs = {"timeout": aiohttp.ClientTimeout(total=10)}
            if auth:
                kwargs["auth"] = auth
            if json_body:
                kwargs["json"] = json_body
            if form_body:
                kwargs["data"] = form_body

            async with session.post(url, **kwargs) as resp:
                if resp.status in (200, 201):
                    result = await resp.json()
                    return parse_response(result, to_number)
                else:
                    error_text = await resp.text()
                    return SMSResponse(
                        success=False,
                        message_id=None,
                        to=to_number,
                        error=f"HTTP {resp.status}: {error_text}",
                    )
        except asyncio.TimeoutError:
            return SMSResponse(
                success=False, message_id=None, to=to_number, error="Request timeout"
            )
        except Exception as e:
            return SMSResponse(success=False, message_id=None, to=to_number, error=str(e))


async def _send_twilio_sms(
    credentials: dict, to_number: str, message: str, from_number: str
) -> SMSResponse:
    """Send SMS via Twilio API."""
    url = f"https://api.twilio.com/2010-04-01/Accounts/{credentials['account_sid']}/Messages.json"
    auth = aiohttp.BasicAuth(credentials["account_sid"], credentials["auth_token"])
    form_body = {"To": to_number, "From": from_number, "Body": message}

    def parse(result: dict, to: str) -> SMSResponse:
        return SMSResponse(success=True, message_id=result.get("sid"), to=to)

    return await _send_sms_http(
        "Twilio",
        url,
        to_number,
        message,
        from_number,
        auth=auth,
        form_body=form_body,
        parse_response=parse,
    )


async def _send_signalwire_sms(
    credentials: dict, to_number: str, message: str, from_number: str
) -> SMSResponse:
    """Send SMS via SignalWire API."""
    space_url = credentials.get("space_url")
    if not space_url:
        return SMSResponse(
            success=False,
            message_id=None,
            to=to_number,
            error="SIGNALWIRE_SPACE_URL not configured",
        )

    url = f"https://{space_url}/api/laml/2010-04-01/Accounts/{credentials['project_id']}/Messages.json"
    auth = aiohttp.BasicAuth(credentials["project_id"], credentials["token"])
    # SignalWire requires + prefix
    form_body = {
        "To": to_number if to_number.startswith("+") else f"+{to_number}",
        "From": from_number if from_number.startswith("+") else f"+{from_number}",
        "Body": message,
    }

    def parse(result: dict, to: str) -> SMSResponse:
        return SMSResponse(success=True, message_id=result.get("sid"), to=to)

    return await _send_sms_http(
        "SignalWire",
        url,
        to_number,
        message,
        from_number,
        auth=auth,
        form_body=form_body,
        parse_response=parse,
    )


async def _send_vonage_sms(
    credentials: dict, to_number: str, message: str, from_number: str
) -> SMSResponse:
    """Send SMS via Vonage (Nexmo) API."""
    url = "https://rest.nexmo.com/sms/json"
    json_body = {
        "api_key": credentials["api_key"],
        "api_secret": credentials["api_secret"],
        "to": to_number,
        "from": from_number,
        "text": message,
    }

    def parse(result: dict, to: str) -> SMSResponse:
        messages = result.get("messages", [])
        if messages and messages[0].get("status") == "0":
            return SMSResponse(success=True, message_id=messages[0].get("message-id"), to=to)
        error = messages[0].get("error-text", "Unknown error") if messages else "Unknown error"
        return SMSResponse(success=False, message_id=None, to=to, error=error)

    return await _send_sms_http(
        "Vonage", url, to_number, message, from_number, json_body=json_body, parse_response=parse
    )


def create_sms_tool(config: SMSToolConfig | None = None) -> FunctionTool:
    """Create an SMS sending tool with auto-detected provider.

    The tool automatically detects which SMS provider to use based on
    environment variables. Supports Twilio, SignalWire, and Vonage.

    Environment Variables:
        Twilio:
            - TWILIO_ACCOUNT_SID
            - TWILIO_AUTH_TOKEN
            - TWILIO_PHONE_NUMBER (optional, can be set in config)

        SignalWire:
            - SIGNALWIRE_PROJECT_ID
            - SIGNALWIRE_TOKEN
            - SIGNALWIRE_SPACE_URL
            - SIGNALWIRE_PHONE_NUMBER (optional, can be set in config)

        Vonage:
            - VONAGE_API_KEY
            - VONAGE_API_SECRET
            - VONAGE_PHONE_NUMBER (optional, can be set in config)

    Args:
        config: Optional configuration to customize tool behavior

    Returns:
        FunctionTool that can be used in Agent.tools

    Example:
        ```python
        from livekit.agents.beta.tools.sms import create_sms_tool, SMSToolConfig

        # Basic usage - auto-detects provider
        sms_tool = create_sms_tool()

        # With customization
        config = SMSToolConfig(
            name="notify_customer",
            description="Send confirmation SMS to customer",
            auto_detect_caller=True,
        )
        sms_tool = create_sms_tool(config)

        # Use in agent
        agent = Agent(
            instructions="You are a helpful assistant",
            tools=[sms_tool]
        )
        ```
    """
    config = config or SMSToolConfig()

    provider_info = _detect_provider()
    if not provider_info:
        raise ValueError(
            "No SMS provider credentials found in environment variables. "
            "Set TWILIO_*, SIGNALWIRE_*, or VONAGE_* env vars to enable SMS. "
            "Required variables:\n"
            "  Twilio: TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN\n"
            "  SignalWire: SIGNALWIRE_PROJECT_ID, SIGNALWIRE_TOKEN, SIGNALWIRE_SPACE_URL\n"
            "  Vonage: VONAGE_API_KEY, VONAGE_API_SECRET"
        )

    provider_name, credentials = provider_info

    tool_name = config.name or "send_sms"
    tool_description = (
        config.description
        or f"Send SMS message via {provider_name}. Automatically sends to the caller if no recipient specified."
    )

    parameters_properties = {
        "message": {"type": "string", "description": "The text message to send"}
    }

    required_params = ["message"]

    # Add 'to' parameter only if we can't determine recipient automatically
    if not config.to_number and not config.auto_detect_caller:
        parameters_properties["to"] = {
            "type": "string",
            "description": "Recipient phone number (international or local format, e.g., +1234567890 or 1234567890)",
        }
        required_params.append("to")

    raw_schema = {
        "name": tool_name,
        "description": tool_description,
        "parameters": {
            "type": "object",
            "properties": parameters_properties,
            "required": required_params,
        },
    }

    # Create the actual SMS sending function with raw_arguments
    async def send_sms(raw_arguments: dict) -> str:
        """Send SMS message via configured provider."""
        message = raw_arguments.get("message")

        if not message:
            raise ToolError("Message text is required")

        if config.to_number:
            recipient = config.to_number
        elif config.auto_detect_caller:
            recipient = await _get_caller_phone_number()
            if not recipient:
                raise ToolError("Could not auto-detect caller phone number from SIP participant")
        else:
            recipient = raw_arguments.get("to")
            if not recipient:
                raise ToolError("Recipient phone number is required")

        # Validate phone number
        if not _validate_phone_number(recipient):
            raise ToolError(
                f"Invalid phone number format: {recipient}. Phone number must contain 7-15 digits"
            )

        sender_number = config.from_number or credentials.get("from")
        if not sender_number:
            raise ToolError(
                f"Sender phone number not configured. Set {provider_name.upper()}_PHONE_NUMBER "
                "environment variable or provide from_number in config"
            )

        try:
            if provider_name == "twilio":
                response = await _send_twilio_sms(credentials, recipient, message, sender_number)
            elif provider_name == "signalwire":
                response = await _send_signalwire_sms(
                    credentials, recipient, message, sender_number
                )
            elif provider_name == "vonage":
                response = await _send_vonage_sms(credentials, recipient, message, sender_number)
            else:
                raise ToolError(f"Unknown provider: {provider_name}")

            if config.output_normalizer:
                return config.output_normalizer(response)

            if response.success:
                return json.dumps(
                    {
                        "success": True,
                        "message_id": response.message_id,
                        "to": recipient,
                    },
                    indent=2,
                )
            else:
                raise ToolError(f"Failed to send SMS: {response.error}")

        except ToolError:
            raise
        except Exception as e:
            raise ToolError(f"Failed to send SMS: {str(e)}") from e

    return function_tool(
        send_sms,
        raw_schema=raw_schema,
    )
