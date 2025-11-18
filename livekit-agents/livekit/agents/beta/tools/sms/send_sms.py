from __future__ import annotations

import json
from typing import Any

from .... import FunctionTool
from ....llm.tool_context import ToolError, function_tool
from ....log import logger
from ....voice import RunContext
from .config import SMSToolConfig, SMSToolRequest
from .provider_utils import detect_provider, get_caller_phone_number, run_sms_request, validate_phone_number


def create_sms_tool(config: SMSToolConfig | None = None) -> FunctionTool:
    """Create an SMS sending tool with auto-detected provider."""
    cfg = config or SMSToolConfig()

    provider = detect_provider()
    if provider is None:
        raise ValueError(
            "No SMS provider credentials found. "
            "Set environment variables for Twilio (TWILIO_*), SignalWire (SIGNALWIRE_*), "
            "or Vonage (VONAGE_*) to enable SMS sending."
        )
    if not cfg.name:
        raise ValueError("Tool name is required for SMS tool configuration.")

    sender_number = cfg.from_number or provider.default_from_number
    if not sender_number:
        raise ValueError(
            f"Sender phone number not configured. "
            f"Set the {provider.name.upper()}_PHONE_NUMBER environment variable or provide from_number in SMSToolConfig."
        )

    request = SMSToolRequest(
        provider_name=provider.name,
        credentials=provider.credentials,
        from_number=sender_number,
        timeout_ms=cfg.timeout_ms,
    )

    raw_schema = _build_schema(cfg)

    async def sms_handler(raw_arguments: dict[str, Any], context: RunContext) -> str:
        message = raw_arguments.get("message")
        if not message:
            raise ToolError("Message text is required.")

        if cfg.to_number:
            recipient = cfg.to_number
        elif cfg.auto_detect_caller:
            recipient = get_caller_phone_number(context)
        else:
            recipient = raw_arguments.get("to")

        if not isinstance(recipient, str) or not recipient:
            raise ToolError("Could not determine the recipient phone number.")

        if not validate_phone_number(recipient):
            raise ToolError(
                f"Invalid phone number format: {recipient}. Expected 7-15 digits (with optional symbols)."
            )

        if cfg.execution_message:
            try:
                await context.session.generate_reply(
                    instructions=cfg.execution_message, allow_interruptions=False
                )
            except Exception:  # pragma: no cover - best effort announcement
                logger.debug("Failed to deliver SMS execution message", exc_info=True)

        context.disallow_interruptions()

        response = await run_sms_request(request, recipient, message)
        if not response.success:
            raise ToolError(f"Failed to send SMS: {response.error}")

        if cfg.output_normalizer:
            normalized = cfg.output_normalizer(response)
            if normalized is not None:
                return normalized

        return json.dumps(
            {
                "success": True,
                "message_id": response.message_id,
                "to": response.to,
            },
            indent=2,
        )

    return function_tool(
        sms_handler,
        raw_schema=raw_schema,
    )


def _build_schema(config: SMSToolConfig) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "message": {
            "type": "string",
            "description": "The text message to send to the user.",
        }
    }
    required = ["message"]

    if not config.to_number and not config.auto_detect_caller:
        logger.debug(
            "SMS tool exposes `to` parameter because no recipient is configured; "
            "ensure the LLM is trusted before enabling this mode."
        )
        properties["to"] = {
            "type": "string",
            "description": "Recipient phone number (e.g., +1234567890) in E.164 format",
        }
        required.append("to")

    schema: dict[str, Any] = {
        "name": config.name,
        "description": config.description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }

    if not config.description:
        schema.pop("description")

    return schema
