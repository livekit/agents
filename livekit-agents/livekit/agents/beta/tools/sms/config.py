from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class SMSResponse:
    """Structured response from SMS send operation."""

    success: bool
    message_id: str | None
    to: str
    error: str | None = None


@dataclass
class SMSToolRequest:
    """Provider configuration passed to the SMS utility."""

    provider_name: str
    credentials: dict[str, str]
    from_number: str
    timeout_ms: int = 10000


@dataclass
class SMSToolConfig:
    """Configuration for SMS tool customization."""

    name: str = "send_sms"
    description: str | None = None
    to_number: str | None = None
    from_number: str | None = None
    auto_detect_caller: bool = True
    timeout_ms: int = 10000
    execution_message: str | None = None
    output_normalizer: Callable[[SMSResponse], str | None] | None = None
