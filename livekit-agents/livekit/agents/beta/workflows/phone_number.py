from __future__ import annotations

import re
from dataclasses import dataclass

from ...llm.tool_context import ToolError

PHONE_REGEX = r"^\+?[1-9]\d{6,14}$"


@dataclass
class GetPhoneNumberResult:
    phone_number: str


def normalize_phone_number(phone_number: str) -> str:
    """Normalize punctuation and validate a phone number."""
    cleaned = re.sub(r"[\s\-().]+", "", phone_number.strip())
    if not re.match(PHONE_REGEX, cleaned):
        raise ToolError(f"Invalid phone number provided: {phone_number}")
    return cleaned
