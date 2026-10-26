from __future__ import annotations

import re
from dataclasses import dataclass

from ...llm.tool_context import ToolError


@dataclass
class GetEmailResult:
    email_address: str


EMAIL_REGEX = (
    r"^[A-Za-z0-9][A-Za-z0-9._%+\-]*@(?:[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}$"
)


def normalize_email_address(email: str) -> str:
    """Trim and validate an email address."""
    email = email.strip()
    if not re.match(EMAIL_REGEX, email):
        raise ToolError(f"Invalid email address provided: {email}")
    return email
