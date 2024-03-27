from typing import Dict, Optional


class APIError(Exception):
    """
    Base exception for all API errors.
    """

    _raw_error: Dict[str, str]
    message: Optional[str]
    """The error message."""
    code: Optional[str]
    """The error code."""
    hint: Optional[str]
    """The error hint."""
    details: Optional[str]
    """The error details."""

    def __init__(self, error: Dict[str, str]) -> None:
        self._raw_error = error
        self.message = error.get("message")
        self.code = error.get("code")
        self.hint = error.get("hint")
        self.details = error.get("details")
        Exception.__init__(self, str(self))

    def __repr__(self) -> str:
        error_text = f"Error {self.code}:" if self.code else ""
        message_text = f"\nMessage: {self.message}" if self.message else ""
        hint_text = f"\nHint: {self.hint}" if self.hint else ""
        details_text = f"\nDetails: {self.details}" if self.details else ""
        complete_error_text = f"{error_text}{message_text}{hint_text}{details_text}"
        return complete_error_text or "Empty error"

    def json(self) -> Dict[str, str]:
        """Convert the error into a dictionary.

        Returns:
            :class:`dict`
        """
        return self._raw_error


def generate_default_error_message(r):
    return {
        "message": "JSON could not be generated",
        "code": r.status_code,
        "hint": "Refer to full message for details",
        "details": str(r.content),
    }
