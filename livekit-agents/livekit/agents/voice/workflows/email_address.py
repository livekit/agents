from ..agent import AgentTask
from ...llm.tool_context import function_tool, ToolError
from dataclasses import dataclass
import re


EMAIL_REGEX = (
    r"^[A-Za-z0-9][A-Za-z0-9._%+\-]*@(?:[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?\.)+[A-Za-z]{2,}$"
)


@dataclass
class GetEmailResult:
    email_address: str


class GetEmailAgent(AgentTask[GetEmailResult]):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a voice assistant that captures and confirms the user's email address. "
                "The input may have transcription errors from the speech-to-text. Quietly fix these errors as needed without mentioning them. "
                "When you have a confident guess of the email, call 'update_email_address'. Only call 'validate_email_address' after the user clearly confirms the email. "
                "If the email is unclear or invalid, guide the user to repeat it in parts—first the section before the @, then the domain—only when necessary."
                "Avoid validating the email address too early and avoid generating any markdown in your output."
            )
        )

        self._current_email = ""

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="Ask for the email address")

    @function_tool
    async def update_email_address(self, email: str) -> str:
        """Store and your best guess of the user's email address.

        Args:
            email: The corrected email address provided by the language model.
        """
        email = email.strip()

        if not re.match(EMAIL_REGEX, email):
            raise ToolError(f"Invalid email address provided: {email}")

        self._current_email = email
        separated_email = " ".join(email)

        return (
            f"Confirm the provided email address with the user: f{email}\n"
            f"For clarity with the text-to-speech, also repeat it character by character: {separated_email}"
        )

    @function_tool
    async def validate_email_address(self) -> None:
        """Validates the email address after explicit user confirmation."""
        if not self._current_email.strip():
            raise ToolError("No valid email address were provided")

        self.complete(GetEmailResult(email_address=self._current_email))
