from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ... import llm, stt, tts, vad
from ...llm.chat_context import Instructions
from ...llm.tool_context import ToolError, ToolFlag, function_tool
from ...types import NOT_GIVEN, NotGivenOr
from ...utils import is_given
from ...voice.agent import AgentTask
from ...voice.events import RunContext

if TYPE_CHECKING:
    from ...voice.audio_recognition import TurnDetectionMode


_BASE_INSTRUCTIONS = """
You are only a single step in a broader system, responsible solely for capturing an address.
You will be handling addresses from any country.
{modality_specific}
Call `update_address` at the first opportunity whenever you form a new hypothesis about the address. (before asking any questions or providing any answers.)
Don't invent new addresses, stick strictly to what the user said.
{confirmation_instructions}
If the address is unclear or invalid, or it takes too much back-and-forth, prompt for it in parts in this order: street address, unit number if applicable, locality, and country.
Ignore unrelated input and avoid going off-topic. Do not generate markdown, greetings, or unnecessary commentary.
Always explicitly invoke a tool when applicable. Do not simulate tool usage, no real action is taken unless the tool is explicitly called.\
{extra_instructions}
"""

_AUDIO_SPECIFIC = """
Expect that users will say address in different formats with fields filled like:
- 'street_address': '450 SOUTH MAIN ST', 'unit_number': 'FLOOR 2', 'locality': 'SALT LAKE CITY UT 84101', 'country': 'UNITED STATES',
- 'street_address': '123 MAPLE STREET', 'unit_number': 'APARTMENT 10', 'locality': 'OTTAWA ON K1A 0B1', 'country': 'CANADA',
- 'street_address': 'GUOMAO JIE 3 HAO, CHAOYANG QU', 'unit_number': 'GUOMAO DA SHA 18 LOU 101 SHI', 'locality': 'BEIJING SHI 100000', 'country': 'CHINA',
- 'street_address': '5 RUE DE L'ANCIENNE COMÉDIE', 'unit_number': 'APP C4', 'locality': '75006 PARIS', 'country': 'FRANCE',
- 'street_address': 'PLOT 10, NEHRU ROAD', 'unit_number': 'OFFICE 403, 4TH FLOOR', 'locality': 'VILE PARLE (E), MUMBAI MAHARASHTRA 400099', 'country': 'INDIA',
Normalize common spoken patterns silently:
- Convert words like 'dash' and 'apostrophe' into symbols: `-`, `'`.
- Convert spelled out numbers like 'six' and 'seven' into numerals: `6`, `7`.
- Recognize patterns where users speak their address field followed by spelling: e.g., 'guomao g u o m a o'.
- Filter out filler words or hesitations.
- Recognize when there may be accents on certain letters if explicitly said or common in the location specified. Be sure to verify the correct accents if existent.
Don't mention corrections. Treat inputs as possibly imperfect but fix them silently.
When reading a numerical ordinal suffix (st, nd, rd, th), the number must be verbally expanded into its full, correctly pronounced word form.
Do not read the number and the suffix letters separately.
Confirm postal codes by reading them out digit-by-digit as a sequence of single numbers. Do not read them as cardinal numbers.
For example, read 90210 as 'nine zero two one zero.'
Avoid using bullet points and parenthese in any responses.
Spell out the address letter-by-letter when applicable, such as street names and provinces, especially when the user spells it out initially.
"""

_TEXT_SPECIFIC = """
Expect users to type their address directly.
If the address looks almost correct but has minor issues (e.g. missing country or postal code), prompt for clarification.
"""


@dataclass
class GetAddressResult:
    address: str


class GetAddressTask(AgentTask[GetAddressResult]):
    def __init__(
        self,
        extra_instructions: str = "",
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.Tool | llm.Toolset]] = NOT_GIVEN,
        stt: NotGivenOr[stt.STT | None] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel | None] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | None] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
        require_confirmation: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        confirmation_instructions = (
            "Call `confirm_address` after the user confirmed the address is correct."
        )
        extra = extra_instructions if extra_instructions else ""

        super().__init__(
            instructions=Instructions(
                _BASE_INSTRUCTIONS.format(
                    modality_specific=_AUDIO_SPECIFIC,
                    confirmation_instructions=(
                        confirmation_instructions if require_confirmation is not False else ""
                    ),
                    extra_instructions=extra,
                ),
                text=_BASE_INSTRUCTIONS.format(
                    modality_specific=_TEXT_SPECIFIC,
                    confirmation_instructions=(
                        confirmation_instructions if require_confirmation is True else ""
                    ),
                    extra_instructions=extra,
                ),
            ),
            chat_ctx=chat_ctx,
            turn_detection=turn_detection,
            tools=tools or [],
            stt=stt,
            vad=vad,
            llm=llm,
            tts=tts,
            allow_interruptions=allow_interruptions,
        )

        self._current_address = ""
        self._require_confirmation = require_confirmation

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="Ask the user to provide their address.")

    @function_tool()
    async def update_address(
        self, street_address: str, unit_number: str, locality: str, country: str, ctx: RunContext
    ) -> str | None:
        """Update the address provided by the user.

        Args:
            street_address (str): Dependent on country, may include fields like house number, street name, block, or district
            unit_number (str): The unit number, for example Floor 1 or Apartment 12. If there is no unit number, return ''
            locality (str): Dependent on country, may include fields like city, zip code, or province
            country (str): The country the user lives in spelled out fully
        """
        address_fields = (
            [street_address, unit_number, locality, country]
            if unit_number.strip()
            else [street_address, locality, country]
        )
        address = " ".join(address_fields)
        self._current_address = address

        if not self._confirmation_required(ctx):
            if not self.done():
                self.complete(GetAddressResult(address=self._current_address))
            return None

        confirm_tool = self._build_confirm_tool(address=address)
        current_tools = [t for t in self.tools if t.id != "confirm_address"]
        current_tools.append(confirm_tool)
        await self.update_tools(current_tools)

        return (
            f"The address has been updated to {address}\n"
            f"Repeat the address field by field: {address_fields} if needed\n"
            f"Prompt the user for confirmation, do not call `confirm_address` directly"
        )

    def _build_confirm_tool(self, *, address: str) -> llm.FunctionTool:
        # confirm tool is only injected after update_address is called,
        # preventing the LLM from hallucinating a confirmation without user input
        @function_tool()
        async def confirm_address() -> None:
            """Call after the user confirms the address is correct."""
            if address != self._current_address:
                self.session.generate_reply(
                    instructions="The address has changed since confirmation was requested, ask the user to confirm the updated address."
                )
                return

            if not self.done():
                self.complete(GetAddressResult(address=address))

        return confirm_address

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def decline_address_capture(self, reason: str) -> None:
        """Handles the case when the user explicitly declines to provide an address.

        Args:
            reason: A short explanation of why the user declined to provide the address
        """
        if not self.done():
            self.complete(ToolError(f"couldn't get the address: {reason}"))

    def _confirmation_required(self, ctx: RunContext) -> bool:
        if is_given(self._require_confirmation):
            return self._require_confirmation
        return ctx.speech_handle.input_details.modality == "audio"
