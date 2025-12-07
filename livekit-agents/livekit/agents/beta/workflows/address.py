from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ... import llm, stt, tts, vad
from ...llm.tool_context import ToolError, ToolFlag, function_tool
from ...types import NOT_GIVEN, NotGivenOr
from ...voice.agent import AgentTask
from ...voice.events import RunContext
from ...voice.speech_handle import SpeechHandle

if TYPE_CHECKING:
    from ...voice.audio_recognition import TurnDetectionMode


@dataclass
class GetAddressResult:
    address: str


class GetAddressTask(AgentTask[GetAddressResult]):
    def __init__(
        self,
        extra_instructions: str = "",
        chat_ctx: NotGivenOr[llm.ChatContext] = NOT_GIVEN,
        turn_detection: NotGivenOr[TurnDetectionMode | None] = NOT_GIVEN,
        tools: NotGivenOr[list[llm.FunctionTool | llm.RawFunctionTool]] = NOT_GIVEN,
        stt: NotGivenOr[stt.STT | None] = NOT_GIVEN,
        vad: NotGivenOr[vad.VAD | None] = NOT_GIVEN,
        llm: NotGivenOr[llm.LLM | llm.RealtimeModel | None] = NOT_GIVEN,
        tts: NotGivenOr[tts.TTS | None] = NOT_GIVEN,
        allow_interruptions: NotGivenOr[bool] = NOT_GIVEN,
    ) -> None:
        super().__init__(
            instructions=(
                "You are only a single step in a broader system, responsible solely for capturing an address.\n"
                "You will be handling addresses from any country. Expect that users will say address in different formats with fields filled like:\n"
                "- 'street_address': '450 SOUTH MAIN ST', 'unit_number': 'FLOOR 2', 'locality': 'SALT LAKE CITY UT 84101', 'country': 'UNITED STATES',\n"
                "- 'street_address': '123 MAPLE STREET', 'unit_number': 'APARTMENT 10', 'locality': 'OTTAWA ON K1A 0B1', 'country': 'CANADA',\n"
                "- 'street_address': 'GUOMAO JIE 3 HAO, CHAOYANG QU', 'unit_number': 'GUOMAO DA SHA 18 LOU 101 SHI', 'locality': 'BEIJING SHI 100000', 'country': 'CHINA',\n"
                "- 'street_address': '5 RUE DE L'ANCIENNE COMÉDIE', 'unit_number': 'APP C4', 'locality': '75006 PARIS', 'country': 'FRANCE',\n"
                "- 'street_address': 'PLOT 10, NEHRU ROAD', 'unit_number': 'OFFICE 403, 4TH FLOOR', 'locality': 'VILE PARLE (E), MUMBAI MAHARASHTRA 400099', 'country': 'INDIA',\n"
                "Normalize common spoken patterns silently:\n"
                "- Convert words like 'dash' and 'apostrophe' into symbols: `-`, `'`.\n"
                "- Convert spelled out numbers like 'six' and 'seven' into numerals: `6`, `7`.\n"
                "- Recognize patterns where users speak their address field followed by spelling: e.g., 'guomao g u o m a o'.\n"
                "- Filter out filler words or hesitations.\n"
                "- Recognize when there may be accents on certain letters if explicitly said or common in the location specified. Be sure to verify the correct accents if existent.\n"
                "Don't mention corrections. Treat inputs as possibly imperfect but fix them silently.\n"
                "Call `update_address` at the first opportunity whenever you form a new hypothesis about the address. "
                "(before asking any questions or providing any answers.) \n"
                "Don't invent new addresses, stick strictly to what the user said. \n"
                "Call `confirm_address` after the user confirmed the address is correct. \n"
                "When reading a numerical ordinal suffix (st, nd, rd, th), the number must be verbally expanded into its full, correctly pronounced word form.\n"
                "Do not read the number and the suffix letters separately.\n"
                "Confirm postal codes by reading them out digit-by-digit as a sequence of single numbers. Do not read them as cardinal numbers.\n"
                "For example, read 90210 as 'nine zero two one zero.'\n"
                "Avoid using bullet points and parenthese in any responses.\n"
                "Spell out the address letter-by-letter when applicable, such as street names and provinces, especially when the user spells it out initially. \n"
                "If the address is unclear or invalid, or it takes too much back-and-forth, prompt for it in parts in this order: street address, unit number if applicable, locality, and country. \n"
                "Ignore unrelated input and avoid going off-topic. Do not generate markdown, greetings, or unnecessary commentary. \n"
                "Always explicitly invoke a tool when applicable. Do not simulate tool usage, no real action is taken unless the tool is explicitly called."
                + extra_instructions
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

        self._address_update_speech_handle: SpeechHandle | None = None

    async def on_enter(self) -> None:
        self.session.generate_reply(instructions="Ask the user to provide their address.")

    @function_tool()
    async def update_address(
        self, street_address: str, unit_number: str, locality: str, country: str, ctx: RunContext
    ) -> str:
        """Update the address provided by the user.

        Args:
            street_address (str): Dependent on country, may include fields like house number, street name, block, or district
            unit_number (str): The unit number, for example Floor 1 or Apartment 12. If there is no unit number, return ''
            locality (str): Dependent on country, may include fields like city, zip code, or province
            country (str): The country the user lives in spelled out fully
        """
        self._address_update_speech_handle = ctx.speech_handle
        address_fields = (
            [street_address, unit_number, locality, country]
            if unit_number.strip()
            else [street_address, locality, country]
        )
        address = " ".join(address_fields)
        self._current_address = address

        return (
            f"The address has been updated to {address}\n"
            f"Repeat the address field by field: {address_fields} if needed\n"
            f"Prompt the user for confirmation, do not call `confirm_address` directly"
        )

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def confirm_address(self, ctx: RunContext) -> None:
        """Call this tool when the user confirms that the address is correct."""
        await ctx.wait_for_playout()

        if ctx.speech_handle == self._address_update_speech_handle:
            raise ToolError("error: the user must confirm the address explicitly")

        if not self._current_address:
            raise ToolError(
                "error: no address was provided, `update_address` must be called before"
            )

        if not self.done():
            self.complete(GetAddressResult(address=self._current_address))

    @function_tool(flags=ToolFlag.IGNORE_ON_ENTER)
    async def decline_address_capture(self, reason: str) -> None:
        """Handles the case when the user explicitly declines to provide an address.

        Args:
            reason: A short explanation of why the user declined to provide the address
        """
        if not self.done():
            self.complete(ToolError(f"couldn't get the address: {reason}"))
