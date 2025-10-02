from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from livekit import rtc
from livekit.agents.llm.tool_context import ToolError
from livekit.agents.voice.events import AgentStateChangedEvent, UserStateChangedEvent

from ... import function_tool
from ...job import get_job_context
from ...llm.chat_context import ChatContext
from ...types import NOT_GIVEN, NotGivenOr
from ...utils import is_given
from ...utils.aio.debounce import Debounced, debounced
from ...voice.agent import AgentTask
from ..tools.dtmf import DtmfEvent, format_dtmf

logger = logging.getLogger("dtmf-inputs")


@dataclass
class GetDtmfResult:
    user_input: str

    @classmethod
    def from_dtmf_inputs(cls, dtmf_inputs: list[DtmfEvent]) -> GetDtmfResult:
        return cls(user_input=format_dtmf(dtmf_inputs))


class GetDtmfTask(AgentTask[GetDtmfResult]):
    """A task to collect DTMF inputs from the user.

    Return a string of DTMF inputs if collected successfully, otherwise None.
    """

    def __init__(
        self,
        *,
        num_digits: int,
        ask_for_confirmation: bool = False,
        dtmf_input_timeout: float = 5.0,
        chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN,
        extra_instructions: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Args:
            num_digits: The number of digits to collect.
            chat_ctx: The chat context to use.
            dtmf_input_timeout: The per-digit timeout.
            ask_for_confirmation: Whether to ask for confirmation when agent has collected full digits.
            extra_instructions: Extra instructions to add to the task.
        """
        if num_digits <= 0:
            raise ValueError("num_digits must be greater than 0")

        self._num_digits: int = num_digits
        self._curr_dtmf_inputs: list[DtmfEvent] = []

        @function_tool
        async def confirm_dtmf_inputs(inputs: list[DtmfEvent]) -> None:
            """Confirm the DTMF inputs.

            Called ONLY when user has explicitly confirmed the DTMF inputs is correct."""
            self.complete(GetDtmfResult.from_dtmf_inputs(inputs))

        instructions = "You are a single step in a broader system, responsible solely for collecting DTMF inputs from the user. "

        if is_given(extra_instructions):
            instructions += extra_instructions

        tools = []

        if ask_for_confirmation:
            instructions += "Once user has confirmed the DTMF inputs, call `confirm_dtmf_inputs` with the inputs."
            tools.append(confirm_dtmf_inputs)

        super().__init__(
            instructions=instructions,
            chat_ctx=chat_ctx,
            tools=tools,
        )

        @debounced(delay=dtmf_input_timeout)
        async def _generate_dtmf_reply() -> None:
            self.session.interrupt()

            dmtf_str = format_dtmf(self._curr_dtmf_inputs)
            logger.debug(f"Generating DTMF reply, current inputs: {dmtf_str}")

            # if input not fully received (i.e. timeout), return None
            if len(self._curr_dtmf_inputs) != num_digits:
                error_msg = f"DTMF inputs not fully received. Expect {num_digits} digits, got {len(self._curr_dtmf_inputs)}"
                self.complete(ToolError(error_msg))
                return

            # if not asking for confirmation, return the DTMF inputs
            if not ask_for_confirmation:
                self.complete(GetDtmfResult.from_dtmf_inputs(self._curr_dtmf_inputs))
                return

            instructions = (
                "<dtmf_inputs>\n"
                "User has provided the following valid DTMF inputs: "
                f"{dmtf_str}. Please confirm it with the user. "
                "Once you are sure, call `confirm_dtmf_inputs` with the inputs.\n"
                "</dtmf_inputs>"
            )

            await self.session.generate_reply(instructions=instructions)
            self._curr_dtmf_inputs.clear()

        def _on_sip_dtmf_received(ev: rtc.SipDTMF) -> None:
            if self.received_full_digits():
                # temporarily pause DTMF receive to prevent new DTMF inputs interrupts the reply generation
                return

            self._curr_dtmf_inputs.append(DtmfEvent(ev.digit))
            logger.info(f"DTMF inputs: {format_dtmf(self._curr_dtmf_inputs)}")
            self._run_dtmf_reply_generation()

        def _on_user_state_changed(ev: UserStateChangedEvent) -> None:
            if self.received_full_digits():
                return

            if ev.new_state == "speaking":
                # clear any pending DTMF reply generation
                self._generate_dtmf_reply.cancel()
            elif len(self._curr_dtmf_inputs) != 0:
                # resume any previously cancelled DTMF reply generation after user is back to non-speaking
                self._run_dtmf_reply_generation()

        def _on_agent_state_changed(ev: AgentStateChangedEvent) -> None:
            if self.received_full_digits():
                return

            if ev.new_state in ["speaking", "thinking"]:
                # clear any pending DTMF reply generation
                self._generate_dtmf_reply.cancel()
            elif len(self._curr_dtmf_inputs) != 0:
                # resume any previously cancelled DTMF reply generation after agent is back to non-speaking
                self._run_dtmf_reply_generation()

        self._generate_dtmf_reply: Debounced[None] = _generate_dtmf_reply
        self._on_sip_dtmf_received: Callable[[rtc.SipDTMF], None] = _on_sip_dtmf_received
        self._on_user_state_changed: Callable[[UserStateChangedEvent], None] = (
            _on_user_state_changed
        )
        self._on_agent_state_changed: Callable[[AgentStateChangedEvent], None] = (
            _on_agent_state_changed
        )

    def received_full_digits(self) -> bool:
        return len(self._curr_dtmf_inputs) >= self._num_digits

    def _run_dtmf_reply_generation(self) -> None:
        if self.received_full_digits():
            self._generate_dtmf_reply()
        else:
            self._generate_dtmf_reply.schedule()

    async def on_enter(self) -> None:
        ctx = get_job_context()

        ctx.room.on("sip_dtmf_received", self._on_sip_dtmf_received)
        self.session.on("agent_state_changed", self._on_user_state_changed)
        self.session.on("agent_state_changed", self._on_agent_state_changed)
        self.session.generate_reply()

    async def on_exit(self) -> None:
        ctx = get_job_context()

        ctx.room.off("sip_dtmf_received", self._on_sip_dtmf_received)
        self.session.off("agent_state_changed", self._on_user_state_changed)
        self.session.off("agent_state_changed", self._on_agent_state_changed)
        self._generate_dtmf_reply.cancel()
