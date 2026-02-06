from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from livekit import rtc

from ... import function_tool
from ...job import get_job_context
from ...llm.chat_context import ChatContext
from ...llm.tool_context import ToolError
from ...types import NOT_GIVEN, NotGivenOr
from ...utils import is_given
from ...utils.aio.debounce import Debounced, debounced
from ...voice.agent import AgentTask
from ...voice.events import AgentStateChangedEvent, UserStateChangedEvent
from ..workflows.utils import DtmfEvent, format_dtmf

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
        dtmf_input_timeout: float = 4.0,
        dtmf_stop_event: DtmfEvent = DtmfEvent.POUND,
        chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN,
        extra_instructions: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        """
        Args:
            num_digits: The number of digits to collect.
            ask_for_confirmation: Whether to ask for confirmation when agent has collected full digits.
            repeat_instructions: The number of times to repeat the initial instructions.
            dtmf_input_timeout: The per-digit timeout.
            dtmf_stop_event: The DTMF event to stop collecting inputs.
            chat_ctx: The chat context to use.
            extra_instructions: Extra instructions to add to the task.
        """
        if num_digits <= 0:
            raise ValueError("num_digits must be greater than 0")

        self._curr_dtmf_inputs: list[DtmfEvent] = []
        self._dtmf_reply_running: bool = False

        @function_tool
        async def confirm_inputs(inputs: list[DtmfEvent]) -> None:
            """Finalize the collected digit inputs after explicit user confirmation.

            Use this ONLY after the confirmation. You should confirm by verbally reading out the digits one by one and, once the
            user confirms they are correct, call this tool with the inputs.

            Do not use this tool to capture the initial digits."""
            self.complete(GetDtmfResult.from_dtmf_inputs(inputs))

        @function_tool
        async def record_inputs(inputs: list[DtmfEvent]) -> None:
            """Record the collected digit inputs without additional confirmation.

            Call this tool as soon as a valid sequence of digits has been provided by the user (via DTMF or spoken)."""
            self.complete(GetDtmfResult.from_dtmf_inputs(inputs))

        instructions = (
            "You are a single step in a broader system, responsible solely for gathering digits input from the user. "
            "You will either receive a sequence of digits through dtmf events tagged by <dtmf_inputs>, or "
            "user will directly say the digits to you. You should be able to handle both cases. "
        )

        if ask_for_confirmation:
            instructions += "Once user has confirmed the digits (by verbally spoken or entered manually), call `confirm_inputs` with the inputs."
        else:
            instructions += "If user provides the digits through voice and it is valid, call `record_inputs` with the inputs."

        if is_given(extra_instructions):
            instructions += f"\n{extra_instructions}"

        super().__init__(
            instructions=instructions,
            chat_ctx=chat_ctx,
            tools=[confirm_inputs] if ask_for_confirmation else [record_inputs],
        )

        def _on_sip_dtmf_received(ev: rtc.SipDTMF) -> None:
            if self._dtmf_reply_running:
                return

            # immediately kick off the DTMF reply generation if matches the stop event
            if ev.digit == dtmf_stop_event.value:
                self._generate_dtmf_reply()
                return

            self._curr_dtmf_inputs.append(DtmfEvent(ev.digit))
            logger.info(f"DTMF inputs: {format_dtmf(self._curr_dtmf_inputs)}")
            self._generate_dtmf_reply.schedule()

        @debounced(delay=dtmf_input_timeout)
        async def _generate_dtmf_reply() -> None:
            self._dtmf_reply_running = True

            try:
                self.session.interrupt()

                dmtf_str = format_dtmf(self._curr_dtmf_inputs)
                logger.debug(f"Generating DTMF reply, current inputs: {dmtf_str}")

                # if input not fully received (i.e. timeout), return None
                if len(self._curr_dtmf_inputs) != num_digits:
                    error_msg = (
                        f"Digits input not fully received. "
                        f"Expect {num_digits} digits, got {len(self._curr_dtmf_inputs)}"
                    )
                    self.complete(ToolError(error_msg))
                    return

                # if not asking for confirmation, return the DTMF inputs
                if not ask_for_confirmation:
                    self.complete(GetDtmfResult.from_dtmf_inputs(self._curr_dtmf_inputs))
                    return

                instructions = (
                    "User has entered the following valid digits on the telephone keypad:\n"
                    f"<dtmf_inputs>{dmtf_str}</dtmf_inputs>\n"
                    "Please confirm it with the user by saying the digits one by one with space in between "
                    "(.e.g. 'one two three four five six seven eight nine ten'). "
                    "Once you are sure, call `confirm_inputs` with the inputs."
                    ""
                )

                await self.session.generate_reply(user_input=instructions)
            finally:
                self._dtmf_reply_running = False
                self._curr_dtmf_inputs.clear()

        def _on_user_state_changed(ev: UserStateChangedEvent) -> None:
            if self.dtmf_reply_running():
                return

            if ev.new_state == "speaking":
                # clear any pending DTMF reply generation
                self._generate_dtmf_reply.cancel()
            elif len(self._curr_dtmf_inputs) != 0:
                # resume any previously cancelled DTMF reply generation after user is back to non-speaking
                self._generate_dtmf_reply.schedule()

        def _on_agent_state_changed(ev: AgentStateChangedEvent) -> None:
            if self.dtmf_reply_running():
                return

            if ev.new_state in ["speaking", "thinking"]:
                # clear any pending DTMF reply generation
                self._generate_dtmf_reply.cancel()
            elif len(self._curr_dtmf_inputs) != 0:
                # resume any previously cancelled DTMF reply generation after agent is back to non-speaking
                self._generate_dtmf_reply.schedule()

        self._generate_dtmf_reply: Debounced[None] = _generate_dtmf_reply
        self._on_sip_dtmf_received: Callable[[rtc.SipDTMF], None] = _on_sip_dtmf_received
        self._on_user_state_changed: Callable[[UserStateChangedEvent], None] = (
            _on_user_state_changed
        )
        self._on_agent_state_changed: Callable[[AgentStateChangedEvent], None] = (
            _on_agent_state_changed
        )

    def dtmf_reply_running(self) -> bool:
        return self._dtmf_reply_running

    async def on_enter(self) -> None:
        ctx = get_job_context()

        ctx.room.on("sip_dtmf_received", self._on_sip_dtmf_received)
        self.session.on("agent_state_changed", self._on_user_state_changed)
        self.session.on("agent_state_changed", self._on_agent_state_changed)
        self.session.generate_reply(tool_choice="none")

    async def on_exit(self) -> None:
        ctx = get_job_context()

        ctx.room.off("sip_dtmf_received", self._on_sip_dtmf_received)
        self.session.off("agent_state_changed", self._on_user_state_changed)
        self.session.off("agent_state_changed", self._on_agent_state_changed)
        self._generate_dtmf_reply.cancel()
