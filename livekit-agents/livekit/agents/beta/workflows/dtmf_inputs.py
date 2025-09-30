from __future__ import annotations

import logging

from livekit import rtc
from livekit.agents.voice.events import AgentStateChangedEvent, UserStateChangedEvent

from ... import function_tool
from ...job import get_job_context
from ...llm.chat_context import ChatContext
from ...types import NOT_GIVEN, NotGivenOr
from ...utils.aio.debounce import debounced
from ...voice.agent import AgentTask
from ..tools.dtmf import DtmfEvent, format_dtmf

logger = logging.getLogger("dtmf-inputs")


class GetDtmfTask(AgentTask[str | None]):
    """A task to collect DTMF inputs from the user.

    Return a string of DTMF inputs if collected successfully, otherwise None.
    """

    def __init__(
        self,
        name: str,
        num_digits: int,
        *,
        chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN,
        input_timeout: float = 5.0,
        ask_for_confirmation: bool = False,
        interrupt_on_complete_input: bool = False,
    ) -> None:
        """
        Args:
            name: The name of the input to collect.
            num_digits: The number of digits to collect.
            chat_ctx: The chat context to use.
            input_timeout: The per-digit timeout.
            ask_for_confirmation: Whether to ask for confirmation when agent has collected full digits.
            interrupt_on_complete_input: Whether to interrupt any active speech on full digits been collected.
        """
        if num_digits <= 0:
            raise ValueError("num_digits must be greater than 0")

        @function_tool
        async def confirm_dtmf_inputs(inputs: list[DtmfEvent]) -> None:
            self.complete(format_dtmf(inputs))

        instructions = (
            "You are a single step in a broader system, responsible solely for collecting DTMF inputs from the user. "
            f'Wait for the user to provide "{name}" which is a {num_digits}-digit DTMF inputs. '
        )
        tools = []

        if ask_for_confirmation:
            instructions += "Once user has confirmed the DTMF inputs, call `confirm_dtmf_inputs` with the inputs."
            tools.append(confirm_dtmf_inputs)

        super().__init__(
            instructions=instructions,
            chat_ctx=chat_ctx,
            tools=tools,
        )

        @debounced(delay=input_timeout)
        async def _generate_dtmf_reply() -> None:
            if interrupt_on_complete_input:
                self.session.interrupt()

            dmtf_str = format_dtmf(self._curr_dtmf_inputs)
            logger.debug(f"Generating DTMF reply, current inputs: {dmtf_str}")

            # if input not fully received (i.e. timeout), return None
            if len(self._curr_dtmf_inputs) != num_digits:
                return self.complete(None)

            # if not asking for confirmation, return the DTMF inputs
            if not ask_for_confirmation:
                return self.complete(dmtf_str)

            instructions = (
                "<dtmf_inputs>\n"
                "User has provided the following valid DTMF inputs: "
                f"{dmtf_str}. Please confirm it with the user. "
                "Once you are sure, call `confirm_dtmf_inputs` with the inputs.\n"
                "</dtmf_inputs>"
            )
            logger.debug("Generating DTMF confirmation prompt")

            self._curr_dtmf_inputs = []
            await self.session.generate_reply(instructions=instructions)

        def _on_sip_dtmf_received(ev: rtc.SipDTMF) -> None:
            self._curr_dtmf_inputs.append(DtmfEvent(ev.digit))
            logger.info(f"DTMF inputs: {format_dtmf(self._curr_dtmf_inputs)}")
            self._run_dtmf_reply_generation()

        def _on_user_state_changed(ev: UserStateChangedEvent) -> None:
            if ev.new_state == "speaking":
                # clear any pending DTMF reply generation
                logger.debug("User is speaking, cancelling DTMF reply generation")
                self._generate_dtmf_reply.cancel()
            elif len(self._curr_dtmf_inputs) != 0:
                # resume any previously cancelled DTMF reply generation after user is back to non-speaking
                logger.debug("User is back to non-speaking, resuming DTMF reply generation")
                self._run_dtmf_reply_generation()

        def _on_agent_state_changed(ev: AgentStateChangedEvent) -> None:
            if ev.new_state in ["speaking", "thinking"]:
                # clear any pending DTMF reply generation
                logger.debug("Agent is speaking, cancelling DTMF reply generation")
                self._generate_dtmf_reply.cancel()
            elif len(self._curr_dtmf_inputs) != 0:
                # resume any previously cancelled DTMF reply generation after user is back to non-speaking
                logger.debug("Agent is back to non-speaking, resuming DTMF reply generation")
                self._run_dtmf_reply_generation()

        self._name = name
        self._num_digits = num_digits
        self._curr_dtmf_inputs: list[DtmfEvent] = []
        self._generate_dtmf_reply = _generate_dtmf_reply
        self._on_sip_dtmf_received = _on_sip_dtmf_received
        self._on_user_state_changed = _on_user_state_changed
        self._on_agent_state_changed = _on_agent_state_changed

    def _run_dtmf_reply_generation(self) -> None:
        logger.debug("Running DTMF reply generation")
        if len(self._curr_dtmf_inputs) == self._num_digits:
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
