from __future__ import annotations

import logging

from livekit import rtc
from livekit.agents.beta.tools.dtmf import DtmfEvent
from livekit.agents.job import get_job_context
from livekit.agents.llm.chat_context import ChatContext
from livekit.agents.llm.tool_context import function_tool
from livekit.agents.utils.aio.debounce import debounced

from ...voice.agent import AgentTask

logger = logging.getLogger("dtmf-inputs")


class DtmfInputsTask(AgentTask[list[str]]):
    def __init__(self, chat_ctx: ChatContext, input_timeout: float = 5.0) -> None:
        super().__init__(
            instructions=(
                "You are a voice assistant that can collect DTMF inputs from the user."
                "Wait for the user to provide the DTMF inputs and confirm it with the user."
                "Once user has confirmed the DTMF inputs, call `confirm_dtmf_inputs` with the inputs."
                "If user decides to change / re-enter the DTMF inputs, wait for it and re-confirm until user is satisfied."
            ),
            chat_ctx=chat_ctx,
        )

        @debounced(delay=input_timeout)
        async def _generate_dtmf_reply() -> None:
            logger.info(
                f"Generating DTMF reply, current inputs: {', '.join(self._curr_dtmf_inputs)}"
            )
            handle = self.session.generate_reply(
                instructions=(
                    "User has provided the following DTMF inputs: "
                    f"{', '.join(self._curr_dtmf_inputs)}"
                    "Please confirm it with the user."
                ),
            )
            self._curr_dtmf_inputs = []
            await handle

        def _on_sip_dtmf_received(ev: rtc.SipDTMF) -> None:
            logger.info(
                f"DTMF input received: {ev.digit}, current inputs: {', '.join(self._curr_dtmf_inputs)}"
            )
            self._curr_dtmf_inputs.append(DtmfEvent(ev.digit))
            self._generate_dtmf_reply.schedule()

        self._curr_dtmf_inputs: list[DtmfEvent] = []
        self._generate_dtmf_reply = _generate_dtmf_reply
        self._on_sip_dtmf_received = _on_sip_dtmf_received

    @function_tool
    async def confirm_dtmf_inputs(self, inputs: list[DtmfEvent]) -> None:
        self.complete([inp.value for inp in inputs])

    async def on_enter(self) -> None:
        ctx = get_job_context()

        ctx.room.on("sip_dtmf_received", self._on_sip_dtmf_received)

    async def on_exit(self) -> None:
        ctx = get_job_context()

        ctx.room.off("sip_dtmf_received", self._on_sip_dtmf_received)
        self._generate_dtmf_reply.cancel()
