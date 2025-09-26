from __future__ import annotations

import logging
from dataclasses import dataclass

from livekit import rtc
from livekit.agents.beta.tools.dtmf import DtmfEvent, format_dtmf
from livekit.agents.job import get_job_context
from livekit.agents.llm.chat_context import ChatContext
from livekit.agents.llm.tool_context import function_tool
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils.aio.debounce import debounced

from ...voice.agent import AgentTask

logger = logging.getLogger("dtmf-inputs")


SINGLE_DIGIT_INSTRUCTIONS = """
You are a voice assistant that can collect DTMF inputs from the user on a phone call.

You need to prompt user to provide a single digit input for a list of choices.

For example, press 1 for x, press 2 for y, press 3 for z, etc.

Below are the choices:

<choices>
{choices}
</choices>

Any message from DTMF are wrapped in <dtmf_inputs> and </dtmf_inputs> tags.

You need to wait for the user to provide the number inputs. User can also provide input via voice (without any tags). If so, ask user to confirm the input.

Once user has confirmed the DTMF inputs, call `confirm_dtmf_inputs` with the inputs.
If user decides to change / re-enter the DTMF inputs, wait for it and re-confirm until user is satisfied.
"""

MULTI_DIGIT_INSTRUCTIONS = """
You are a voice assistant that can collect DTMF inputs from the user.
Wait for the user to provide {name} with {num_digits} digits. The can be described as follows:

<description>
{description}
</description>

Any message from DTMF are wrapped in <dtmf_inputs> and </dtmf_inputs> tags.

User can also provide input via voice (without any tags). If so, ask user to confirm the input.

Once user has confirmed the DTMF inputs, call `confirm_dtmf_inputs` with the inputs.
If user decides to change / re-enter the DTMF inputs, wait for it and re-confirm until user is satisfied.
"""


@dataclass
class SingleDigitConfig:
    choices: dict[DtmfEvent, str]

    def validate(self, inputs: list[DtmfEvent]) -> bool:
        return len(inputs) == 1 and inputs[0] in self.choices.keys()

    def stop_reached(self, inputs: list[DtmfEvent]) -> bool:
        return len(inputs) >= 1

    def format_instructions(self) -> str:
        choices_str = "\n".join(
            [
                "\n".join(
                    [
                        "<choice>",
                        f"<value>{k.value}</value>",
                        f"<description>{v}</description>",
                        "</choice>",
                    ]
                )
                for k, v in self.choices.items()
            ]
        )
        return SINGLE_DIGIT_INSTRUCTIONS.format(choices=choices_str)


@dataclass
class MultiDigitConfig:
    name: str
    num_digits: int
    description: str

    def validate(self, inputs: list[DtmfEvent]) -> bool:
        return len(inputs) == self.num_digits

    def stop_reached(self, inputs: list[DtmfEvent]) -> bool:
        return len(inputs) >= self.num_digits

    def format_instructions(self) -> str:
        return MULTI_DIGIT_INSTRUCTIONS.format(
            name=self.name,
            num_digits=self.num_digits,
            description=self.description,
        )


class GetDtmfTask(AgentTask[list[DtmfEvent]]):
    def __init__(
        self,
        input_config: SingleDigitConfig | MultiDigitConfig,
        chat_ctx: NotGivenOr[ChatContext] = NOT_GIVEN,
        input_timeout: float = 5.0,
        interrupt_on_dtmf_sent: bool = False,
    ) -> None:
        super().__init__(
            instructions=input_config.format_instructions(),
            chat_ctx=chat_ctx,
        )

        @debounced(delay=input_timeout)
        async def _generate_dtmf_reply() -> None:
            if interrupt_on_dtmf_sent:
                self.session.interrupt()

            dmtf_str = format_dtmf(self._curr_dtmf_inputs)
            logger.info(f"Generating DTMF reply, current inputs: {dmtf_str}")

            instructions = (
                (
                    "<dtmf_inputs>\n"
                    "User has provided the following valid DTMF inputs: "
                    f"{dmtf_str}. Please confirm it with the user.\n"
                    "</dtmf_inputs>"
                )
                if input_config.validate(self._curr_dtmf_inputs)
                else (
                    "<dtmf_inputs>\n"
                    "User has provided the following invalid DTMF inputs:"
                    f"{dmtf_str}. Please inform the user that the inputs are invalid and ask them to provide the correct inputs.\n"
                    "</dtmf_inputs>"
                )
            )
            logger.info(f"Generating DTMF reply, instructions: {instructions}")

            handle = self.session.generate_reply(instructions=instructions)
            self._curr_dtmf_inputs = []
            await handle

        def _on_sip_dtmf_received(ev: rtc.SipDTMF) -> None:
            self._curr_dtmf_inputs.append(DtmfEvent(ev.digit))
            logger.info(
                f"DTMF input received. Current inputs: {format_dtmf(self._curr_dtmf_inputs)}"
            )

            if input_config.stop_reached(self._curr_dtmf_inputs):
                self._generate_dtmf_reply()
            else:
                self._generate_dtmf_reply.schedule()

        self._curr_dtmf_inputs: list[DtmfEvent] = []
        self._generate_dtmf_reply = _generate_dtmf_reply
        self._on_sip_dtmf_received = _on_sip_dtmf_received

    @function_tool
    async def confirm_dtmf_inputs(self, inputs: list[DtmfEvent]) -> None:
        self.complete(inputs)

    async def on_enter(self) -> None:
        ctx = get_job_context()

        ctx.room.on("sip_dtmf_received", self._on_sip_dtmf_received)
        self.session.generate_reply()

    async def on_exit(self) -> None:
        ctx = get_job_context()

        ctx.room.off("sip_dtmf_received", self._on_sip_dtmf_received)
        self._generate_dtmf_reply.cancel()
