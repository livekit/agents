import os

from dotenv import load_dotenv
from support_agent import SUMMARY_INSTRUCTIONS, SupportAgent, run

from livekit.agents.beta.workflows import WarmTransferResult, WarmTransferTask

load_dotenv()

# ensure the following variables/env vars are set
SIP_TRUNK_ID = os.getenv("LIVEKIT_SIP_OUTBOUND_TRUNK")  # "ST_abcxyz"
SUPERVISOR_PHONE_NUMBER = os.getenv("LIVEKIT_SUPERVISOR_PHONE_NUMBER")  # "+12003004000"
SIP_NUMBER = os.getenv("LIVEKIT_SIP_NUMBER")  # "+15005006000" - caller ID shown to supervisor


class SIPSupportAgent(SupportAgent):
    async def _start_transfer(self) -> WarmTransferResult:
        assert SIP_TRUNK_ID is not None
        assert SUPERVISOR_PHONE_NUMBER is not None
        return await WarmTransferTask(
            sip_call_to=SUPERVISOR_PHONE_NUMBER,
            sip_trunk_id=SIP_TRUNK_ID,
            sip_number=SIP_NUMBER,
            chat_ctx=self.chat_ctx,
            # to reach an extension behind an IVR, pass DTMF tones to send once
            # answered, e.g. dtmf="wwww1234#" (each `w` pauses ~0.5s):
            # dtmf=SUPERVISOR_EXTENSION,
            # give up if the supervisor doesn't pick up within 25s:
            # ringing_timeout=25,
            # add extra instructions for summarization
            extra_instructions=SUMMARY_INSTRUCTIONS,
        )


if __name__ == "__main__":
    run(SIPSupportAgent)
