"""Warm transfer that shows the supervisor the customer's number on a Telnyx SIP trunk.

Telnyx-only variant of warm_transfer.py. It needs an outbound trunk that terminates on
Telnyx. For any other SIP provider, use warm_transfer.py. See README.md.
"""

import logging
import os
import re
from typing import Any

from dotenv import load_dotenv
from support_agent import SUMMARY_INSTRUCTIONS, SupportAgent, start_session

from livekit import rtc
from livekit.agents import AgentServer, JobContext, cli
from livekit.agents.beta.workflows import WarmTransferResult, WarmTransferTask

load_dotenv()

logger = logging.getLogger("telnyx-sip-trunk-warm-transfer")

# ensure the following variables/env vars are set
SIP_TRUNK_ID = os.getenv("LIVEKIT_SIP_OUTBOUND_TRUNK")  # "ST_abcxyz"
SUPERVISOR_PHONE_NUMBER = os.getenv("LIVEKIT_SUPERVISOR_PHONE_NUMBER")  # "+12003004000"
SIP_NUMBER = os.getenv("LIVEKIT_SIP_NUMBER")  # "+15005006000" - fallback caller ID

E164 = re.compile(r"\+[1-9][0-9]{1,14}")


def telnyx_caller_id(caller: rtc.RemoteParticipant | None) -> dict[str, Any]:
    """Show the supervisor the inbound caller's number.

    Telnyx accepts the customer's number as the caller ID while the inbound call is
    still active, if a ``Diversion`` header names the Telnyx number that received it.
    Both numbers come from the caller's own SIP attributes, so they always belong to
    the same call. Falls back to the business number when either is unavailable.
    """
    fallback: dict[str, Any] = {"sip_number": SIP_NUMBER}
    if caller is None or caller.kind != rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        return fallback
    customer_number = caller.attributes.get("sip.phoneNumber", "")
    telnyx_number = caller.attributes.get("sip.trunkPhoneNumber", "")
    if not (E164.fullmatch(customer_number) and E164.fullmatch(telnyx_number)):
        logger.warning("caller or Telnyx number is missing or not E.164, using the business number")
        return fallback
    return {
        "sip_number": customer_number,
        "sip_headers": {"Diversion": f"<sip:{telnyx_number}@sip.telnyx.com>;reason=unconditional"},
    }


class TelnyxSIPSupportAgent(SupportAgent):
    async def _start_transfer(self) -> WarmTransferResult:
        assert SIP_TRUNK_ID is not None
        assert SUPERVISOR_PHONE_NUMBER is not None
        return await WarmTransferTask(
            sip_call_to=SUPERVISOR_PHONE_NUMBER,
            sip_trunk_id=SIP_TRUNK_ID,
            **telnyx_caller_id(self.session.room_io.linked_participant),
            chat_ctx=self.chat_ctx,
            # to reach an extension behind an IVR, pass DTMF tones to send once
            # answered, e.g. dtmf="wwww1234#" (each `w` pauses ~0.5s):
            # dtmf=SUPERVISOR_EXTENSION,
            # give up if the supervisor doesn't pick up within 25s:
            # ringing_timeout=25,
            # add extra instructions for summarization
            extra_instructions=SUMMARY_INSTRUCTIONS,
        )


server = AgentServer()


@server.rtc_session(agent_name="sip-inbound")
async def entrypoint(ctx: JobContext) -> None:
    await start_session(ctx, TelnyxSIPSupportAgent())


if __name__ == "__main__":
    cli.run_app(server)
