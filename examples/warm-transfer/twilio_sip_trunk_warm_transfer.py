"""Warm transfer that shows the supervisor the customer's number on Twilio Elastic SIP Trunking.

Twilio-only variant of warm_transfer.py. It needs an outbound trunk that terminates on
Twilio Elastic SIP Trunking and Twilio's CallToken beta. For any other SIP provider, use
warm_transfer.py. See README.md.
"""

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from support_agent import SUMMARY_INSTRUCTIONS, SupportAgent, start_session

from livekit import rtc
from livekit.agents import AgentServer, JobContext, cli, get_job_context
from livekit.agents.beta.workflows import WarmTransferResult, WarmTransferTask

load_dotenv()

logger = logging.getLogger("twilio-sip-trunk-warm-transfer")

# ensure the following variables/env vars are set
SIP_TRUNK_ID = os.getenv("LIVEKIT_SIP_OUTBOUND_TRUNK")  # "ST_abcxyz"
SUPERVISOR_PHONE_NUMBER = os.getenv("LIVEKIT_SUPERVISOR_PHONE_NUMBER")  # "+12003004000"
SIP_NUMBER = os.getenv("LIVEKIT_SIP_NUMBER")  # "+15005006000" - fallback caller ID


async def twilio_caller_id(caller: rtc.RemoteParticipant | None) -> dict[str, Any]:
    """Show the supervisor the inbound caller's number.

    Reads ``X-Twilio-CallToken`` from the caller's SIP leg and forwards it as-is with the
    caller's number from the same call. Falls back to the business number when either is
    unavailable.
    """
    fallback: dict[str, Any] = {"sip_number": SIP_NUMBER}
    if caller is None or caller.kind != rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
        return fallback
    caller_number = caller.attributes.get("sip.phoneNumber")
    if not caller_number:
        return fallback
    try:
        response = await get_job_context().room.local_participant.perform_rpc(
            destination_identity=caller.identity,
            method="lk.sip.GetRemoteHeaders",
            payload=json.dumps({"include": ["X-Twilio-CallToken"]}),
        )
    except rtc.RpcError:
        logger.warning("could not read SIP headers, using the business number")
        return fallback
    try:
        headers = json.loads(response).get("headers") or {}
        call_token = next(
            (v for k, v in headers.items() if k.lower() == "x-twilio-calltoken"), None
        )
    except (ValueError, AttributeError):
        # an empty or malformed response counts as a missing header
        logger.warning("unexpected SIP headers response, using the business number")
        return fallback
    if not isinstance(call_token, str) or not call_token:
        return fallback
    return {"sip_number": caller_number, "sip_headers": {"X-Twilio-CallToken": call_token}}


class TwilioSIPSupportAgent(SupportAgent):
    async def _start_transfer(self) -> WarmTransferResult:
        assert SIP_TRUNK_ID is not None
        assert SUPERVISOR_PHONE_NUMBER is not None
        return await WarmTransferTask(
            sip_call_to=SUPERVISOR_PHONE_NUMBER,
            sip_trunk_id=SIP_TRUNK_ID,
            **await twilio_caller_id(self.session.room_io.linked_participant),
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
    await start_session(ctx, TwilioSIPSupportAgent())


if __name__ == "__main__":
    cli.run_app(server)
