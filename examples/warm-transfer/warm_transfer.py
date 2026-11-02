import os
from typing import Any

from dotenv import load_dotenv
from support_agent import SUMMARY_INSTRUCTIONS, SupportAgent, start_session

from livekit import rtc
from livekit.agents import AgentServer, JobContext, cli, get_job_context
from livekit.agents.beta.workflows import WarmTransferResult, WarmTransferTask

load_dotenv()

# ensure the following variables/env vars are set
SIP_TRUNK_ID = os.getenv("LIVEKIT_SIP_OUTBOUND_TRUNK")  # "ST_abcxyz"
SUPERVISOR_PHONE_NUMBER = os.getenv("LIVEKIT_SUPERVISOR_PHONE_NUMBER")  # "+12003004000"
SIP_NUMBER = os.getenv("LIVEKIT_SIP_NUMBER")  # "+15005006000" - caller ID shown to supervisor


def twilio_caller_id() -> dict[str, Any]:
    """Show the supervisor the inbound caller's number on Twilio Elastic SIP Trunking.

    Requires Twilio's CallToken beta and
    ``headers_to_attributes: {"X-Twilio-CallToken": "twilio.call_token"}`` on the LiveKit
    inbound trunk. The token is forwarded as-is with the caller's number from the same
    call. Without both (the attribute is set asynchronously), the business number is used.
    See README.md.
    """
    caller = next(
        (
            p
            for p in get_job_context().room.remote_participants.values()
            if p.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
        ),
        None,
    )
    call_token = caller.attributes.get("twilio.call_token") if caller else None
    caller_number = caller.attributes.get("sip.phoneNumber") if caller else None
    if not call_token or not caller_number:
        return {"sip_number": SIP_NUMBER}
    return {"sip_number": caller_number, "sip_headers": {"X-Twilio-CallToken": call_token}}


class SIPSupportAgent(SupportAgent):
    async def _start_transfer(self) -> WarmTransferResult:
        assert SIP_TRUNK_ID is not None
        assert SUPERVISOR_PHONE_NUMBER is not None
        return await WarmTransferTask(
            sip_call_to=SUPERVISOR_PHONE_NUMBER,
            sip_trunk_id=SIP_TRUNK_ID,
            **twilio_caller_id(),
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
    await start_session(ctx, SIPSupportAgent())


if __name__ == "__main__":
    cli.run_app(server)
