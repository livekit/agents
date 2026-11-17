import asyncio
import logging
import os
from typing import Any
from urllib.parse import quote
from xml.sax.saxutils import quoteattr

import aiohttp
from dotenv import load_dotenv

from livekit import api, rtc
from livekit.agents import (
    AMD,
    Agent,
    AgentServer,
    AgentSession,
    AMDCategory,
    AMDMenuObservedEvent,
    AMDPredictionEvent,
    AMDReason,
    JobContext,
    cli,
    inference,
    room_io,
    utils,
)
from livekit.agents.beta.tools import EndCallTool
from livekit.plugins import ai_coustics

logger = logging.getLogger("amd-twilio-connector-example")
load_dotenv()

# The same outbound AMD call as amd.py, placed through Twilio and the LiveKit Twilio
# Connector instead of a SIP trunk.
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")  # your Twilio number
TWILIO_TO_NUMBER = os.getenv("TWILIO_TO_NUMBER")  # the number to call
CALLEE_IDENTITY = "callee"
# Twilio reports an unanswered call only through status webhooks, so cap the wait
ANSWER_TIMEOUT = 45.0
TWILIO_API_BASE = "https://api.twilio.com/2010-04-01"


class MyAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are Alex from Acme Dental. You are calling Sam to confirm "
                "a dental appointment tomorrow at 10 AM. Keep replies brief. "
                "If asked to leave a message, give the appointment details and "
                "ask Sam to call the office to confirm. Do not invent a phone number. "
                "After leaving a voicemail, wait for any follow-up prompts. "
                "Use end_call when the conversation with a human is complete."
            ),
            tools=[
                EndCallTool(
                    # The default (True) also deletes the room; hangup() already does
                    # that on every exit path.
                    delete_room=False,
                    extra_description="Call after completing the appointment conversation with a human.",
                ),
            ],
        )


async def twilio_request(path: str, data: dict[str, str]) -> dict[str, Any]:
    assert TWILIO_ACCOUNT_SID is not None and TWILIO_AUTH_TOKEN is not None
    async with utils.http_context.http_session().post(
        f"{TWILIO_API_BASE}/Accounts/{quote(TWILIO_ACCOUNT_SID, safe='')}{path}",
        data=data,
        auth=aiohttp.BasicAuth(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
    ) as response:
        body = await response.json(content_type=None)
        if not isinstance(body, dict):
            body = {}
        if not 200 <= response.status < 300:
            raise RuntimeError(
                f"Twilio request failed (HTTP {response.status}, code {body.get('code')})"
            )
        return body


async def end_twilio_call(call_sid: str) -> None:
    path = f"/Calls/{quote(call_sid, safe='')}.json"
    try:
        # a ringing call is canceled; an answered one is completed
        await twilio_request(path, {"Status": "canceled"})
    except RuntimeError:
        try:
            await twilio_request(path, {"Status": "completed"})
        except RuntimeError:
            logger.warning("failed to end the Twilio call")


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }
    session = AgentSession(
        stt=inference.STT("deepgram/nova-3", language="multi"),
        llm=inference.LLM("google/gemma-4-31b-it"),
        tts=inference.TTS("cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"),
        # the callee can speak before the greeting ends; AgentSession skips the AEC warmup
        # for outbound SIP calls on its own but cannot tell an outbound connector call apart
        aec_warmup_duration=None,
    )

    await session.start(
        agent=MyAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S,
                ),
            ),
        ),
    )

    call_sid: str | None = None

    async def hangup() -> None:
        # deleting the room ends the media stream; ending the call also stops a call
        # that is still ringing
        if call_sid is not None:
            await end_twilio_call(call_sid)
        await ctx.delete_room()

    ctx.add_shutdown_callback(hangup)

    # AMD needs a phone call, so console mode is not supported.
    # room_io raises when the session was started without a room.
    _ = session.room_io

    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER and TWILIO_TO_NUMBER):
        logger.error("set the TWILIO_* variables to place the call")
        ctx.shutdown("missing Twilio configuration")
        return

    connect = await ctx.api.connector.connect_twilio_call(
        api.ConnectTwilioCallRequest(
            twilio_call_direction=api.ConnectTwilioCallRequest.TwilioCallDirection.TWILIO_CALL_DIRECTION_OUTBOUND,
            room_name=ctx.room.name,
            participant_identity=CALLEE_IDENTITY,
        )
    )
    twiml = (
        f"<Response><Connect><Stream url={quoteattr(connect.connect_url)}/></Connect></Response>"
    )
    call = await twilio_request(
        "/Calls.json", {"To": TWILIO_TO_NUMBER, "From": TWILIO_FROM_NUMBER, "Twiml": twiml}
    )
    call_sid = call["sid"]

    # Unlike a SIP participant, the connector publishes the callee's audio only once the
    # call is answered, and AMD waits only a few seconds for that track. So enter AMD
    # after the answer; there is no early media to discard.
    try:
        await asyncio.wait_for(
            utils.wait_for_track_publication(
                ctx.room, identity=CALLEE_IDENTITY, kind=rtc.TrackKind.KIND_AUDIO
            ),
            ANSWER_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.info("call was not answered")
        ctx.shutdown("call not answered")
        return

    detector = AMD(session, participant_identity=CALLEE_IDENTITY)

    @detector.on("amd_prediction")
    def on_prediction(event: AMDPredictionEvent) -> None:
        logger.info("amd prediction", extra={"lk.pii.amd.prediction": event.model_dump_json()})

    @detector.on("amd_menu_observed")
    def on_menu(event: AMDMenuObservedEvent) -> None:
        logger.info("amd menu (informational)", extra={"lk.pii.amd.menu": event.model_dump_json()})

    async with detector:
        # TODO: IVR replies press keys with DTMF; verify that the Twilio Connector forwards
        # it (not yet tested with an IVR menu)
        result = await detector.execute()
        logger.info("amd completed", extra={"lk.pii.amd.result": result.model_dump_json()})

    if result.is_error or result.reason == AMDReason.PARTICIPANT_DISCONNECTED:
        # the callee hung up, or AMD stopped without detecting the call, such as when
        # end_call closed the session (cancelled)
        ctx.shutdown(result.reason.value)
    elif result.voicemail_message_played and result.category != AMDCategory.HUMAN:
        ctx.shutdown("voicemail completed")
    elif result.category == AMDCategory.MACHINE_UNAVAILABLE:
        # nobody can take the call, such as a full mailbox; hang up
        ctx.shutdown("machine unavailable")


if __name__ == "__main__":
    cli.run_app(server)
