"""Connect WhatsApp Business calls to a LiveKit agent with the WhatsApp connector.

Meta delivers call events to a webhook you register with your WhatsApp
Business app. This server handles those events: it accepts inbound calls,
completes outbound calls, and cleans up when a call ends. The call joins a
LiveKit room as a regular participant.

This example reuses the DTMF agent from ../basic_dtmf_agent.py. Run that agent
in another terminal, then run this server. See README.md for setup.

Docs: https://docs.livekit.io/telephony/connectors/whatsapp/
Meta docs: https://developers.facebook.com/documentation/business-messaging/whatsapp/calling
"""

import argparse
import asyncio
import hashlib
import hmac
import json
import logging
import os
import uuid
from collections.abc import Coroutine

from aiohttp import web
from dotenv import load_dotenv

from livekit import api
from livekit.protocol.agent_dispatch import RoomAgentDispatch
from livekit.protocol.rtc import SessionDescription

load_dotenv()

logger = logging.getLogger("whatsapp-connector-example")
logger.setLevel(logging.INFO)

# Must match the dispatch name the agent registered with.
AGENT_NAME = os.getenv("DTMF_AGENT_DISPATCH_NAME", "my-telephony-agent")
PORT = int(os.getenv("PORT", "8080"))

# From your Meta app: the business phone number ID and an access token.
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")
WHATSAPP_API_KEY = os.getenv("WHATSAPP_API_KEY")
# Must be a version the connector supports. See the LiveKit docs for the list.
WHATSAPP_CLOUD_API_VERSION = os.getenv("WHATSAPP_CLOUD_API_VERSION", "26.0")
# The token you chose when registering the webhook with Meta.
WHATSAPP_VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "livekit-connector-example")
# Optional: your Meta app secret. When set, webhook signatures are verified.
WHATSAPP_APP_SECRET = os.getenv("WHATSAPP_APP_SECRET")


async def handle_verification(request: web.Request) -> web.Response:
    """Meta verifies the webhook once at registration with a GET challenge."""
    query = request.rel_url.query
    if (
        query.get("hub.mode") == "subscribe"
        and query.get("hub.verify_token") == WHATSAPP_VERIFY_TOKEN
    ):
        return web.Response(text=query.get("hub.challenge", ""))
    return web.Response(status=403)


async def handle_call_event(lkapi: api.LiveKitAPI, call: dict, phone_number_id: str) -> None:
    """Route one entry of the webhook's `calls` array to the connector API."""
    call_id = call.get("id", "")
    event = call.get("event", "")
    direction = call.get("direction", "")
    logger.info(f"Call event {event} ({direction}) for {call_id}")

    if event == "connect" and direction == "USER_INITIATED":
        # An inbound call. Accept it right away: the caller's phone is already
        # ringing, and WhatsApp drops the call if media takes too long to start.
        # wait_until_answered makes this request block until the agent is in
        # the room, so a failure to answer surfaces here as an error.
        res = await accept_call(lkapi, call, call_id, phone_number_id)
        if res is not None:
            logger.info(f"Accepted call {call_id} into room {res.room_name}")

    elif event == "connect" and direction == "BUSINESS_INITIATED":
        # The callee answered an outbound call placed with `dial`. The webhook
        # carries their SDP answer; pass it on to complete the connection.
        try:
            await lkapi.connector.connect_whatsapp_call(
                api.ConnectWhatsAppCallRequest(
                    whatsapp_call_id=call_id,
                    sdp=SessionDescription(
                        type=call["session"]["sdp_type"], sdp=call["session"]["sdp"]
                    ),
                    # Block until audio is streaming, so a media failure
                    # surfaces here instead of as silence on the call.
                    wait_until_answered=True,
                )
            )
            logger.info(f"Connected outbound call {call_id}")
        except api.TwirpError as e:
            # Meta redelivers webhooks, so a second connect for the same call is expected.
            if e.code == api.TwirpErrorCode.ALREADY_EXISTS:
                logger.info(f"Call {call_id} was already connected")
            else:
                logger.error(f"Failed to connect call {call_id}: {e.code}: {e.message}")

    elif event == "terminate":
        # Tell LiveKit to clean up the connector session and the room.
        # Meta also sends this event when the business ended the call, and the
        # session is already gone in that case, so an error here is expected.
        try:
            await lkapi.connector.disconnect_whatsapp_call(
                api.DisconnectWhatsAppCallRequest(
                    whatsapp_call_id=call_id,
                    disconnect_reason=api.DisconnectWhatsAppCallRequest.USER_INITIATED,
                )
            )
            logger.info(f"Disconnected call {call_id}")
        except api.TwirpError as e:
            logger.info(f"Call {call_id} was already cleaned up: {e.message}")

    else:
        logger.warning(f"Unhandled call event {event} for {call_id}")


async def accept_call(
    lkapi: api.LiveKitAPI, call: dict, call_id: str, phone_number_id: str
) -> api.AcceptWhatsAppCallResponse | None:
    """Accept an inbound call and dispatch the agent. Returns None when the
    call was a webhook redelivery or the accept failed."""
    try:
        return await lkapi.connector.accept_whatsapp_call(
            api.AcceptWhatsAppCallRequest(
                whatsapp_phone_number_id=phone_number_id,
                whatsapp_api_key=WHATSAPP_API_KEY,
                whatsapp_cloud_api_version=WHATSAPP_CLOUD_API_VERSION,
                whatsapp_call_id=call_id,
                sdp=SessionDescription(
                    type=call["session"]["sdp_type"], sdp=call["session"]["sdp"]
                ),
                room_name=f"whatsapp-{call_id}",
                # The identity ends up in logs, so keep the phone number out of it.
                # The name field is redacted and can carry it.
                participant_identity=f"wa-{uuid.uuid4().hex[:8]}",
                participant_name=call.get("from", ""),
                agents=[RoomAgentDispatch(agent_name=AGENT_NAME)],
                wait_until_answered=True,
            )
        )
    except api.TwirpError as e:
        if e.code == api.TwirpErrorCode.ALREADY_EXISTS:
            # Meta redelivers webhooks, so a second accept for the same call is expected.
            logger.info(f"Call {call_id} was already accepted")
        else:
            logger.error(f"Failed to accept call {call_id}: {e.code}: {e.message}")
        return None


def signature_valid(raw: bytes, header: str) -> bool:
    """Check the X-Hub-Signature-256 header: HMAC-SHA256 over the raw request
    body, keyed with the Meta app secret."""
    assert WHATSAPP_APP_SECRET is not None
    expected = "sha256=" + hmac.new(WHATSAPP_APP_SECRET.encode(), raw, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header)


async def run_logged(coro: Coroutine, description: str) -> None:
    """Await a webhook-triggered task and log any unexpected failure."""
    try:
        await coro
    except Exception:
        logger.exception(f"Failed to handle {description}")


async def handle_webhook(request: web.Request) -> web.Response:
    """Meta requires a fast 200, so call handling runs as a background task."""
    raw = await request.read()

    if WHATSAPP_APP_SECRET and not signature_valid(
        raw, request.headers.get("X-Hub-Signature-256", "")
    ):
        logger.warning("Rejected webhook with a bad signature")
        return web.Response(status=403)

    try:
        body = json.loads(raw)
    except ValueError:
        # Respond 200 anyway. An error response makes Meta redeliver the same payload.
        logger.warning(f"Ignoring unparseable webhook body: {raw[:200]!r}")
        return web.Response(text="ok")

    lkapi: api.LiveKitAPI = request.app["lkapi"]

    for entry in body.get("entry", []):
        for change in entry.get("changes", []):
            if change.get("field") != "calls":
                continue
            value = change.get("value", {})
            for error in value.get("errors", []):
                logger.warning(f"WhatsApp reported an error: {error}")
            for status in value.get("statuses", []):
                logger.info(f"Status update: {status}")
            # Prefer the number ID the event arrived on; multi-number apps get several.
            phone_number_id = (
                value.get("metadata", {}).get("phone_number_id") or WHATSAPP_PHONE_NUMBER_ID or ""
            )
            for call in value.get("calls", []):
                coro = handle_call_event(lkapi, call, phone_number_id)
                task = asyncio.create_task(run_logged(coro, f"call event {call.get('id', '')}"))
                request.app["tasks"].add(task)
                task.add_done_callback(request.app["tasks"].discard)

    return web.Response(text="ok")


def build_app() -> web.Application:
    app = web.Application()
    app["tasks"] = set()
    app.router.add_get("/whatsapp/webhook", handle_verification)
    app.router.add_post("/whatsapp/webhook", handle_webhook)

    async def _lkapi_ctx(app: web.Application):
        app["lkapi"] = api.LiveKitAPI()
        yield
        await app["lkapi"].aclose()

    app.cleanup_ctx.append(_lkapi_ctx)
    return app


async def dial(to_number: str) -> None:
    """Place an outbound call. Keep the webhook server running: Meta sends the
    SDP answer there, and the server completes the connection.

    Outbound calling requires user permission and is not available in every
    country. See the Meta docs linked above.
    """
    if not (WHATSAPP_PHONE_NUMBER_ID and WHATSAPP_API_KEY):
        logger.error("Set WHATSAPP_PHONE_NUMBER_ID and WHATSAPP_API_KEY to dial")
        raise SystemExit(1)

    async with api.LiveKitAPI() as lkapi:
        try:
            res = await lkapi.connector.dial_whatsapp_call(
                api.DialWhatsAppCallRequest(
                    whatsapp_phone_number_id=WHATSAPP_PHONE_NUMBER_ID,
                    whatsapp_to_phone_number=to_number,
                    whatsapp_api_key=WHATSAPP_API_KEY,
                    whatsapp_cloud_api_version=WHATSAPP_CLOUD_API_VERSION,
                    agents=[RoomAgentDispatch(agent_name=AGENT_NAME)],
                )
            )
        except api.TwirpError as e:
            # Meta rejections ride along in the message, including the fbtrace_id.
            logger.error(f"Dial failed: {e.code}: {e.message}")
            raise SystemExit(1) from None
    logger.info(f"Dialing {to_number}: call {res.whatsapp_call_id}, room {res.room_name}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="WhatsApp connector example")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("serve", help="run the Meta webhook server")
    dial_parser = sub.add_parser("dial", help="place an outbound call")
    dial_parser.add_argument(
        "--to", required=True, help="number to call, with country code and no plus sign"
    )
    args = parser.parse_args()

    if args.command == "serve":
        if not (WHATSAPP_PHONE_NUMBER_ID and WHATSAPP_API_KEY):
            logger.warning(
                "WHATSAPP_PHONE_NUMBER_ID or WHATSAPP_API_KEY is not set; accepting calls will fail"
            )
        web.run_app(build_app(), port=PORT)
    else:
        asyncio.run(dial(args.to))


if __name__ == "__main__":
    main()
