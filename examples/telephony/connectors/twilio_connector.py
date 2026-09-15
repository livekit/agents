"""Connect Twilio phone calls to a LiveKit agent with the Twilio connector.

The connector uses Twilio Media Streams instead of a SIP trunk. Your webhook
answers Twilio's request with TwiML that points at a LiveKit WebSocket URL,
and the call joins a LiveKit room as a regular participant.

This example reuses the DTMF agent from ../basic_dtmf_agent.py. Run that agent
in another terminal, then run this server. See README.md for setup.

Docs: https://docs.livekit.io/telephony/connectors/twilio/
"""

import argparse
import asyncio
import base64
import hashlib
import hmac
import logging
import os
import re
import time
from collections.abc import Mapping

from aiohttp import BasicAuth, ClientSession, web
from dotenv import load_dotenv

from livekit import api
from livekit.protocol.agent_dispatch import RoomAgentDispatch

load_dotenv()

logger = logging.getLogger("twilio-connector-example")
logger.setLevel(logging.INFO)

# Must match the dispatch name the agent registered with.
AGENT_NAME = os.getenv("DTMF_AGENT_DISPATCH_NAME", "my-telephony-agent")
PORT = int(os.getenv("PORT", "8080"))

# Twilio credentials are needed for the `dial` command. When the auth token is
# set, inbound webhook signatures are verified with it too.
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_FROM_NUMBER = os.getenv("TWILIO_FROM_NUMBER")
# The exact public URL configured in the Twilio console. Twilio signs this
# URL, so signature verification needs it verbatim.
TWILIO_WEBHOOK_URL = os.getenv("TWILIO_WEBHOOK_URL")

TWIML = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Connect>
        <Stream url="{connect_url}" />
    </Connect>
</Response>"""


def mask(number: str) -> str:
    """Keep phone numbers out of logs, matching the other telephony examples."""
    return f"...{number[-4:]}" if len(number) > 4 else "****"


def redact_numbers(text: str) -> str:
    """Mask phone-number-like digit runs inside provider payloads before logging."""
    return re.sub(r"\+?\d{7,15}", lambda m: mask(m.group()), text)


FAILURE_TWIML = """<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Say>We are unable to connect your call right now. Please try again later.</Say>
</Response>"""


def twilio_signature_valid(request: web.Request, form: Mapping[str, str]) -> bool:
    """Check the X-Twilio-Signature header: HMAC-SHA1 over the public URL
    followed by the sorted form parameters, keyed with the auth token."""
    assert TWILIO_AUTH_TOKEN and TWILIO_WEBHOOK_URL
    payload = TWILIO_WEBHOOK_URL + "".join(k + form[k] for k in sorted(form.keys()))
    expected = base64.b64encode(
        hmac.new(TWILIO_AUTH_TOKEN.encode(), payload.encode(), hashlib.sha1).digest()
    ).decode()
    return hmac.compare_digest(expected, request.headers.get("X-Twilio-Signature", ""))


async def handle_voice_webhook(request: web.Request) -> web.Response:
    """Answer Twilio's inbound call webhook with TwiML that bridges the call
    into a LiveKit room and dispatches the agent."""
    form = await request.post()

    if request.app["verify_signatures"] and not twilio_signature_valid(request, form):
        logger.warning("Rejected webhook with a bad signature")
        return web.Response(status=403)

    call_sid = form.get("CallSid", "")
    caller = form.get("From", "")
    logger.info(f"Inbound call {call_sid} from {mask(caller)}")

    lkapi: api.LiveKitAPI = request.app["lkapi"]
    try:
        res = await lkapi.connector.connect_twilio_call(
            api.ConnectTwilioCallRequest(
                twilio_call_direction=api.ConnectTwilioCallRequest.TWILIO_CALL_DIRECTION_INBOUND,
                room_name=f"call-{call_sid}",
                participant_identity=caller,
                participant_name=caller,
                agents=[RoomAgentDispatch(agent_name=AGENT_NAME)],
            )
        )
    except api.TwirpError as e:
        # Answer with TwiML either way: an HTTP error plays an error tone to the caller.
        logger.error(f"Connector rejected call {call_sid}: {e.code}: {redact_numbers(e.message)}")
        return web.Response(text=FAILURE_TWIML, content_type="text/xml")
    except Exception:
        logger.exception(f"Failed to connect call {call_sid}")
        return web.Response(text=FAILURE_TWIML, content_type="text/xml")

    logger.info(f"Bridging call {call_sid} into room call-{call_sid}")
    return web.Response(text=TWIML.format(connect_url=res.connect_url), content_type="text/xml")


def build_app(verify_signatures: bool) -> web.Application:
    app = web.Application()
    app["verify_signatures"] = verify_signatures
    app.router.add_post("/twilio/voice", handle_voice_webhook)

    async def _lkapi_ctx(app: web.Application):
        app["lkapi"] = api.LiveKitAPI()
        yield
        await app["lkapi"].aclose()

    app.cleanup_ctx.append(_lkapi_ctx)
    return app


async def dial(to_number: str) -> None:
    """Place an outbound call through the connector.

    The flow has two steps. First, ConnectTwilioCall returns a WebSocket URL
    and pre-joins the room. The connector participant stays hidden until the
    callee answers. Second, the Twilio REST API creates the call with the
    HTTPS form of that URL, which returns the TwiML above when Twilio fetches it.
    """
    if not (TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN and TWILIO_FROM_NUMBER):
        logger.error("Set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, and TWILIO_FROM_NUMBER to dial")
        raise SystemExit(1)

    room_name = f"call-out-{int(time.time())}"
    async with api.LiveKitAPI() as lkapi:
        try:
            res = await lkapi.connector.connect_twilio_call(
                api.ConnectTwilioCallRequest(
                    twilio_call_direction=api.ConnectTwilioCallRequest.TWILIO_CALL_DIRECTION_OUTBOUND,
                    room_name=room_name,
                    participant_identity=to_number,
                    agents=[RoomAgentDispatch(agent_name=AGENT_NAME)],
                )
            )
        except api.TwirpError as e:
            logger.error(f"Connector rejected the call: {e.code}: {redact_numbers(e.message)}")
            raise SystemExit(1) from None

    # Twilio fetches TwiML over HTTPS from the same single-use URL.
    twiml_url = res.connect_url.replace("wss://", "https://", 1)

    async with ClientSession() as session:
        resp = await session.post(
            f"https://api.twilio.com/2010-04-01/Accounts/{TWILIO_ACCOUNT_SID}/Calls.json",
            data={"To": to_number, "From": TWILIO_FROM_NUMBER, "Url": twiml_url},
            auth=BasicAuth(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN),
        )
        body = await resp.json()
        if resp.status >= 400:
            logger.error(f"Twilio call creation failed: {redact_numbers(str(body))}")
            raise SystemExit(1)
        logger.info(f"Dialing {mask(to_number)}, Twilio call SID {body['sid']}, room {room_name}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Twilio connector example")
    sub = parser.add_subparsers(dest="command", required=True)
    serve_parser = sub.add_parser("serve", help="run the inbound-call webhook server")
    serve_parser.add_argument(
        "--allow-unverified",
        action="store_true",
        help="run without webhook signature verification (local testing only)",
    )
    dial_parser = sub.add_parser("dial", help="place an outbound call")
    dial_parser.add_argument("--to", required=True, help="number to call, E.164 format")
    args = parser.parse_args()

    if args.command == "serve":
        verify = bool(TWILIO_AUTH_TOKEN and TWILIO_WEBHOOK_URL) and not args.allow_unverified
        if not verify:
            if not args.allow_unverified:
                logger.error(
                    "Set TWILIO_AUTH_TOKEN and TWILIO_WEBHOOK_URL to verify webhook"
                    " signatures, or pass --allow-unverified for local testing"
                )
                raise SystemExit(1)
            logger.warning("Webhook signature verification is disabled")
        web.run_app(build_app(verify), port=PORT)
    else:
        asyncio.run(dial(args.to))


if __name__ == "__main__":
    main()
