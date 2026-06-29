"""
Outbound Call Example with Telnyx SIP Trunk

This example demonstrates how to make outbound calls using LiveKit Agents
with Telnyx as the SIP trunk provider.

Usage:
    python outbound_call.py --phone "+1xxxxxxxxxx"

Requirements:
- LiveKit Cloud account with SIP trunk configured using Telnyx
- Telnyx account with phone number and SIP connection
- Environment variables set (see README.md)
"""

import argparse
import asyncio
import logging
import os

from dotenv import load_dotenv

from livekit import api

load_dotenv()

logger = logging.getLogger("telnyx-outbound")
logger.setLevel(logging.INFO)

# Environment configuration
ROOM_NAME = "telnyx-outbound-call"
AGENT_DISPATCH_NAME = os.getenv("TELNYX_AGENT_DISPATCH_NAME", "telnyx-agent")
OUTBOUND_TRUNK_ID = os.getenv("LIVEKIT_SIP_OUTBOUND_TRUNK")
OUTBOUND_CALLER_ID = os.getenv("LIVEKIT_SIP_NUMBER")


async def make_outbound_call(phone_number: str, user_request: str) -> None:
    """Create a dispatch and initiate an outbound call through Telnyx.

    Args:
        phone_number: The phone number to call (E.164 format)
        user_request: The user request/intent for the agent
    """
    if not OUTBOUND_TRUNK_ID:
        logger.error("LIVEKIT_SIP_OUTBOUND_TRUNK environment variable not set")
        return

    if not OUTBOUND_CALLER_ID:
        logger.error("LIVEKIT_SIP_NUMBER environment variable not set")
        return

    lkapi = api.LiveKitAPI()

    try:
        # Create an agent dispatch for the call
        logger.info(f"Creating dispatch for agent {AGENT_DISPATCH_NAME} in room {ROOM_NAME}")
        dispatch = await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                agent_name=AGENT_DISPATCH_NAME,
                room=ROOM_NAME,
                metadata=user_request,
            )
        )
        logger.info(f"Created dispatch: {dispatch.dispatch_id}")

        # Create SIP participant to initiate the call through Telnyx
        masked_number = (
            f"{'*' * (len(phone_number) - 4)}{phone_number[-4:]}"
            if len(phone_number) > 4
            else "****"
        )
        logger.info(f"Initiating call to {masked_number} from {OUTBOUND_CALLER_ID}")

        sip_participant = await lkapi.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ROOM_NAME,
                sip_trunk_id=OUTBOUND_TRUNK_ID,
                sip_call_to=phone_number,
                participant_identity="telnyx_caller",
                sip_headers=[
                    # Telnyx-specific headers for caller ID
                    api.SIPHeader(name="X-Telnyx-Caller-ID", value=OUTBOUND_CALLER_ID),
                ],
            )
        )
        logger.info(f"SIP participant created: {sip_participant.participant.identity}")

    except Exception as e:
        logger.error(f"Error creating outbound call: {e}")
        raise
    finally:
        await lkapi.aclose()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Make an outbound call using Telnyx SIP trunk")
    parser.add_argument(
        "--phone",
        dest="phone_number",
        default="+14155551234",
        help="Phone number to call in E.164 format (default: +14155551234)",
    )
    parser.add_argument(
        "--request",
        dest="user_request",
        default="Customer inquiry about product availability",
        help="User request/intent passed as dispatch metadata",
    )
    args = parser.parse_args()

    logger.info(f"Making outbound call to {args.phone_number}")
    await make_outbound_call(args.phone_number, args.user_request)


if __name__ == "__main__":
    asyncio.run(main())
