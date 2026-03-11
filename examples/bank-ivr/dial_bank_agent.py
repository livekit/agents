# Adapted from https://github.com/livekit-examples/python-agents-examples/blob/main/telephony/make_call/make_call.py
import argparse
import asyncio
import logging
import os

from dotenv import load_dotenv

from livekit import api

load_dotenv()

logger = logging.getLogger("make-call")
logger.setLevel(logging.INFO)

ROOM_NAME = "dtmf-agent-example"
AGENT_NAME = os.getenv("PHONE_TREE_AGENT_DISPATCH_NAME", "my-telephony-agent")
OUTBOUND_TRUNK_ID = os.getenv("SIP_OUTBOUND_TRUNK_ID")


async def call_ivr_system(phone_number: str, user_request: str) -> None:
    """Create a dispatch and add a SIP participant to call the phone number"""
    lkapi = api.LiveKitAPI()

    logger.info(f"Creating dispatch for agent {AGENT_NAME} in room {ROOM_NAME}")
    dispatch = await lkapi.agent_dispatch.create_dispatch(
        api.CreateAgentDispatchRequest(agent_name=AGENT_NAME, room=ROOM_NAME, metadata=user_request)
    )
    logger.info(f"Created dispatch: {dispatch}")

    if not OUTBOUND_TRUNK_ID or not OUTBOUND_TRUNK_ID.startswith("ST_"):
        logger.error("SIP_OUTBOUND_TRUNK_ID is not set or invalid")
        return

    # Mask all but the last 4 digits of the phone number for the log
    masked_number = (
        f"{'*' * (len(phone_number) - 4)}{phone_number[-4:]}" if len(phone_number) > 4 else "****"
    )
    logger.info(f"Dialing {masked_number} to room {ROOM_NAME}")

    try:
        sip_participant = await lkapi.sip.create_sip_participant(
            api.CreateSIPParticipantRequest(
                room_name=ROOM_NAME,
                sip_trunk_id=OUTBOUND_TRUNK_ID,
                sip_call_to=phone_number,
                participant_identity="phone_user",
            )
        )
        logger.info(f"Created SIP participant: {sip_participant}")
    except Exception as e:
        logger.error(f"Error creating SIP participant: {e}")

    await lkapi.aclose()  # type: ignore


async def main() -> None:
    parser = argparse.ArgumentParser(description="Dial an IVR system and dispatch an agent")
    parser.add_argument(
        "--phone",
        dest="phone_number",
        default="+12132896618",
        help="Phone number to dial (default: +12132896618)",
    )
    parser.add_argument(
        "--request",
        dest="user_request",
        default="check balance for all accounts I have",
        help=(
            "User request/intent passed as dispatch metadata "
            "(default: 'check balance for all accounts I have')"
        ),
    )
    parser.add_argument(
        "--id",
        dest="customer_id",
        default="10000001",
        help="Customer ID to use for authentication (default: 10000001)",
    )
    parser.add_argument(
        "--pin",
        dest="pin",
        default="0000",
        help="PIN to use for authentication (default: 0000)",
    )
    args = parser.parse_args()

    user_request = (
        args.user_request
        + f'\n\nUse account number "{" ".join(args.customer_id)}" and PIN "{" ".join(args.pin)}" to authenticate and navigate the IVR.'
    )

    print(f"==> User request: {user_request}")
    await call_ivr_system(
        args.phone_number,
        user_request,
    )


if __name__ == "__main__":
    asyncio.run(main())
