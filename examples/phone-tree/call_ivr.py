import asyncio
import logging
import os

from dotenv import load_dotenv

from livekit import api

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger("make-call")
logger.setLevel(logging.INFO)

# Configuration
ROOM_NAME = "phone-tree-example"
AGENT_NAME = os.getenv("PHONE_TREE_AGENT_DISPATCH_NAME", "my-phone-tree-agent")
OUTBOUND_TRUNK_ID = os.getenv("SIP_OUTBOUND_TRUNK_ID")


async def call_ivr_system(phone_number: str) -> None:
    """Create a dispatch and add a SIP participant to call the phone number"""
    lkapi = api.LiveKitAPI()

    # Create agent dispatch
    logger.info(f"Creating dispatch for agent {AGENT_NAME} in room {ROOM_NAME}")
    dispatch = await lkapi.agent_dispatch.create_dispatch(
        api.CreateAgentDispatchRequest(agent_name=AGENT_NAME, room=ROOM_NAME, metadata=phone_number)
    )
    logger.info(f"Created dispatch: {dispatch}")

    # Create SIP participant to make the call
    if not OUTBOUND_TRUNK_ID or not OUTBOUND_TRUNK_ID.startswith("ST_"):
        logger.error("SIP_OUTBOUND_TRUNK_ID is not set or invalid")
        return

    # Mask all but the last 4 digits of the phone number for the log
    masked_number = f"{'*' * (len(phone_number) - 4)}{phone_number[-4:]}" if len(phone_number) > 4 else "****"
    logger.info(f"Dialing {masked_number} to room {ROOM_NAME}")

    try:
        # Create SIP participant to initiate the call
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

    # Close API connection
    await lkapi.aclose()  # type: ignore


async def main() -> None:
    # Replace with the actual phone number including country code
    phone_number = "+15625738475"
    await call_ivr_system(phone_number)


if __name__ == "__main__":
    asyncio.run(main())
