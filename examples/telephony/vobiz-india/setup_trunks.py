"""Create the LiveKit SIP resources for an India-pinned Vobiz trunk. Run this once."""

import asyncio
import os

from dotenv import load_dotenv

from livekit import api

load_dotenv()

AGENT_NAME = "vobiz-agent"  # must match AGENT_NAME in agent.py


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise SystemExit(f"{name} is not set")
    return value


async def main() -> None:
    phone_number = require_env("VOBIZ_PHONE_NUMBER")
    sip_subdomain = require_env("LIVEKIT_SIP_SUBDOMAIN")
    sip_domain = require_env("VOBIZ_SIP_DOMAIN")
    username = require_env("VOBIZ_SIP_USERNAME")
    password = require_env("VOBIZ_SIP_PASSWORD")

    lkapi = api.LiveKitAPI()
    try:
        inbound_trunk = await lkapi.sip.create_sip_inbound_trunk(
            api.CreateSIPInboundTrunkRequest(
                trunk=api.SIPInboundTrunkInfo(name="Vobiz inbound", numbers=[phone_number])
            )
        )

        # each caller gets a new room, and the agent is dispatched into it
        dispatch_rule = await lkapi.sip.create_sip_dispatch_rule(
            api.CreateSIPDispatchRuleRequest(
                name="Vobiz inbound to agent",
                trunk_ids=[inbound_trunk.sip_trunk_id],
                rule=api.SIPDispatchRule(
                    dispatch_rule_individual=api.SIPDispatchRuleIndividual(room_prefix="call-")
                ),
                room_config=api.RoomConfiguration(
                    agents=[api.RoomAgentDispatch(agent_name=AGENT_NAME)]
                ),
            )
        )

        # destination_country makes outbound calls originate from LiveKit's India region
        outbound_trunk = await lkapi.sip.create_sip_outbound_trunk(
            api.CreateSIPOutboundTrunkRequest(
                trunk=api.SIPOutboundTrunkInfo(
                    name="Vobiz outbound",
                    address=sip_domain,
                    numbers=[phone_number],
                    auth_username=username,
                    auth_password=password,
                    destination_country="in",
                )
            )
        )
    finally:
        await lkapi.aclose()

    print(f"Inbound trunk:  {inbound_trunk.sip_trunk_id}")
    print(f"Dispatch rule:  {dispatch_rule.sip_dispatch_rule_id}")
    print(f"Outbound trunk: {outbound_trunk.sip_trunk_id}")
    print()
    print(f"Add to .env:          SIP_OUTBOUND_TRUNK_ID={outbound_trunk.sip_trunk_id}")
    print(f"Vobiz trunk Primary URI: {sip_subdomain}.india.sip.livekit.cloud")


if __name__ == "__main__":
    asyncio.run(main())
