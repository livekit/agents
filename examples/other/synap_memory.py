"""Synap memory integration for LiveKit Agents.

Demonstrates three patterns for giving a LiveKit voice agent persistent,
cross-session memory via Synap (https://maximem.ai):

1. preload_synap_context    — inject long-term memory before the call starts
2. attach_synap_recording   — record every turn automatically
3. synap_search_tool /
   synap_store_tool         — LLM-callable tools for explicit memory search and store

Install: pip install maximem-synap-livekit-agents
Get an API key at synap.maximem.ai
Open source: https://github.com/maximem-ai/maximem_synap_sdk/tree/main/packages/integrations
"""

import os

from dotenv import load_dotenv
from maximem_synap import MaximemSynapSDK
from synap_livekit_agents import (
    attach_synap_recording,
    preload_synap_context,
    synap_search_tool,
    synap_store_tool,
)

from livekit.agents import Agent, AgentSession, ChatContext
from livekit.plugins import deepgram, openai, silero

load_dotenv()

sdk = MaximemSynapSDK(api_key=os.environ["SYNAP_API_KEY"])

# In production, derive USER_ID from the room participant identity or JWT claims
# so each caller gets their own Synap memory namespace.
USER_ID = "demo-user-001"
CUSTOMER_ID = "demo-customer"


async def entrypoint(ctx):
    await ctx.connect()

    chat_ctx = ChatContext()

    # 1. Inject long-term memory before the first LLM turn
    await preload_synap_context(
        chat_ctx,
        sdk,
        user_id=USER_ID,
        customer_id=CUSTOMER_ID,
    )

    agent = Agent(
        instructions=(
            "You are a helpful voice assistant with long-term memory. "
            "Use synap_search to recall user preferences when relevant. "
            "Use synap_store to save important new facts the user mentions."
        ),
        chat_ctx=chat_ctx,
        tools=[
            synap_search_tool(sdk=sdk, user_id=USER_ID, customer_id=CUSTOMER_ID),
            synap_store_tool(sdk=sdk, user_id=USER_ID, customer_id=CUSTOMER_ID),
        ],
    )

    session = AgentSession(
        llm=openai.LLM(model="gpt-4o"),
        stt=deepgram.STT(),
        tts=openai.TTS(),
        vad=silero.VAD.load(),
    )

    # 2. Automatically record every user transcription and assistant response
    attach_synap_recording(session, sdk, user_id=USER_ID, customer_id=CUSTOMER_ID)

    await session.start(
        agent=agent,
        room=ctx.room,
    )


if __name__ == "__main__":
    from livekit.agents import WorkerOptions, cli

    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
