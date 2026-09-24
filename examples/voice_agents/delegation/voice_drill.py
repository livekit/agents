"""The phone agent over text, for the crash drill: each line you type is a caller's turn.

It runs voice.py's Receptionist on a text model with the same delegate and persistence, so
killing it mid-call and starting it again on the same conversation resumes the call, a durable
collect_email included. The README's "Persistence" section has the drill.
"""

import argparse
import asyncio
import logging

from voice import DB, FARE_DESK_URL, Receptionist

from livekit.agents import AgentSession, ConversationItemAddedEvent, inference
from livekit.agents.delegation import A2ADelegate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s - %(message)s")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--conversation", help="resume this DB_... conversation")
    args = parser.parse_args()
    if DB is None:
        raise SystemExit("set LIVEKIT_AGENTDB_URL: the drill needs somewhere to persist")

    conversation_id = args.conversation or await DB.create_database()
    print(f"conversation {conversation_id}")
    session: AgentSession = AgentSession(
        llm=inference.LLM("openai/gpt-4.1-mini"),
        delegate={"delegate": A2ADelegate(FARE_DESK_URL), "announce": False},
    )

    @session.on("conversation_item_added")
    def _on_item_added(ev: ConversationItemAddedEvent) -> None:
        if ev.item.type == "message" and ev.item.role == "assistant":
            print(f"agent> {ev.item.text_content}")
        elif ev.item.type == "agent_handoff":
            print(f"  [{ev.item.old_agent_id} -> {ev.item.new_agent_id}]")

    await session.start(agent=Receptionist(), persist=DB.session(conversation_id, "voice"))
    print(f"agent    {session.current_agent.id}, {len(session.history.messages())} messages back")

    loop = asyncio.get_running_loop()
    try:
        while True:
            try:
                line = await loop.run_in_executor(None, input, "you> ")
            except EOFError:
                break
            if line.strip():
                session.generate_reply(user_input=line)
    finally:
        await session.aclose()
        await DB.aclose()


if __name__ == "__main__":
    asyncio.run(main())
