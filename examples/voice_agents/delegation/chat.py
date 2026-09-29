"""A text client for the fare desk: each line you type is a turn, over A2A, with no microphone.

Run it against ``python expert.py dev``. The desk is the only agent here, so its session is the
conversation's front session and the conversation id doubles as the A2A context id: one id to
keep. Ending the input says goodbye, which closes the context and saves it; the README's
"Persistence" section has the drill.
"""

import argparse
import asyncio
import os

from dotenv import load_dotenv

from livekit.agents import store
from livekit.agents.a2a import A2AClient, TaskInput
from livekit.agents.utils import shortuuid

load_dotenv()


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--url", default="http://localhost:8321/fare-desk")
    parser.add_argument("--conversation", help="reuse this DB_... conversation")
    parser.add_argument("--delegate", action="store_true", help="send lines as instructions")
    args = parser.parse_args()

    conversation_id = args.conversation
    if conversation_id is None and (url := os.environ.get("LIVEKIT_AGENTDB_URL")):
        # the desk persists into whichever conversation the caller names, so the caller mints it
        # todo: devLocal should accept the project key; until then a local agent-db takes its own
        local_key = {"api_key": "devkey", "api_secret": "secret"} if "localhost" in url else {}
        db = store.AgentDB(ws_url=os.environ.get("LIVEKIT_AGENTDB_WS_URL"), **local_key)
        conversation_id = await db.create_database()
        await db.aclose()
    # with no store the id names only the context, and nothing outlives the desk's memory of it
    context_id = conversation_id or shortuuid("chat-")
    print(f"conversation {conversation_id or '(not persisted)'}")

    client = A2AClient(args.url, context_id=context_id)
    loop = asyncio.get_running_loop()
    try:
        while True:
            try:
                line = await loop.run_in_executor(None, input, "you> ")
            except EOFError:
                break
            if not line.strip():
                continue
            task_input = (
                TaskInput(instruction=line, conversation_id=conversation_id)
                if args.delegate
                else TaskInput(text=line, conversation_id=conversation_id)
            )
            try:
                async with client.send(task_input) as stream:
                    async for update in stream:
                        if update.state == "working":
                            if update.text:
                                print(f"  … {update.text}")
                            continue
                        print(f"desk> {update.text}")
                        print(f"  [{update.state}]")
            except Exception as e:
                # a desk that is down, or restarting, fails this turn and not the client
                print(f"  [unreachable: {e!r}]")
    finally:
        await client.aclose()


if __name__ == "__main__":
    asyncio.run(main())
