"""A text client for the fare desk: each line you type is a turn, over A2A, with no microphone.

Start the desk first, with agent-db configured (see the README's "Persistence" section):

    python expert.py dev

then this:

    python chat.py

It creates a conversation database and a context and prints both. Pass them back to pick
the same conversation up from another run, or after the desk was killed and restarted:

    python chat.py --conversation DB_... --context chat-...

Each line is sent as a person's turn; with --delegate it is sent as an agent's instruction
instead, the way voice.py's phone agent asks. Relayed progress prints as it arrives, then
the answer and how the task ended. Ctrl-D says goodbye, which lets the desk drop the
conversation at once rather than wait for it to idle.
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
    parser.add_argument("--context", help="reuse this A2A context")
    parser.add_argument("--delegate", action="store_true", help="send lines as instructions")
    args = parser.parse_args()

    conversation_id = args.conversation
    if conversation_id is None and os.environ.get("LIVEKIT_AGENTDB_URL"):
        # the desk writes into whichever database the caller names, so the caller makes it
        agentdb = store.AgentDB.from_env()
        conversation_id = (await agentdb.service.create_database()).database_id
        await agentdb.aclose()
    context_id = args.context or shortuuid("chat-")
    print(f"conversation {conversation_id or '(not persisted)'}")
    print(f"context      {context_id}")

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
