from dotenv import load_dotenv

from livekit.agents import Agent, AgentServer, RunResult, TextSession, cli, inference
from livekit.agents.a2a import A2ASessionContext
from livekit.agents.storage import AgentDBStorage

load_dotenv()

assistant = Agent(
    instructions="You are a friendly, helpful assistant in a web chat. Answer clearly and "
    "briefly, in plain text.",
    llm=inference.LLM(model="google/gemma-4-31b-it"),
)

server = AgentServer(storage=AgentDBStorage(), agent_name="web-chat")


# the text door: one turn per message, the history kept in the message's thread
@server.a2a_session(endpoint="chat", description="A general chat assistant.")
async def chat(ctx: A2ASessionContext) -> RunResult[str]:
    turn = await TextSession(agent=assistant).run(ctx.message, context=ctx.store.bind())
    return turn.result


if __name__ == "__main__":
    cli.run_app(server)
