import logging

from dotenv import load_dotenv

from livekit.agents import Agent, AgentSession, JobContext, WorkerOptions, cli, function_tool
from livekit.plugins import openai

# This demo defines an agent using a raw function tool to open predefined gates via enum input.
# When using raw function tools, compatibility across LLM providers is not guaranteed,
# as different models may interpret or format raw schemas differently.

logger = logging.getLogger("raw-function-description")

load_dotenv()


class RawFunctionAgent(Agent):
    def __init__(self):
        super().__init__(instructions="You are a helpful assistant")

    @function_tool(
        raw_schema={
            "name": "open_gate",
            "description": "Opens a specified gate from a predefined set of access points.",
            "parameters": {
                "type": "object",
                "properties": {
                    "gate_id": {
                        "type": "string",
                        "description": (
                            "Identifier of the gate to open. Must be one of the "
                            "system's predefined access points."
                        ),
                        "enum": [
                            "main_entrance",
                            "north_parking",
                            "loading_dock",
                            "side_gate",
                            "service_entry",
                        ],
                    }
                },
                "required": ["gate_id"],
            },
        }
    )
    async def open_gate(self, raw_arguments: dict[str, object]):
        gate_id = raw_arguments["gate_id"]
        logger.info(f"Opening gate: {gate_id}")
        return f"Gate {gate_id} opened successfully"


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    session = AgentSession(llm=openai.realtime.RealtimeModel())
    await session.start(RawFunctionAgent(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
