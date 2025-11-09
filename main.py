from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

from livekit import agents
from livekit.agents import WorkerOptions
from salescode_interrupt_handler.example_agent import entrypoint


if __name__ == "__main__":
    # optional sanity check (just to confirm .env is loading)
    print("LIVEKIT_WS_URL:", os.getenv("LIVEKIT_WS_URL"))
    
    agents.cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
        )
    )

