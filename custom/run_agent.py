import os
import sys
from dotenv import load_dotenv
from livekit import agents

# ----------------------------------------------------
# 1. Load .env FIRST (correct order)
# ----------------------------------------------------
load_dotenv()

# ----------------------------------------------------
# 2. Make project root importable so "custom.*" works
# ----------------------------------------------------
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

# ----------------------------------------------------
# 3. Now safe to import your agent entrypoint
# ----------------------------------------------------
from custom.agent_runner import entrypoint

# ----------------------------------------------------
# 4. Start worker
# ----------------------------------------------------
if __name__ == "__main__":
    opts = agents.WorkerOptions(
        entrypoint_fnc=entrypoint,
    )

    agents.cli.run_app(opts)
