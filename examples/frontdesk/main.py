import os

from dotenv import load_dotenv

# before the agents import: they build their models from these keys
load_dotenv()

import backend_agent  # noqa: E402
import frontend_agent  # noqa: E402

from livekit.agents import AgentServer, cli  # noqa: E402
from livekit.agents.storage import AgentDBStorage, SQLiteStorage  # noqa: E402

# AgentDB needs a LiveKit Cloud project; AGENT_STORAGE=sqlite keeps threads in a local file
storage = SQLiteStorage() if os.environ.get("AGENT_STORAGE") == "sqlite" else AgentDBStorage()
server = AgentServer(storage=storage, agent_name="dentist")
server.add(frontend_agent.front_desk, backend_agent.back_office)

if __name__ == "__main__":
    cli.run_app(server)
