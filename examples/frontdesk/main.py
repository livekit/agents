from dotenv import load_dotenv

# before the agents import: they build their models from these keys
load_dotenv()

import backend_agent  # noqa: E402
import frontend_agent  # noqa: E402

from livekit.agents import AgentServer, cli  # noqa: E402
from livekit.agents.storage import AgentDBStorage  # noqa: E402

# one worker for both: the front desk on its lane, the back office on its own
server = AgentServer(storage=AgentDBStorage(), agent_name="dentist")
server.add(frontend_agent.front_desk, backend_agent.back_office)

if __name__ == "__main__":
    cli.run_app(server)
