import backend_agent
import frontend_agent

from livekit.agents import AgentServer, cli
from livekit.agents.storage import AgentDBStorage

# one worker for both: the front desk on its lane, the back office on its own
server = AgentServer(storage=AgentDBStorage(), agent_name="dentist")
server.add(frontend_agent.front_desk, backend_agent.back_office)

if __name__ == "__main__":
    cli.run_app(server)
