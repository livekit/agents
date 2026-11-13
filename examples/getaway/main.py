import flights_agent
import frontend_agent
import hotels_agent
import payments_agent

from livekit.agents import AgentServer, cli
from livekit.agents.storage import AgentDBStorage

# one worker for all four: the concierge on its lane, each desk on its own
server = AgentServer(storage=AgentDBStorage(), agent_name="getaway")
server.add(frontend_agent.concierge, flights_agent.desk, hotels_agent.desk, payments_agent.desk)

if __name__ == "__main__":
    cli.run_app(server)
