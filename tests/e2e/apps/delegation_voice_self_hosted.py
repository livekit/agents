"""`examples/delegation-voice`'s agents on a self-hosted server: the worker's own listener
serves the backend, and threads stay in local SQLite."""

import backend_agent
import frontend_agent

from livekit.agents import AgentServer
from livekit.agents.storage import SQLiteStorage

server = AgentServer(storage=SQLiteStorage(), agent_name="delegation-voice")
server.add(frontend_agent.front_desk, backend_agent.back_office)
