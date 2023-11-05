from livekit import agents
import asyncio

worker = agents.Worker(agents.JobType.JT_ROOM, "ws://localhost:7880", "", "")
asyncio.get_event_loop().run_until_complete(worker.run())
