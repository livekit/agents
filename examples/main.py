import os
import asyncio
import subprocess
import threading
import livekit.agents as lkagents
import livekit.api as lkapi
from dataclasses import dataclass
from agents.vad.vad import vad_agent
from agents.stt.stt import stt_agent
import dotenv
import uuid
import aiohttp
import aiohttp_cors

dotenv.load_dotenv()

ws_url = os.environ.get("LIVEKIT_WS_URL")
api_key = os.environ.get("LIVEKIT_API_KEY")
api_secret = os.environ.get("LIVEKIT_API_SECRET")

print(f"ws_url: {ws_url} - api_key{api_key} - api_secret{api_secret}")


async def vad_job_available_cb(worker, job):
    print("Accepting vad job")
    await job.accept(agent=vad_agent)


async def stt_job_available_cb(worker, job):
    print("Accepting stt job")
    await job.accept(agent=stt_agent)

workers = {
    "vad": lkagents.ManualWorker(ws_url=ws_url,
                                 api_key=api_key,
                                 api_secret=api_secret,
                                 handler=lkagents.Worker.Handler(
                                     agent_identity_generator=lambda room: f"vad-{uuid.uuid4()}",
                                     job_available_cb=vad_job_available_cb,
                                 )),
    "stt": lkagents.ManualWorker(ws_url=ws_url,
                                 api_key=api_key,
                                 api_secret=api_secret,
                                 handler=lkagents.Worker.Handler(
                                     agent_identity_generator=lambda room: f"stt-{uuid.uuid4()}",
                                     job_available_cb=stt_job_available_cb,
                                 )),
}


@dataclass
class ConnectionDetails:
    ws_url: str
    token: str


def start_web_frontend():
    def start_process():
        subprocess.Popen(["yarn && yarn dev"],
                         shell=True, cwd="./web-frontend")
    threading.Thread(target=start_process, daemon=True).start()


async def add_agent(request):
    data = await request.json()
    agent_type = data['type']

    await workers[agent_type].simulate_job(
        job_type=lkagents.JobType.JT_ROOM, room=data.get("room"))

    return aiohttp.web.json_response({})


async def generate_connection_details(request):
    data = await request.json()
    room = data['room']
    identity = data['identity']
    token = lkapi.AccessToken(api_key, api_secret).with_identity(identity).with_grants(
        lkapi.VideoGrants(room=room, can_publish=True, can_subscribe=True, can_publish_data=True, room_join=True, room_create=True))
    return aiohttp.web.json_response(ConnectionDetails(ws_url=ws_url, token=token.to_jwt()).__dict__)

app = aiohttp.web.Application()
app.add_routes([aiohttp.web.post('/add_agent', add_agent)])
app.add_routes(
    [aiohttp.web.post('/generate_connection_details', generate_connection_details)])
cors = aiohttp_cors.setup(app, defaults={"*": aiohttp_cors.ResourceOptions(
    allow_credentials=True, expose_headers="*", allow_headers="*")})
for route in list(app.router.routes()):
    cors.add(route, {"localhost:3000": aiohttp_cors.ResourceOptions()})


async def main():
    start_web_frontend()
    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(runner, host='0.0.0.0', port=8000)
    await site.start()
    await asyncio.Event().wait()

if __name__ == "__main__":
    main_task = asyncio.ensure_future(main())
    asyncio.get_event_loop().run_until_complete(main_task)
