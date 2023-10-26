import asyncio
import subprocess
import threading
from dataclasses import dataclass

import dotenv
import uuid
import aiohttp
import aiohttp_cors

dotenv.load_dotenv()

@dataclass
class ConnectionDetails:
    ws_url: str
    token: str

def start_web_frontend():
    def start_process():
        subprocess.Popen(["bash ./start.sh"], shell=True, cwd="./web-frontend")
    threading.Thread(target=start_process, daemon=True).start()

def generate_connection_details(identity: str) -> ConnectionDetails:
    # TODO when livekit python sdk is released with the api namespace
    pass

async def add_agent(request):
    data = await request.json()
    agent_type = data['agent']
    identity = f"{agent_type}-{uuid.uuid4()}"
    details = generate_connection_details(identity)

    # room = livekt.rtc.Room()
    # await room.connect(details.ws_url, details.token)

    if agent_type == "vad":
        pass

    return aiohttp.web.json_response({'success': True})

app = aiohttp.web.Application()
app.add_routes([aiohttp.web.post('/add_agent', add_agent)])
cors = aiohttp_cors.setup(app, defaults={"*": aiohttp_cors.ResourceOptions(
    allow_credentials=True, expose_headers="*", allow_headers="*")})
for route in list(app.router.routes()):
    cors.add(route)

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
