import asyncio
import logging
import subprocess
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from livekit.agents.avatar import AvatarConnectionInfo

logger = logging.getLogger("avatar-dispatcher")
logging.basicConfig(level=logging.INFO)

THIS_DIR = Path(__file__).parent.absolute()


class WorkerLauncher:
    """Local implementation that launches workers as subprocesses"""

    @dataclass
    class WorkerInfo:
        room_name: str
        process: subprocess.Popen

    def __init__(self):
        self.workers: dict[str, WorkerLauncher.WorkerInfo] = {}
        self._monitor_task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        self._monitor_task = asyncio.create_task(self._monitor())

    def close(self) -> None:
        if self._monitor_task:
            self._monitor_task.cancel()

        for worker in self.workers.values():
            worker.process.terminate()
            try:
                worker.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.process.kill()

    async def launch_worker(self, connection_info: AvatarConnectionInfo) -> None:
        if connection_info.room_name in self.workers:
            worker = self.workers[connection_info.room_name]
            worker.process.terminate()
            try:
                worker.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.process.kill()

        # Launch new worker process
        cmd = [
            sys.executable,
            str(THIS_DIR / "avatar_worker.py"),
            "--url",
            connection_info.url,
            "--token",
            connection_info.token,
            "--room",
            connection_info.room_name,
        ]

        try:
            room_name = connection_info.room_name
            process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
            self.workers[room_name] = WorkerLauncher.WorkerInfo(
                room_name=room_name, process=process
            )
            logger.info(f"Launched avatar worker for room: {room_name}")
        except Exception as e:
            logger.error(f"Failed to launch worker: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def _monitor(self) -> None:
        while True:
            for worker in list(self.workers.values()):
                if worker.process.poll() is not None:
                    logger.info(
                        f"Worker for room {worker.room_name} exited with code {worker.process.returncode}"
                    )
                    self.workers.pop(worker.room_name)
            await asyncio.sleep(1)


class AvatarDispatcher:
    def __init__(self):
        self.launcher = WorkerLauncher()

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await self.launcher.start()
            yield
            self.launcher.close()

        self.app = FastAPI(title="Avatar Dispatcher", lifespan=lifespan)
        self.app.post("/launch")(self.handle_launch)

    async def handle_launch(self, connection_info: AvatarConnectionInfo) -> dict:
        """Handle request to launch an avatar worker"""
        try:
            await self.launcher.launch_worker(connection_info)
            return {
                "status": "success",
                "message": f"Avatar worker launched for room: {connection_info.room_name}",
            }
        except Exception as e:
            logger.error(f"Error handling launch request: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to launch worker: {str(e)}"
            )


def run_server(host: str = "0.0.0.0", port: int = 8089):
    dispatcher = AvatarDispatcher()
    uvicorn.run(dispatcher.app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host to run server on")
    parser.add_argument("--port", default=8089, help="Port to run server on")
    args = parser.parse_args()
    run_server(args.host, args.port)
