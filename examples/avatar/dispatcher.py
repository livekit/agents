import asyncio
import logging
import subprocess
import sys
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from livekit.agents.avatar import AvatarConnectionInfo

logger = logging.getLogger(__name__)

THIS_DIR = Path(__file__).parent.absolute()


@dataclass
class WorkerInfo:
    room_name: str
    process: Optional[subprocess.Popen] = None  # Only used by local launcher


class WorkerLauncher(ABC):
    """Abstract base class for launching avatar workers"""

    @abstractmethod
    async def launch_worker(self, connection_info: AvatarConnectionInfo) -> None:
        """Launch a new avatar worker"""
        pass

    @abstractmethod
    async def cleanup_worker(self, room_name: str) -> None:
        """Cleanup a worker for a given room"""
        pass


class LocalWorkerLauncher(WorkerLauncher):
    """Local implementation that launches workers as subprocesses"""

    def __init__(self, log_level: str = "INFO"):
        self.workers: Dict[str, WorkerInfo] = {}
        self.log_level = log_level

    async def launch_worker(self, connection_info: AvatarConnectionInfo) -> None:
        # Cleanup existing worker if any
        await self.cleanup_worker(connection_info.room_name)

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
            "--log-level",
            self.log_level,
        ]

        try:
            room_name = connection_info.room_name
            process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
            self.workers[room_name] = WorkerInfo(room_name=room_name, process=process)
            logger.info(f"Launched avatar worker for room: {room_name}")

            # Monitor process in background
            asyncio.create_task(self._monitor_process(room_name))

        except Exception as e:
            logger.error(f"Failed to launch worker: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def cleanup_worker(self, room_name: str) -> None:
        worker = self.workers.get(room_name)
        if worker and worker.process:
            logger.info(f"Cleaning up worker for room: {room_name}")
            worker.process.terminate()
            try:
                worker.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.process.kill()
            self.workers.pop(room_name)

    async def _monitor_process(self, room_name: str) -> None:
        """Monitor worker process and cleanup when it exits"""
        worker = self.workers.get(room_name)
        if not worker or not worker.process:
            return

        # Wait for process to exit
        while True:
            if worker.process.poll() is not None:
                # Process exited
                await self.cleanup_worker(room_name)
                logger.info(
                    f"Worker for room {room_name} exited with code {worker.process.returncode}"
                )
                break
            await asyncio.sleep(1)


class AvatarDispatcher:
    def __init__(self, debug: bool = False):
        self.launcher: WorkerLauncher = LocalWorkerLauncher(
            log_level="DEBUG" if debug else "INFO"
        )

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            yield
            # Cleanup on shutdown
            if isinstance(self.launcher, LocalWorkerLauncher):
                for room_name in list(self.launcher.workers.keys()):
                    await self.launcher.cleanup_worker(room_name)

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


def create_app() -> FastAPI:
    dispatcher = AvatarDispatcher()
    return dispatcher.app


def run_server(host: str = "0.0.0.0", port: int = 8089):
    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host to run server on")
    parser.add_argument("--port", default=8089, help="Port to run server on")
    args = parser.parse_args()
    run_server(args.host, args.port)
