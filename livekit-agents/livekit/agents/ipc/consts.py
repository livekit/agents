from dataclasses import dataclass
from typing import Callable

IPC_PORT = 2003
MAX_PACKET_SIZE = 1024 * 16
START_TIMEOUT = 5


@dataclass
class JobMainArgs:
    job_id: str
    url: str
    token: str
    target: Callable
