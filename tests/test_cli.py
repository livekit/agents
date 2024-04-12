import os
import shutil
import subprocess
import sys

AGENT_PY = """
from livekit.agents import (
    JobContext,
    JobRequest,
    WorkerOptions,
    cli,
)
from livekit.plugins.silero import VAD

async def request_fnc(req: JobRequest) -> None:
    pass

cli.run_app(WorkerOptions(request_fnc=request_fnc))
"""


# TODO: maybe better to have a test plugin that can be imported that doesn't
# require downloading a file, and instead just writes a file to the filesystem
async def test_download_files():
    with open("/tmp/agent.py", "w") as f:
        f.write(AGENT_PY)

    args = [sys.executable, "/tmp/agent.py", "download-files"]

    # make tmp directory if it doesn't exist
    if not os.path.exists("/tmp"):
        os.makedirs("/tmp")

    # Remove the TORCH_HOME directory
    if os.path.exists("/tmp/torch_home"):
        shutil.rmtree("/tmp/torch_home")

    subprocess.run(args, check=True, env={"TORCH_HOME": "/tmp/torch_home"})

    # Make sure something go downloaded
    assert len(os.listdir("/tmp/torch_home")) > 0
