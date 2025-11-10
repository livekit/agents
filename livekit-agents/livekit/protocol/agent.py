"""
Shim: livekit.protocol.agent

Expanded placeholder shim so tests can import JobType members expected by the code.
Add more members if pytest complains about missing ones.
"""
from enum import Enum
from typing import Any, Dict, Optional

class JobType(Enum):
    # common job types used across livekit-agents code
    JT_ROOM = "JT_ROOM"
    JT_SIMULATE = "JT_SIMULATE"
    JT_JOB = "JT_JOB"
    JT_PUBLISHER = "JT_PUBLISHER"
    JT_SUBSCRIBER = "JT_SUBSCRIBER"
    JT_WORKER = "JT_WORKER"
    JT_CUSTOM = "JT_CUSTOM"
    # add more if tests require them

class Agent:
    """Placeholder Agent class used by code that imports this module"""
    def __init__(self, *args, **kwargs):
        self._kwargs = kwargs

class WorkerOptions:
    """Placeholder WorkerOptions - expand fields if runtime needs them"""
    def __init__(self, *args, **kwargs):
        self._kwargs = kwargs

class SimulateJobInfo:
    """Placeholder SimulateJobInfo"""
    def __init__(self, *args, **kwargs):
        self._kwargs = kwargs

class RunningJobInfo:
    def __init__(self, job_id: Optional[str] = None, **kwargs):
        self.job_id = job_id
        self._kwargs = kwargs

class JobAcceptArguments:
    def __init__(self, *args, **kwargs):
        self._kwargs = kwargs

__all__ = ["JobType", "Agent", "WorkerOptions", "SimulateJobInfo", "RunningJobInfo", "JobAcceptArguments"]
