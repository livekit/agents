"""Handing reasoning and tool use to an expert.

The conversation model keeps the person talking and gains one tool, the expert, attached as
``Agent(tools=[A2ADelegate(url)])``. What needs reasoning, lookups or actions goes to the
expert, which answers once; the model phrases the answer. The expert is an A2A server, an
endpoint of this project or anyone's.
"""

from .a2a import A2ADelegate
from .delegate import (
    CONTEXT_ID_KEY,
    ENDPOINT_KEY,
    PROFILE_KEY,
    TASK_ID_KEY,
    TRANSPORT_KEY,
    Delegate,
    DelegationFailed,
    DelegationResult,
    record_delegation,
)

__all__ = [
    "TASK_ID_KEY",
    "CONTEXT_ID_KEY",
    "ENDPOINT_KEY",
    "TRANSPORT_KEY",
    "PROFILE_KEY",
    "record_delegation",
    "A2ADelegate",
    "Delegate",
    "DelegationFailed",
    "DelegationResult",
]
