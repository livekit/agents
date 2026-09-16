"""What a session is answering, when a caller asked for it.

An ``AgentSession`` usually answers a person in a room. It can instead answer requests from
another agent or a chat client, one at a time, and then :attr:`AgentSession.request` is what
it is working on. The carrier does not matter here: a request that arrived over A2A and one
handed over in-process are the same object to the code answering it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

DirectiveKind = Literal["escalate", "end_session"]
"""What a caller is asked to do once it has said the answer.

``escalate`` hands the conversation to a human; ``end_session`` closes it. A caller that does
not act on the kind still says the answer.
"""


@dataclass
class Directive:
    """Advice carried back with the answer, for the caller to act on after saying it."""

    kind: DirectiveKind
    reason: str = ""
    """Why, in the sender's own words. Implementation-neutral: a reader that does not act on
    the reason still acts on the kind."""


@dataclass
class ServedRequest:
    """One request this session is answering for a caller.

    Read it anywhere the session is reachable, and check it before assuming there is one::

        if (request := session.request) is not None:
            request.set_directive("end_session", reason="user_request")
        else:
            await session.aclose()

    That branch is the point: a tool that ends a call has something to do either way, and
    whether a caller is waiting on an answer decides which.
    """

    metadata: dict[str, Any] = field(default_factory=dict)
    """Application data the caller attached to this request, handed over untouched."""
    directive: Directive | None = None
    """What the caller is asked to do after saying the answer, if anything."""

    def set_directive(self, kind: DirectiveKind, reason: str = "") -> None:
        """Ask the caller to act once it has said the answer.

        This sends advice and does nothing else here: the caller says the answer first and
        then decides how to act, so a directive never cuts the answer short. Only the answer
        of a request that finished successfully carries one, and the last call before the
        answer is the one that travels.
        """
        self.directive = Directive(kind, reason)
