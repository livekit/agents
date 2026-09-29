"""What a session is answering, when a caller asked for it rather than a person in a room.

The carrier does not matter here: a request that arrived over A2A and one handed over
in-process are the same object to the code answering it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

DirectiveKind = Literal["escalate", "end_session"]
"""What a caller is asked to do once it has said the answer: hand the conversation to a
human, or close it."""


@dataclass
class Directive:
    """Advice carried back with the answer, for the caller to act on after saying it."""

    kind: DirectiveKind
    reason: str = ""
    """Why, in the sender's own words; a reader that ignores it still acts on the kind."""


@dataclass
class ServedRequest:
    """One request a session is answering for a caller.

    Reached from the speech that answers it, and checked before it is assumed to be there::

        if (request := ctx.request) is not None:
            request.set_directive("end_session", reason="user_request")
        else:
            await ctx.session.aclose()

    That branch is the point: a tool that ends a call has something to do either way, and
    whether a caller is waiting on an answer decides which.
    """

    metadata: dict[str, Any] = field(default_factory=dict)
    """Application data the caller attached to this request, handed over untouched."""
    is_delegation: bool = False
    """Whether an agent asked for this, rather than a person taking their turn.

    An agent relays what a tool reports in the tool's own words, so this session's model
    stays out of it; a person's turn is answered by this session's model as usual.
    """
    directive: Directive | None = None
    """What the caller is asked to do after saying the answer, if anything."""

    def set_directive(self, kind: DirectiveKind, reason: str = "") -> None:
        """Ask the caller to act once it has said the answer.

        Advice and nothing else here, so a directive never cuts the answer short; only a
        request that finished successfully carries one, and the last one set travels.
        """
        self.directive = Directive(kind, reason)
