from __future__ import annotations

from deprecation import deprecated

from . import __version__
from ._async.client import AsyncPostgrestClient


class Client(AsyncPostgrestClient):
    """Alias to PostgrestClient."""

    @deprecated("0.2.0", "1.0.0", __version__, "Use PostgrestClient instead")
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


PostgrestClient = Client
