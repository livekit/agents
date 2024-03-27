from __future__ import annotations

from deprecation import deprecated

from . import __version__
from ._async.request_builder import AsyncSelectRequestBuilder


class GetRequestBuilder(AsyncSelectRequestBuilder):
    """Alias to SelectRequestBuilder."""

    @deprecated("0.4.0", "1.0.0", __version__, "Use SelectRequestBuilder instead")
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
