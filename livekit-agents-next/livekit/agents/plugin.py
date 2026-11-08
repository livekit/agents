"""The ``Plugin`` a provider package registers itself with (docs/models.md, "Plugin authors").

The server reads the registry: it logs each registered plugin at start, and the job processes'
forkserver imports each plugin's package and its lazy exports once, ahead of any job.
"""

from __future__ import annotations

import logging
import threading


class Plugin:
    """A provider package, registered on import with ``Plugin.register_plugin(...)``."""

    registered_plugins: list[Plugin] = []

    def __init__(
        self,
        title: str,
        version: str,
        package: str,
        logger: logging.Logger | None = None,
    ) -> None:
        self._title = title
        self._version = version
        self._package = package
        self._logger = logger

    @classmethod
    def register_plugin(cls, plugin: Plugin) -> None:
        if threading.current_thread() != threading.main_thread():
            raise RuntimeError("Plugins must be registered on the main thread")
        cls.registered_plugins.append(plugin)

    @property
    def package(self) -> str:
        return self._package

    @property
    def title(self) -> str:
        return self._title

    @property
    def version(self) -> str:
        return self._version

    @property
    def logger(self) -> logging.Logger | None:
        return self._logger
