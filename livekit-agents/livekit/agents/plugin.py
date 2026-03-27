from __future__ import annotations

import logging
import threading
from abc import ABC
from typing import Literal

from . import utils

EventTypes = Literal["plugin_registered",]

PluginMaintainer = Literal["livekit", "community"]


class Plugin(ABC):  # noqa: B024
    registered_plugins: list[Plugin] = []
    emitter: utils.EventEmitter[EventTypes] = utils.EventEmitter()

    # TODO(theomonnom): make logger mandatory once all plugins have been updated
    def __init__(
        self,
        title: str,
        version: str,
        package: str,
        logger: logging.Logger | None = None,
        maintained_by: PluginMaintainer = "community",
    ) -> None:
        self._title = title
        self._version = version
        self._package = package
        self._logger = logger
        self._maintained_by = maintained_by

    @classmethod
    def register_plugin(cls, plugin: Plugin) -> None:
        if threading.current_thread() != threading.main_thread():
            raise RuntimeError("Plugins must be registered on the main thread")

        cls.registered_plugins.append(plugin)
        cls.emitter.emit("plugin_registered", plugin)

    # plugin can implement an optional download_files method
    def download_files(self) -> None:  # noqa: B027
        pass

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

    @property
    def maintained_by(self) -> PluginMaintainer:
        return self._maintained_by
