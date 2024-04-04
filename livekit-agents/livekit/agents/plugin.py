from abc import ABC, abstractmethod
from typing import List


class Plugin(ABC):
    registered_plugins: List["Plugin"] = []

    def __init__(self, title: str, version: str, package: str) -> None:
        self._title = title
        self._version = version
        self._package = package

    @classmethod
    def register_plugin(cls, plugin: "Plugin") -> None:
        cls.registered_plugins.append(plugin)

    @abstractmethod
    def download_files(self) -> None:
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
