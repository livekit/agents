from abc import ABC, abstractmethod
from typing import List


class Plugin(ABC):
    registered_plugins: List["Plugin"] = []

    def __init__(self, title: str, version: str) -> None:
        self._title = title
        self._version = version

    @classmethod
    def register_plugin(cls, plugin: "Plugin") -> None:
        cls.registered_plugins.append(plugin)

    @abstractmethod
    def download_files(self) -> None:
        """
        Blocking is allowed inside this method
        This is the perfect place to download models for e.g
        """
        pass

    @property
    def title(self) -> str:
        return self._title

    @property
    def version(self) -> str:
        return self._version
