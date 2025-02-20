from __future__ import annotations

import logging
import threading
from abc import ABC
from typing import List, Literal

from . import utils

EventTypes = Literal["plugin_registered",]


class Plugin(ABC):
    """
    Abstract base class for creating LiveKit agent plugins.
    
    Provides core functionality for plugin registration, lifecycle management,
    and common utilities. Plugins should subclass this and implement required methods.
    
    Attributes:
        registered_plugins (List[Plugin]): List of all registered plugin instances
        emitter (EventEmitter): Event emitter for plugin lifecycle events
        
    Example:
        class MyPlugin(Plugin):
            def __init__(self):
                super().__init__(
                    title="My Plugin",
                    version="1.0",
                    package="my_plugin_package"
                )
            
            def download_files(self):
                # Custom implementation
                pass
    """
    registered_plugins: List["Plugin"] = []
    emitter: utils.EventEmitter[EventTypes] = utils.EventEmitter()

    # TODO(theomonnom): make logger mandatory once all plugins have been updated
    def __init__(
        self,
        title: str,
        version: str,
        package: str,
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize a new plugin instance.
        
        Args:
            title: Human-readable name of the plugin
            version: Semantic version string (e.g., "1.0.0")
            package: Python package name containing the plugin
            logger: Optional logger instance (recommended for production use)
        """
        self._title = title
        self._version = version
        self._package = package
        self._logger = logger

    @classmethod
    def register_plugin(cls, plugin: "Plugin") -> None:
        """
        Register a plugin instance with the framework.
        
        Args:
            plugin: Plugin instance to register
            
        Raises:
            RuntimeError: If not called from main thread
        """
        if threading.current_thread() != threading.main_thread():
            raise RuntimeError("Plugins must be registered on the main thread")

        cls.registered_plugins.append(plugin)
        cls.emitter.emit("plugin_registered", plugin)

    def download_files(self) -> None:
        """
        Optional implementation for downloading plugin dependencies.
        
        Subclasses should override this to download any required assets
        before plugin activation. Base implementation does nothing.
        """
        ...

    @property
    def package(self) -> str:
        """Get the Python package name containing this plugin."""
        return self._package

    @property
    def title(self) -> str:
        """Get the human-readable display name of the plugin."""
        return self._title

    @property
    def version(self) -> str:
        """Get the semantic version string of the plugin."""
        return self._version

    @property
    def logger(self) -> logging.Logger | None:
        """
        Get the logger instance for this plugin.
        
        Note: Currently optional but recommended for production use.
        """
        return self._logger
