from __future__ import annotations

import importlib
import logging

import pytest

from livekit.agents import Plugin

pytestmark = pytest.mark.unit


# plugins that have a `livekit.plugins.<name>` logger of their own. The CLI gives the log
# level to the loggers of registered plugins, so a plugin that registers without its
# logger stays at the `livekit` logger's WARNING and loses its debug and info lines
@pytest.mark.parametrize("name", ["clova", "gradium", "hume", "resemble", "rime", "smallestai"])
def test_plugin_registers_its_logger(name: str) -> None:
    module = pytest.importorskip(f"livekit.plugins.{name}")
    log = importlib.import_module(f"livekit.plugins.{name}.log")

    registered = [p for p in Plugin.registered_plugins if p.title == module.__name__]
    assert len(registered) == 1, f"{module.__name__} should register exactly one plugin"

    (plugin,) = registered
    assert isinstance(plugin.logger, logging.Logger)
    assert plugin.logger is log.logger
    assert plugin.logger.name == f"livekit.plugins.{name}"
