"""Config precedence tests.

These tests simulate the full lifecycle of an AgentServer to verify
the config override order matches what the simulation system depends on:
  url/keys:    update_options (CLI) > constructor arg > env var
  agent_name:  LIVEKIT_AGENT_NAME env > @rtc_session decorator arg
"""

from __future__ import annotations

import os
from unittest.mock import patch

from livekit.agents.cli import proto
from livekit.agents.cli.cli import _run_worker
from livekit.agents.worker import AgentServer


class _TestableServer(AgentServer):
    """AgentServer that captures config at run() time instead of blocking."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.captured_config: dict | None = None

    async def run(self, **kwargs):
        self.captured_config = {
            "ws_url": self._ws_url,
            "api_key": self._api_key,
            "api_secret": self._api_secret,
            "agent_name": self._agent_name,
        }
        raise SystemExit(0)


def _run_and_capture(server: _TestableServer, args: proto.CliArgs) -> dict:
    """Run _run_worker and return the config that was active when run() was called."""
    try:
        _run_worker(server, args)
    except SystemExit:
        pass
    assert server.captured_config is not None, "run() was never called"
    return server.captured_config


class TestCLIOverridesConstructorAndEnv:
    def test_cli_url_wins(self):
        with patch.dict(os.environ, {"LIVEKIT_URL": "ws://env"}):
            server = _TestableServer(ws_url="ws://constructor")
            config = _run_and_capture(server, proto.CliArgs(log_level="INFO", url="ws://cli"))
            assert config["ws_url"] == "ws://cli"

    def test_cli_keys_win(self):
        with patch.dict(os.environ, {"LIVEKIT_API_KEY": "env-key", "LIVEKIT_API_SECRET": "env-secret"}):
            server = _TestableServer(api_key="constructor-key", api_secret="constructor-secret")
            config = _run_and_capture(server, proto.CliArgs(
                log_level="INFO", api_key="cli-key", api_secret="cli-secret",
            ))
            assert config["api_key"] == "cli-key"
            assert config["api_secret"] == "cli-secret"

    def test_constructor_wins_over_env_when_no_cli(self):
        with patch.dict(os.environ, {"LIVEKIT_URL": "ws://env"}):
            server = _TestableServer(ws_url="ws://constructor")
            config = _run_and_capture(server, proto.CliArgs(log_level="INFO"))
            assert config["ws_url"] == "ws://constructor"

    def test_env_fallback_when_no_constructor_no_cli(self):
        with patch.dict(os.environ, {"LIVEKIT_URL": "ws://env"}, clear=False):
            server = _TestableServer()
            config = _run_and_capture(server, proto.CliArgs(log_level="INFO"))
            assert config["ws_url"] == "ws://env"


class TestAgentNameEnvWinsOverDecorator:
    def test_env_wins(self):
        with patch.dict(os.environ, {"LIVEKIT_AGENT_NAME": "env-agent"}):
            server = _TestableServer()

            @server.rtc_session(agent_name="decorator-agent")
            async def entrypoint(ctx):
                pass

            config = _run_and_capture(server, proto.CliArgs(log_level="INFO"))
            assert config["agent_name"] == "env-agent"

    def test_decorator_fallback(self):
        with patch.dict(os.environ, {}, clear=True):
            server = _TestableServer()

            @server.rtc_session(agent_name="decorator-agent")
            async def entrypoint(ctx):
                pass

            config = _run_and_capture(server, proto.CliArgs(log_level="INFO"))
            assert config["agent_name"] == "decorator-agent"

    def test_empty_when_neither(self):
        with patch.dict(os.environ, {}, clear=True):
            server = _TestableServer()
            config = _run_and_capture(server, proto.CliArgs(log_level="INFO"))
            assert config["agent_name"] == ""


class TestFullPrecedenceChain:
    def test_everything_set(self):
        with patch.dict(os.environ, {
            "LIVEKIT_URL": "ws://env",
            "LIVEKIT_API_KEY": "env-key",
            "LIVEKIT_API_SECRET": "env-secret",
            "LIVEKIT_AGENT_NAME": "env-agent",
        }):
            server = _TestableServer(
                ws_url="ws://constructor",
                api_key="constructor-key",
                api_secret="constructor-secret",
            )

            @server.rtc_session(agent_name="decorator-agent")
            async def entrypoint(ctx):
                pass

            config = _run_and_capture(server, proto.CliArgs(
                log_level="INFO",
                url="ws://cli",
                api_key="cli-key",
                api_secret="cli-secret",
            ))

            assert config["ws_url"] == "ws://cli"
            assert config["api_key"] == "cli-key"
            assert config["api_secret"] == "cli-secret"
            assert config["agent_name"] == "env-agent"
