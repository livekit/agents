from __future__ import annotations

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from livekit.agents.cli.cli import _build_cli
from livekit.agents.worker import AgentServer, ServerEnvOption, ServerOptions


def _make_server(**kwargs) -> AgentServer:
    async def _fake_entrypoint(ctx):
        pass

    opts = ServerOptions(entrypoint_fnc=_fake_entrypoint, **kwargs)
    return AgentServer.from_server_options(opts)


@pytest.fixture
def runner():
    return CliRunner()


class TestServerOptionsLogLevel:
    def test_default_uses_server_env_option(self):
        opts = ServerOptions(entrypoint_fnc=lambda ctx: None)
        assert isinstance(opts.log_level, ServerEnvOption)
        assert ServerEnvOption.getvalue(opts.log_level, False) == "INFO"
        assert ServerEnvOption.getvalue(opts.log_level, True) == "DEBUG"

    def test_custom_string_log_level(self):
        opts = ServerOptions(entrypoint_fnc=lambda ctx: None, log_level="WARN")
        assert opts.log_level == "WARN"
        assert ServerEnvOption.getvalue(opts.log_level, False) == "WARN"
        assert ServerEnvOption.getvalue(opts.log_level, True) == "WARN"

    def test_custom_server_env_option(self):
        opts = ServerOptions(
            entrypoint_fnc=lambda ctx: None,
            log_level=ServerEnvOption(dev_default="ERROR", prod_default="CRITICAL"),
        )
        assert ServerEnvOption.getvalue(opts.log_level, False) == "CRITICAL"
        assert ServerEnvOption.getvalue(opts.log_level, True) == "ERROR"


class TestAgentServerLogLevel:
    def test_default_log_level(self):
        server = AgentServer()
        assert isinstance(server.log_level, ServerEnvOption)
        assert ServerEnvOption.getvalue(server.log_level, False) == "INFO"
        assert ServerEnvOption.getvalue(server.log_level, True) == "DEBUG"

    def test_custom_log_level(self):
        server = AgentServer(log_level="WARN")
        assert server.log_level == "WARN"

    def test_from_server_options_passes_log_level(self):
        server = _make_server(log_level="ERROR")
        assert server.log_level == "ERROR"


class TestStartCommandLogLevel:
    @patch("livekit.agents.cli.cli._run_worker")
    def test_default_log_level_is_info(self, mock_run_worker, runner):
        server = _make_server()
        app = _build_cli(server)
        result = runner.invoke(app, ["start"])
        assert result.exit_code == 0
        args = mock_run_worker.call_args
        assert args.kwargs["args"].log_level == "INFO"

    @patch("livekit.agents.cli.cli._run_worker")
    def test_cli_arg_overrides_default(self, mock_run_worker, runner):
        server = _make_server()
        app = _build_cli(server)
        result = runner.invoke(app, ["start", "--log-level", "error"])
        assert result.exit_code == 0
        assert mock_run_worker.call_args.kwargs["args"].log_level == "ERROR"

    @patch("livekit.agents.cli.cli._run_worker")
    def test_env_var_overrides_default(self, mock_run_worker, runner):
        server = _make_server()
        app = _build_cli(server)
        result = runner.invoke(app, ["start"], env={"LIVEKIT_LOG_LEVEL": "CRITICAL"})
        assert result.exit_code == 0
        assert mock_run_worker.call_args.kwargs["args"].log_level == "CRITICAL"

    @patch("livekit.agents.cli.cli._run_worker")
    def test_cli_arg_overrides_env_var(self, mock_run_worker, runner):
        server = _make_server()
        app = _build_cli(server)
        result = runner.invoke(
            app,
            ["start", "--log-level", "warn"],
            env={"LIVEKIT_LOG_LEVEL": "CRITICAL"},
        )
        assert result.exit_code == 0
        assert mock_run_worker.call_args.kwargs["args"].log_level == "WARN"

    @patch("livekit.agents.cli.cli._run_worker")
    def test_server_opts_log_level_used_when_no_cli_or_env(self, mock_run_worker, runner):
        server = _make_server(log_level="ERROR")
        app = _build_cli(server)
        result = runner.invoke(app, ["start"])
        assert result.exit_code == 0
        assert mock_run_worker.call_args.kwargs["args"].log_level == "ERROR"

    @patch("livekit.agents.cli.cli._run_worker")
    def test_cli_arg_overrides_server_opts(self, mock_run_worker, runner):
        server = _make_server(log_level="ERROR")
        app = _build_cli(server)
        result = runner.invoke(app, ["start", "--log-level", "debug"])
        assert result.exit_code == 0
        assert mock_run_worker.call_args.kwargs["args"].log_level == "DEBUG"


class TestDevCommandLogLevel:
    @patch("livekit.agents.cli.cli._run_worker")
    def test_default_log_level_is_debug(self, mock_run_worker, runner):
        server = _make_server()
        app = _build_cli(server)
        result = runner.invoke(app, ["dev", "--no-reload"])
        assert result.exit_code == 0
        assert mock_run_worker.call_args.kwargs["args"].log_level == "DEBUG"

    @patch("livekit.agents.cli.cli._run_worker")
    def test_cli_arg_overrides_default(self, mock_run_worker, runner):
        server = _make_server()
        app = _build_cli(server)
        result = runner.invoke(app, ["dev", "--no-reload", "--log-level", "info"])
        assert result.exit_code == 0
        assert mock_run_worker.call_args.kwargs["args"].log_level == "INFO"

    @patch("livekit.agents.cli.cli._run_worker")
    def test_env_var_overrides_default(self, mock_run_worker, runner):
        server = _make_server()
        app = _build_cli(server)
        result = runner.invoke(app, ["dev", "--no-reload"], env={"LIVEKIT_LOG_LEVEL": "ERROR"})
        assert result.exit_code == 0
        assert mock_run_worker.call_args.kwargs["args"].log_level == "ERROR"

    @patch("livekit.agents.cli.cli._run_worker")
    def test_server_opts_log_level_used_when_no_cli_or_env(self, mock_run_worker, runner):
        server = _make_server(log_level="CRITICAL")
        app = _build_cli(server)
        result = runner.invoke(app, ["dev", "--no-reload"])
        assert result.exit_code == 0
        assert mock_run_worker.call_args.kwargs["args"].log_level == "CRITICAL"


class TestLogLevelValidation:
    def test_invalid_string_rejected_by_server_options(self):
        with pytest.raises(ValueError, match="Invalid log level"):
            ServerOptions(entrypoint_fnc=lambda ctx: None, log_level="WARNING")

    def test_invalid_string_rejected_by_agent_server(self):
        with pytest.raises(ValueError, match="Invalid log level"):
            AgentServer(log_level="NOTSET")

    def test_invalid_server_env_option_rejected(self):
        with pytest.raises(ValueError, match="Invalid log level"):
            ServerOptions(
                entrypoint_fnc=lambda ctx: None,
                log_level=ServerEnvOption(dev_default="WARNING", prod_default="INFO"),
            )

    def test_valid_levels_accepted_by_server_options(self):
        for level in ("TRACE", "DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"):
            opts = ServerOptions(entrypoint_fnc=lambda ctx: None, log_level=level)
            assert opts.log_level == level

    def test_valid_levels_accepted_by_agent_server(self):
        for level in ("TRACE", "DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"):
            server = AgentServer(log_level=level)
            assert server.log_level == level

    def test_lowercase_normalized_to_uppercase_server_options(self):
        opts = ServerOptions(entrypoint_fnc=lambda ctx: None, log_level="info")
        assert opts.log_level == "INFO"

    def test_lowercase_normalized_to_uppercase_agent_server(self):
        server = AgentServer(log_level="debug")
        assert server.log_level == "DEBUG"

    def test_lowercase_server_env_option_normalized(self):
        opts = ServerOptions(
            entrypoint_fnc=lambda ctx: None,
            log_level=ServerEnvOption(dev_default="debug", prod_default="error"),
        )
        assert ServerEnvOption.getvalue(opts.log_level, True) == "DEBUG"
        assert ServerEnvOption.getvalue(opts.log_level, False) == "ERROR"

    @patch("livekit.agents.cli.cli._run_worker")
    def test_lowercase_server_opts_works_through_cli(self, mock_run_worker, runner):
        server = _make_server(log_level="error")
        app = _build_cli(server)
        result = runner.invoke(app, ["start"])
        assert result.exit_code == 0
        assert mock_run_worker.call_args.kwargs["args"].log_level == "ERROR"
