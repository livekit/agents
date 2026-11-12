"""The CLI entry, run as users run it: ``python agent.py <command>``."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import certifi
import pytest

from livekit.agents import Agent, VoiceSession
from tests_new.fakes import FakeLLM

from .conftest import ConsolePeer

AGENT_MODULE = """
from livekit.agents import AgentServer, cli

server = AgentServer()

@server.rtc_session()
async def entrypoint(ctx):
    pass

if __name__ == "__main__":
    cli.run_app(server)
"""

GREETING_AGENT_MODULE = """
from livekit.agents import Agent, AgentServer, VoiceSession, cli
from tests_new.fakes import FakeTTS

server = AgentServer()

@server.rtc_session()
async def entrypoint(ctx):
    session = VoiceSession(tts=FakeTTS(streaming=False))
    await session.start(Agent(instructions="Greet."))
    await session.say("Hello from the console.")
    await session.aclose()
    ctx.shutdown()

if __name__ == "__main__":
    cli.run_app(server)
"""

NO_EVENT_LOOP = """
import asyncio

asyncio.set_event_loop(None)  # as on Python 3.14's main thread
"""

LOGGING_SCRIPT = """
import logging, sys

from livekit.agents import cli

logging.getLogger("customer.chatty").setLevel(logging.DEBUG)
cli.setup_logging("info", sys.argv[1])
logging.getLogger().info("root says hello")
logging.getLogger("httpx").info("httpx request line")
logging.getLogger("httpx").warning("httpx warning")
logging.getLogger("customer.chatty").debug("chatty detail")
"""


NO_TRUST_STORE_SCRIPT = """
import os, ssl
from livekit.agents import AgentServer, cli

# a container image without a system trust store
ssl.get_default_verify_paths = lambda: ssl.DefaultVerifyPaths(None, None, "", "", "", "")
try:
    cli.run_app(AgentServer())
except SystemExit:
    pass
print(os.environ.get("SSL_CERT_FILE"))
"""


async def run_console(agent: Path, console: ConsolePeer, env: dict[str, str]) -> int:
    process = await asyncio.create_subprocess_exec(
        sys.executable, agent, "console", "--connect-addr", console.addr, env=env
    )
    return await process.wait()


async def test_console_runs_without_livekit_credentials(
    console: ConsolePeer, no_livekit_credentials: None
) -> None:
    """A console call needs no LiveKit URL or keys. https://github.com/livekit/agents/pull/3920"""
    session = VoiceSession(llm=FakeLLM(["Hi!"]))
    await session.start(Agent(instructions="Answer."))
    await session.generate_reply(user_input="hello")
    await session.aclose()

    assert "Hi!" in console.assistant_texts()


async def test_console_command_runs_a_call_and_exits(
    console: ConsolePeer,
    agent_module: Callable[[str], Path],
    agent_env: dict[str, str],
    no_livekit_credentials: None,
) -> None:
    """``python agent.py console`` runs one call without LiveKit credentials, then exits 0.
    https://github.com/livekit/agents/pull/1826"""
    agent = agent_module(GREETING_AGENT_MODULE)

    status = await run_console(agent, console, agent_env)

    assert status == 0
    assert console.assistant_texts() == ["Hello from the console."]


async def test_console_runs_where_the_main_thread_has_no_event_loop(
    console: ConsolePeer,
    agent_module: Callable[[str], Path],
    agent_env: dict[str, str],
) -> None:
    """The CLI creates the loop it runs on, as Python 3.14's main thread has none.
    https://github.com/livekit/agents/issues/4755"""
    agent = agent_module(NO_EVENT_LOOP + GREETING_AGENT_MODULE)

    status = await run_console(agent, console, agent_env)

    assert status == 0
    assert console.assistant_texts() == ["Hello from the console."]


def test_start_without_credentials_names_the_missing_setting(
    agent_module: Callable[[str], Path], no_livekit_credentials: None
) -> None:
    """`start` without LiveKit credentials exits naming LIVEKIT_URL. https://github.com/livekit/agents/pull/1826"""
    agent = agent_module(AGENT_MODULE)

    run = subprocess.run([sys.executable, agent, "start"], capture_output=True, text=True)

    assert run.returncode != 0
    assert "LIVEKIT_URL" in run.stdout + run.stderr


def test_without_a_system_trust_store_every_client_trusts_certifi() -> None:
    """With no system CA paths, `run_app` points every client at certifi's bundle.

    https://github.com/livekit/agents/pull/6231
    """
    env = {k: v for k, v in os.environ.items() if k not in ("SSL_CERT_FILE", "SSL_CERT_DIR")}

    run = subprocess.run(
        [sys.executable, "-c", NO_TRUST_STORE_SCRIPT, "--help"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )

    assert run.stdout.splitlines()[-1] == certifi.where()


def logged(style: str) -> str:
    run = subprocess.run(
        [sys.executable, "-c", LOGGING_SCRIPT, style], capture_output=True, text=True, check=True
    )
    return run.stdout


@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/issues/4056 reproduces: dev log lines print "
    "%H:%M:%S, without the milliseconds",
)
def test_dev_log_lines_carry_milliseconds() -> None:
    """Development log lines time records to the millisecond.
    https://github.com/livekit/agents/issues/4056"""
    line = logged("dev").splitlines()[0]

    assert line[8] in ".," and line[9:12].isdigit(), line


def test_cli_logging_shows_root_info_and_keeps_application_levels() -> None:
    """At INFO the root logger's records show and a level the application set is kept.
    https://github.com/livekit/agents/pull/768"""
    out = logged("json")

    assert "root says hello" in out
    assert "chatty detail" in out


def test_cli_logging_quiets_noisy_libraries_below_warning() -> None:
    """A noisy library's INFO lines are dropped while its warnings show."""
    out = logged("json")

    assert "httpx request line" not in out
    assert "httpx warning" in out


@pytest.mark.parametrize(
    ("flag", "module"),
    [
        ("LIVEKIT_EVALS_VERBOSE", "livekit.agents"),
        ("LK_GOOGLE_DEBUG", "livekit.plugins.google"),
        ("LK_OPENAI_DEBUG", "livekit.plugins.openai.realtime.gpt_live_model"),
        ("LK_BEDROCK_DEBUG", "livekit.plugins.aws.experimental.realtime"),
        ("LK_SESSION_MAX_DURATION", "livekit.plugins.aws.experimental.realtime"),
    ],
)
@pytest.mark.parametrize("value", ["", "yes"], ids=["blank", "non_numeric"])
@pytest.mark.xfail(
    strict=True,
    reason="https://github.com/livekit/agents/pull/7470 reproduces: the flag is parsed with int() "
    "at import",
)
def test_a_debug_flag_value_cannot_break_an_import(flag: str, module: str, value: str) -> None:
    """Importing the package or a plugin succeeds whatever a numeric debug flag holds.
    https://github.com/livekit/agents/pull/7470"""
    run = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        env={**os.environ, flag: value},
        capture_output=True,
        text=True,
    )

    assert run.returncode == 0, run.stderr.splitlines()[-1]
