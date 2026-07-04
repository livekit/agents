from __future__ import annotations

import argparse
import importlib
import logging
import pkgutil
import sys
from typing import TYPE_CHECKING

from .plugin import Plugin

if TYPE_CHECKING:
    from .worker import AgentServer

logger = logging.getLogger("livekit.agents")


def _discover_and_import_plugins() -> list[str]:
    try:
        import livekit.plugins as ns
    except ImportError:
        logger.warning("no livekit.plugins namespace found — nothing to download")
        return []

    attempted: list[str] = []
    for _finder, name, is_pkg in pkgutil.iter_modules(ns.__path__, prefix="livekit.plugins."):
        if not is_pkg:
            continue
        attempted.append(name)
        try:
            importlib.import_module(name)
        except Exception as e:
            logger.warning("failed to import %s: %s", name, e)
    return attempted


def _download_files() -> int:
    attempted = _discover_and_import_plugins()
    logger.info(
        "discovered %d plugin package(s): %s",
        len(attempted),
        ", ".join(attempted) if attempted else "(none)",
    )

    exit_code = 0
    for plugin in Plugin.registered_plugins:
        logger.info("downloading files for %s", plugin.package)
        try:
            plugin.download_files()
        except Exception as e:
            logger.error("failed downloading files for %s: %s", plugin.package, e)
            exit_code = 1
        else:
            logger.info("finished downloading files for %s", plugin.package)
    return exit_code


def _dispatch(server: AgentServer, args: argparse.Namespace) -> None:
    from .cli import proto
    from .cli.cli import _run_tcp_console, _run_worker

    if args.command == "console":
        _run_tcp_console(server=server, connect_addr=args.connect_addr, record=args.record)
    elif args.command == "start":
        _run_worker(
            server=server,
            args=proto.CliArgs(
                log_level=args.log_level,
                url=args.url,
                api_key=args.api_key,
                api_secret=args.api_secret,
                reload_addr=args.reload_addr,
                log_format=args.log_format,
                dev=args.dev,
                simulation=args.simulation,
            ),
        )


def _discover_server(entrypoint: str | None) -> AgentServer:
    """Import the user's entrypoint and return the AgentServer it defines.

    Reuses the discovery in cli.discover (prefers an app, server, or agent global,
    else the single AgentServer in the module).
    """
    from pathlib import Path

    from .cli.discover import get_import_data

    import_data = get_import_data(path=Path(entrypoint) if entrypoint else None)
    mod = importlib.import_module(import_data.module_data.module_import_str)
    return getattr(mod, import_data.app_name)  # type: ignore[no-any-return]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m livekit.agents",
        description="LiveKit Agents utilities.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser(
        "download-files",
        help="Discover installed livekit-plugins-* packages and run their download_files step.",
    )

    start_p = sub.add_parser("start")
    start_p.add_argument("entrypoint", nargs="?")
    start_p.add_argument("--log-level", default="INFO")
    start_p.add_argument("--log-format", choices=["json", "colored"], default="json")
    start_p.add_argument("--url")
    start_p.add_argument("--api-key")
    start_p.add_argument("--api-secret")
    start_p.add_argument("--dev", action="store_true", default=False)
    start_p.add_argument("--reload-addr")
    # set by `lk simulate`: disables the worker load limit so simulation runs
    # can saturate the agent
    start_p.add_argument("--simulation", action="store_true", default=False)

    console_p = sub.add_parser("console")
    console_p.add_argument("entrypoint", nargs="?")
    console_p.add_argument("--connect-addr", required=True)
    console_p.add_argument("--record", action="store_true", default=False)

    args = parser.parse_args(argv)

    if args.command == "download-files":
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        return _download_files()

    _dispatch(_discover_server(args.entrypoint), args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
