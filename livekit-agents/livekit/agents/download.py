from __future__ import annotations

import importlib
import logging
import pkgutil
import sys

from .plugin import Plugin

logger = logging.getLogger("livekit.agents.download")


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


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

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


if __name__ == "__main__":
    sys.exit(main())
