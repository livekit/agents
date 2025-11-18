# filler_agent/settings_loader.py
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Set, Optional, Tuple

logger = logging.getLogger("filler_agent.settings_loader")


@dataclass
class RuntimeWordConfig:
    """
    Holds per-language filler & command words in memory,
    and provides helpers for looking them up and updating them.
    """
    ignored_by_lang: Dict[str, Set[str]]
    commands_by_lang: Dict[str, Set[str]]
    default_language: str

    def _normalize_lang(self, lang: Optional[str]) -> str:
        if not lang:
            return self.default_language
        # e.g. "en-US" -> "en"
        return lang.split("-")[0].lower()

    def get_sets_for_language(self, lang: Optional[str]) -> Tuple[Set[str], Set[str]]:
        """
        Return (ignored_words, command_words) for the given language code.
        Falls back to default language, then global unions.
        """
        lang = self._normalize_lang(lang)

        ignored = self.ignored_by_lang.get(lang)
        commands = self.commands_by_lang.get(lang)

        if ignored is None:
            ignored = self.ignored_by_lang.get(self.default_language, set())
        if commands is None:
            commands = self.commands_by_lang.get(self.default_language, set())

        # Final safety: if still empty, union everything
        if not ignored:
            ignored = set().union(*self.ignored_by_lang.values())
        if not commands:
            commands = set().union(*self.commands_by_lang.values())

        return ignored, commands

    def update_from_dict(self, data: dict) -> None:
        """
        Update word lists from a dict with structure:

        {
          "ignored": { "en": ["uh", "umm"], "hi": ["haan"] },
          "commands": { "en": ["wait", "stop"], "hi": ["ruko"] }
        }
        """
        if "ignored" in data:
            for lang, words in data["ignored"].items():
                lang = lang.lower()
                self.ignored_by_lang[lang] = {w.strip().lower() for w in words}

        if "commands" in data:
            for lang, words in data["commands"].items():
                lang = lang.lower()
                self.commands_by_lang[lang] = {w.strip().lower() for w in words}

        logger.info("Runtime word config updated: %s", self)


async def watch_config_file(path: str, cfg: RuntimeWordConfig, poll_interval: float = 5.0) -> None:
    """
    Periodically watch a JSON file for changes and update cfg when it changes.

    This enables dynamic runtime updates to filler/command word lists
    without restarting the agent.
    """
    logger.info("Starting dynamic config watcher for %s", path)
    last_mtime: Optional[float] = None

    while True:
        try:
            stat = os.stat(path)
        except FileNotFoundError:
            # File might be created later; just wait.
            await asyncio.sleep(poll_interval)
            continue

        if last_mtime is None or stat.st_mtime > last_mtime:
            last_mtime = stat.st_mtime
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                cfg.update_from_dict(data)
                logger.info("Reloaded dynamic filler config from %s", path)
            except Exception:
                logger.exception("Failed to reload dynamic config from %s", path)

        await asyncio.sleep(poll_interval)
