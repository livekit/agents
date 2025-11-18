"""
Configuration file watcher for dynamic reloading.
Author: Niranjani Sharma
Date: November 19, 2025

BONUS FEATURE #1: File-based configuration reload
This is an alternative to Raghav's REST API approach.
"""

import asyncio
import time
from pathlib import Path
from typing import Callable


class ConfigFileWatcher:
    """Watches a configuration file and triggers reload on changes.

    Monitors a JSON file and automatically reloads configuration
    when the file is modified.
    """

    def __init__(self, config_path: str, reload_callback: Callable[[], None]):
        """Initialize the config file watcher.

        Args:
            config_path: Path to the configuration JSON file
            reload_callback: Function to call when config changes detected
        """
        self.config_path = Path(config_path)
        self.reload_callback = reload_callback
        self._last_modified: float | None = None
        self._watch_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """Start watching the config file for changes."""
        if self._running:
            print("[CONFIG_WATCHER] Already running")
            return

        self._running = True
        self._watch_task = asyncio.create_task(self._watch_loop())
        print(f"[CONFIG_WATCHER] Started monitoring {self.config_path}")

    async def stop(self) -> None:
        """Stop watching the config file."""
        self._running = False
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
        print("[CONFIG_WATCHER] Stopped")
    
    async def _watch_loop(self) -> None:
        """
        Main watch loop that checks file modification time.
        
        Polls the file every 2 seconds for changes.
        """
        while self._running:
            try:
                if self.config_path.exists():
                    current_mtime = self.config_path.stat().st_mtime
                    
                    if self._last_modified is None:
                        # First time seeing the file
                        self._last_modified = current_mtime
                        print(f"[CONFIG_WATCHER] Initial file check: {self.config_path}")
                    elif current_mtime > self._last_modified:
                        # File has been modified
                        print(f"[CONFIG_WATCHER] Detected change in {self.config_path}")
                        print(f"[CONFIG_WATCHER] Last modified: {time.ctime(self._last_modified)}")
                        print(f"[CONFIG_WATCHER] New modified: {time.ctime(current_mtime)}")
                        
                        self._last_modified = current_mtime
                        
                        # Call the reload callback
                        try:
                            self.reload_callback()
                            print("[CONFIG_WATCHER] Configuration reloaded successfully")
                        except Exception as e:
                            print(f"[CONFIG_WATCHER] Error during reload: {e}")
                else:
                    if self._last_modified is not None:
                        print(f"[CONFIG_WATCHER] Config file no longer exists: {self.config_path}")
                        self._last_modified = None
                
                # Wait before next check (polling interval)
                await asyncio.sleep(2)
                
            except Exception as e:
                print(f"[CONFIG_WATCHER] Error watching file: {e}")
                await asyncio.sleep(5)  # Longer wait on error
    
    def check_now(self) -> bool:
        """
        Synchronously check if file has changed without waiting.
        
        Returns:
            True if file was modified and callback was triggered
        """
        try:
            if self.config_path.exists():
                current_mtime = self.config_path.stat().st_mtime
                
                if self._last_modified is None:
                    self._last_modified = current_mtime
                    return False
                elif current_mtime > self._last_modified:
                    self._last_modified = current_mtime
                    self.reload_callback()
                    return True
            return False
        except Exception as e:
            print(f"[CONFIG_WATCHER] Error in check_now: {e}")
            return False
