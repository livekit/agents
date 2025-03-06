import os
from abc import ABC, abstractmethod

import psutil


class MemoryMonitor(ABC):
    @abstractmethod
    def memory_total(self) -> int:
        """Total memory available in bytes."""
        pass

    @abstractmethod
    def memory_used(self) -> int:
        """Memory currently in use in bytes."""
        pass

    @abstractmethod
    def memory_percent(self) -> float:
        """Memory usage percentage between 0 and 1"""
        pass


class DefaultMemoryMonitor(MemoryMonitor):
    def memory_total(self) -> int:
        return psutil.virtual_memory().total

    def memory_used(self) -> int:
        return psutil.virtual_memory().used

    def memory_percent(self) -> float:
        return psutil.virtual_memory().percent / 100.0


class CGroupV2MemoryMonitor(MemoryMonitor):
    def memory_total(self) -> int:
        try:
            with open("/sys/fs/cgroup/memory.max", "r") as f:
                max_memory = f.read().strip()
                if max_memory == "max":
                    return psutil.virtual_memory().total
                return int(max_memory)
        except FileNotFoundError:
            return psutil.virtual_memory().total

    def memory_used(self) -> int:
        with open("/sys/fs/cgroup/memory.current", "r") as f:
            return int(f.read().strip())

    def memory_percent(self) -> float:
        used = self.memory_used()
        total = self.memory_total()
        return min(used / total, 1.0)


def get_memory_monitor() -> MemoryMonitor:
    if _is_cgroup_v2():
        return CGroupV2MemoryMonitor()
    return DefaultMemoryMonitor()


def _is_cgroup_v2() -> bool:
    return os.path.exists("/sys/fs/cgroup/memory.current")
