import os
import time
from abc import ABC, abstractmethod

import psutil


class CPUMonitor(ABC):
    @abstractmethod
    def cpu_count(self) -> float:
        """Number of logical CPUs.

        Returns a float to allow for fractional CPUs (in the case of cgroups)."""
        pass

    @abstractmethod
    def cpu_percent(self, interval: float = 0.5) -> float:
        """CPU usage percentage between 0 and 1"""
        pass


class DefaultCPUMonitor(CPUMonitor):
    def cpu_count(self) -> float:
        return psutil.cpu_count() or 1.0

    def cpu_percent(self, interval: float = 0.5) -> float:
        return psutil.cpu_percent(interval) / 100.0


class CGroupV2CPUMonitor(CPUMonitor):
    def cpu_count(self) -> float:
        # quota: The maximum CPU time in microseconds that the cgroup can use within a given period.
        # period: The period of time in microseconds over which the quota applies.
        # If the quota is set to "max", it means the cgroup is allowed to use all available CPUs without restriction.  # noqa: E501
        # Otherwise, the quota is a number that represents the maximum CPU time in microseconds that the cgroup can use within a given period.  # noqa: E501
        quota, period = self._read_cpu_max()
        if quota == "max":
            return os.cpu_count() or 1
        return 1.0 * int(quota) / period

    def cpu_percent(self, interval: float = 0.5) -> float:
        cpu_usage_start = self._read_cpu_usage()
        time.sleep(interval)
        cpu_usage_end = self._read_cpu_usage()
        cpu_usage_diff = cpu_usage_end - cpu_usage_start

        # Convert microseconds to seconds
        cpu_usage_seconds = cpu_usage_diff / 1_000_000

        # Get the number of CPUs available to the container
        num_cpus = self.cpu_count()

        # Calculate the percentage
        cpu_usage_percent = cpu_usage_seconds / (interval * num_cpus)

        return min(cpu_usage_percent, 1)

    def _read_cpu_max(self) -> tuple[str, int]:
        try:
            with open("/sys/fs/cgroup/cpu.max") as f:
                data = f.read().strip().split()
            quota = data[0]
            period = int(data[1])
        except FileNotFoundError:
            quota = "max"
            period = 100000
        return quota, period

    def _read_cpu_usage(self) -> int:
        with open("/sys/fs/cgroup/cpu.stat") as f:
            for line in f:
                if line.startswith("usage_usec"):
                    return int(line.split()[1])
        raise RuntimeError("Failed to read CPU usage")


def get_cpu_monitor() -> CPUMonitor:
    if _is_cgroup_v2():
        return CGroupV2CPUMonitor()
    return DefaultCPUMonitor()


def _is_cgroup_v2() -> bool:
    return os.path.exists("/sys/fs/cgroup/cpu.stat")
