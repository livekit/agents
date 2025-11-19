import os
import time
from abc import ABC, abstractmethod
from typing import Optional

import psutil

from ...log import logger


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


def _cpu_count_from_env() -> Optional[float]:
    try:
        if "NUM_CPUS" in os.environ:
            return float(os.environ["NUM_CPUS"])
    except ValueError:
        logger.warning("Failed to parse NUM_CPUS from environment", exc_info=True)
    return None


class DefaultCPUMonitor(CPUMonitor):
    def cpu_count(self) -> float:
        return _cpu_count_from_env() or psutil.cpu_count() or 1.0

    def cpu_percent(self, interval: float = 0.5) -> float:
        return psutil.cpu_percent(interval) / 100.0


class CGroupV2CPUMonitor(CPUMonitor):
    def cpu_count(self) -> float:
        # quota: The maximum CPU time in microseconds that the cgroup can use within a given period.
        # period: The period of time in microseconds over which the quota applies.
        # If the quota is set to "max", it means the cgroup is allowed to use all available CPUs without restriction.  # noqa: E501
        # Otherwise, the quota is a number that represents the maximum CPU time in microseconds that the cgroup can use within a given period.  # noqa: E501
        env_cpus = _cpu_count_from_env()
        if env_cpus is not None:
            return env_cpus
        quota, period = self._read_cpu_max()
        if quota == "max":
            return psutil.cpu_count() or 1.0
        return 1.0 * int(quota) / period

    def cpu_percent(self, interval: float = 0.5) -> float:
        cpu_usage_start = self._read_cpu_usage()
        time.sleep(interval)
        cpu_usage_end = self._read_cpu_usage()
        cpu_usage_diff = cpu_usage_end - cpu_usage_start

        # microseconds to seconds
        cpu_usage_seconds = cpu_usage_diff / 1_000_000

        num_cpus = self.cpu_count()
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


class CGroupV1CPUMonitor(CPUMonitor):
    def cpu_count(self) -> float:
        # often, cgroups v1 quota isn't set correctly, so we need to rely on an env var to
        # correctly determine the number of CPUs
        env_cpus = _cpu_count_from_env()
        if env_cpus is not None:
            return env_cpus
        quota, period = self._read_cfs_quota_and_period()
        if quota is None or quota < 0 or period is None or period <= 0:
            # we do not want to use the node CPU count, as it could overstate the number
            # available to the container
            return 2.0
        return max(1.0 * quota / period, 1.0)

    def cpu_percent(self, interval: float = 0.5) -> float:
        usage_start = self._read_cpuacct_usage()
        time.sleep(interval)
        usage_end = self._read_cpuacct_usage()
        usage_diff_ns = usage_end - usage_start

        usage_seconds = usage_diff_ns / 1_000_000_000
        num_cpus = self.cpu_count()
        percent = usage_seconds / (interval * num_cpus)
        return max(min(percent, 1.0), 0.0)

    def _read_cfs_quota_and_period(self) -> tuple[Optional[int], Optional[int]]:
        quota_path_candidates = [
            "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
        ]
        period_path_candidates = [
            "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
        ]
        quota = self._read_first_int(quota_path_candidates)
        period = self._read_first_int(period_path_candidates)
        return quota, period

    def _read_cpuacct_usage(self) -> int:
        candidates = [
            "/sys/fs/cgroup/cpuacct/cpuacct.usage",
        ]
        value = self._read_first_int(candidates)
        if value is None:
            raise RuntimeError("Failed to read cpuacct.usage for cgroup v1")
        return value

    def _read_first_int(self, paths: list[str]) -> Optional[int]:
        for p in paths:
            try:
                with open(p) as f:
                    return int(f.read().strip())
            except FileNotFoundError:
                continue
            except ValueError:
                continue
        return None


def get_cpu_monitor() -> CPUMonitor:
    if _is_cgroup_v2():
        return CGroupV2CPUMonitor()
    if _is_cgroup_v1():
        return CGroupV1CPUMonitor()
    return DefaultCPUMonitor()


def _is_cgroup_v2() -> bool:
    return os.path.exists("/sys/fs/cgroup/cpu.stat")


def _is_cgroup_v1() -> bool:
    candidates = [
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
        "/sys/fs/cgroup/cpuacct/cpuacct.usage",
    ]
    return any(os.path.exists(p) for p in candidates)
