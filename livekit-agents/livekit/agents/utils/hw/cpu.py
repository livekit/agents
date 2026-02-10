import os
import time
from abc import ABC, abstractmethod

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


def _cpu_count_from_env() -> float | None:
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
            try:
                return len(psutil.Process().cpu_affinity())
            except:
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
        """Read CPU quota and period from cgroup v2.
        
        The cpu.max file can be in different locations depending on the cgroup hierarchy:
        - /sys/fs/cgroup/cpu.max (container root)
        - /sys/fs/cgroup/system.slice/cpu.max (systemd system slice)
        - /sys/fs/cgroup/user.slice/cpu.max (systemd user slice)
        - /proc/self/cgroup can tell us the actual cgroup path
        """
        cpu_max_paths = [
            "/sys/fs/cgroup/cpu.max",  # Most common in containers
        ]

        # try to find the actual cgroup path from /proc/self/cgroup
        try:
            with open("/proc/self/cgroup", "r") as f:
                for line in f:
                    # Format: hierarchy-ID:controller-list:cgroup-path
                    # Example: 0::/system.slice/docker-xyz.scope
                    parts = line.strip().split(":")
                    if len(parts) >= 3:
                        cgroup_path = parts[2]
                        if cgroup_path and cgroup_path != "/":
                            # Add the specific cgroup path
                            specific_path = f"/sys/fs/cgroup{cgroup_path}/cpu.max"
                            cpu_max_paths.insert(0, specific_path)
                            # Also try parent directories
                            parent_path = os.path.dirname(cgroup_path)
                            if parent_path and parent_path != "/":
                                cpu_max_paths.append(f"/sys/fs/cgroup{parent_path}/cpu.max")
        except (FileNotFoundError, OSError, IOError) as e:
            pass

        # common fallback paths
        cpu_max_paths.extend([
            "/sys/fs/cgroup/system.slice/cpu.max",
            "/sys/fs/cgroup/user.slice/cpu.max",
            # For Kubernetes/Docker containers that might be in a pod slice
            "/sys/fs/cgroup/kubepods.slice/cpu.max",
            "/sys/fs/cgroup/docker/cpu.max",
        ])

        # try each path in order
        for cpu_max_path in cpu_max_paths:
            try:
                with open(cpu_max_path) as f:
                    data = f.read().strip().split()
                quota = data[0]
                period = int(data[1]) if len(data) > 1 else 100000
                if period <= 0:
                    logger.warning(f"Invalid CPU period {period} in {cpu_max_path}, using default")
                    period = 100000
                return quota, period
            except FileNotFoundError:
                continue
            except (ValueError, IndexError) as e:
                continue
            except (OSError, IOError) as e:
                continue

        # if we couldn't find any cpu.max file, return defaults
        return "max", 100000

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

    def _read_cfs_quota_and_period(self) -> tuple[int | None, int | None]:
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

    def _read_first_int(self, paths: list[str]) -> int | None:
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
