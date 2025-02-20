"""
CPU resource monitoring utilities with container-aware metrics.
Provides accurate CPU measurements in both host and containerized environments.

Features:
- Automatic detection of cgroups v2 environments
- Container CPU quota/period awareness
- Unified interface for host/container metrics
- Percentage-based CPU utilization
"""

import os
import time
from abc import ABC, abstractmethod

import psutil


class CPUMonitor(ABC):
    """Abstract base class for CPU monitoring implementations.
    
    Provides consistent interface for:
    - Logical CPU count (including container quotas)
    - CPU utilization percentage
    """

    @abstractmethod
    def cpu_count(self) -> float:
        """Get available logical CPUs as float to handle fractional allocations.
        
        Returns:
            float: Effective CPU count, may be fractional in constrained environments
        """
        pass

    @abstractmethod
    def cpu_percent(self, interval: float = 0.5) -> float:
        """Measure CPU utilization as percentage of available resources.
        
        Args:
            interval: Measurement duration in seconds
        Returns:
            float: Utilization between 0.0 (0%) and 1.0 (100%)
        """
        pass


class DefaultCPUMonitor(CPUMonitor):
    """Standard CPU monitor for non-containerized environments using psutil."""
    
    def cpu_count(self) -> float:
        """Get total host logical CPUs."""
        return psutil.cpu_count() or 1.0  # Fallback for undefined environments

    def cpu_percent(self, interval: float = 0.5) -> float:
        """System-wide CPU utilization using psutil's sampling method."""
        return psutil.cpu_percent(interval) / 100.0  # Convert percentage to 0-1 range


class CGroupV2CPUMonitor(CPUMonitor):
    """cgroups v2 aware monitor for containerized environments.
    
    Reads CPU metrics directly from cgroup interface files:
    - /sys/fs/cgroup/cpu.max: CPU quota/period
    - /sys/fs/cgroup/cpu.stat: CPU usage statistics
    
    See Linux kernel documentation for details:
    https://docs.kernel.org/admin-guide/cgroup-v2.html
    """
    
    def cpu_count(self) -> float:
        """Calculate effective CPU count from cgroup quota/period."""
        quota, period = self._read_cpu_max()
        
        # Handle unlimited quota case
        if quota == "max":
            return os.cpu_count() or 1.0  # Fallback if host CPUs undetectable
            
        return float(quota) / period  # Fractional CPU count

    def cpu_percent(self, interval: float = 0.5) -> float:
        """Calculate CPU usage as percentage of allocated resources."""
        # Measure usage over interval
        cpu_usage_start = self._read_cpu_usage()
        time.sleep(interval)
        cpu_usage_end = self._read_cpu_usage()
        
        # Convert microseconds to seconds
        cpu_usage_diff = (cpu_usage_end - cpu_usage_start) / 1_000_000  
        num_cpus = self.cpu_count()
        
        # Calculate percentage of allocated CPU time used
        cpu_usage_percent = cpu_usage_diff / (interval * num_cpus)
        return min(cpu_usage_percent, 1.0)  # Clamp at 100% utilization

    def _read_cpu_max(self) -> tuple[str, int]:
        """Read cgroup CPU quota and period from cpu.max file."""
        try:
            with open("/sys/fs/cgroup/cpu.max", "r") as f:
                data = f.read().strip().split()
            return data[0], int(data[1])
        except FileNotFoundError:
            # Fallback if cgroup file missing (shouldn't happen in cgroupv2)
            return "max", 100000  # Default period of 100ms

    def _read_cpu_usage(self) -> int:
        """Read total CPU time used from cpu.stat."""
        with open("/sys/fs/cgroup/cpu.stat", "r") as f:
            for line in f:
                if line.startswith("usage_usec"):
                    return int(line.split()[1])
        raise RuntimeError("cgroup v2 cpu.stat missing usage_usec entry")


def get_cpu_monitor() -> CPUMonitor:
    """Factory function to get appropriate CPU monitor for current environment.
    
    Automatically detects cgroups v2 environments. Use this rather than
    instantiating monitors directly.
    
    Usage:
        monitor = get_cpu_monitor()
        print(f"CPU count: {monitor.cpu_count()}")
        print(f"Usage: {monitor.cpu_percent()*100}%")
    """
    if _is_cgroup_v2():
        return CGroupV2CPUMonitor()
    return DefaultCPUMonitor()



def _is_cgroup_v2() -> bool:
    """Detect cgroups v2 environment by checking for characteristic file."""
    return os.path.exists("/sys/fs/cgroup/cpu.stat")
