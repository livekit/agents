"""
Implements an efficient fixed-size moving average algorithm using circular buffer.
Designed for real-time metrics tracking in voice/audio processing pipelines.

Features:
- O(1) time complexity for adds/updates
- Fixed memory footprint
- Thread-safe for single producer/consumer
- Avoids floating point precision issues

Common use cases:
- Audio level monitoring
- Network jitter calculation
- Frame processing time tracking
- Packet loss percentage estimation
"""

from __future__ import annotations


class MovingAverage:
    """Efficient moving window average calculator.
    
    Usage:
        # Track last 100 samples
        avg = MovingAverage(window_size=100)
        while True:
            avg.add_sample(get_latency())
            current_avg = avg.get_avg()
    """
    
    def __init__(self, window_size: int) -> None:
        """
        Args:
            window_size: Number of samples to consider in rolling window
                        (uses circular buffer implementation)
        """
        self._hist: list[float] = [0] * window_size  # Circular buffer
        self._sum: float = 0  # Running total of current window
        self._count: int = 0  # Total samples seen

    def add_sample(self, sample: float) -> None:
        """Add new measurement to the average window.
        
        Args:
            sample: New value to include in average (e.g. latency in seconds)
        """
        self._count += 1
        index = self._count % len(self._hist)
        
        # Remove oldest value if window full
        if self._count > len(self._hist):
            self._sum -= self._hist[index]
            
        self._sum += sample
        self._hist[index] = sample

    def get_avg(self) -> float:
        """Get current average of samples in window.
        
        Returns:
            Average of samples in current window, or 0 if no samples
        """
        if self._count == 0:
            return 0.0
        return self._sum / self.size()

    def reset(self) -> None:
        """Clear all historical data and reset counters."""
        self._count = 0
        self._sum = 0.0

    def size(self) -> int:
        """Get number of samples currently in window.
        
        Returns:
            Actual sample count (until window fills), then window_size
        """
        return min(self._count, len(self._hist))
