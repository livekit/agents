"""
Implements an exponential smoothing filter with dynamic smoothing factors.
Designed for real-time smoothing of time-series data in media processing pipelines.

Common use cases:
- Audio level smoothing
- Network metric stabilization
- Sensor data filtering
- Adaptive threshold calculation
"""

class ExpFilter:
    """Exponential smoothing filter with configurable decay and clamping.
    
    Features:
    - Time-varying smoothing via exponential factors
    - Optional maximum value clamping
    - State reset capabilities
    
    Usage:
        # Smooth audio levels with alpha=0.9 and max level=1.0
        filter = ExpFilter(alpha=0.9, max_val=1.0)
        while True:
            raw_level = get_audio_level()
            smoothed = filter.apply(exp=1.0, sample=raw_level)
    """
    
    def __init__(self, alpha: float, max_val: float = -1.0) -> None:
        """
        Args:
            alpha: Base smoothing factor (0.0-1.0)
                   Higher values = more smoothing
            max_val: Maximum allowed value (disabled if <= 0)
        """
        self._alpha = alpha
        self._filtered = -1.0  # Initial state flag
        self._max_val = max_val

    def reset(self, alpha: float = -1.0) -> None:
        """Reset filter state while optionally updating alpha.
        
        Args:
            alpha: New alpha value if >= 0 (keeps current if -1)
        """
        if alpha != -1.0:
            self._alpha = alpha
        self._filtered = -1.0  # Reset to initial state

    def apply(self, exp: float, sample: float) -> float:
        """Update filter with new sample using exponential smoothing.
        
        Args:
            exp: Exponent applied to alpha for this update
                 Higher values increase smoothing effect temporarily
            sample: New input value to incorporate
        
        Returns:
            Current filtered value
        """
        if self._filtered == -1.0:  # Initial state
            self._filtered = sample
        else:
            a = self._alpha**exp  # Dynamic smoothing factor
            self._filtered = a * self._filtered + (1 - a) * sample

        if self._max_val > 0 and self._filtered > self._max_val:
            self._filtered = self._max_val  # Clamp value

        return self._filtered

    def filtered(self) -> float:
        """Get current filtered value without updating."""
        return self._filtered

    def update_base(self, alpha: float) -> None:
        """Update base alpha while preserving current state."""
        self._alpha = alpha
