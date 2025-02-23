import time
from collections import deque
from typing import Optional

import cv2
import numpy as np


class WaveformVisualizer:
    """Real-time audio waveform and volume visualization renderer.
    
    Features:
    - Frequency spectrum visualization using FFT
    - RMS volume history tracking
    - Smooth temporal filtering of visual elements
    - Timestamp and FPS display
    - Configurable visual styling
    """
    
    def __init__(
        self,
        history_length: int = 500,
        sample_rate: int = 24000,
        n_fft: int = 512,
        freq_bands: int = 128,
    ):
        """Initialize the waveform visualizer with audio processing parameters.
        
        Args:
            history_length: Number of volume history points to track
            sample_rate: Input audio sample rate (Hz)
            n_fft: FFT window size for frequency analysis
            freq_bands: Number of frequency bands to display
        """
        self.history_length = history_length
        self.sample_rate = sample_rate
        self.n_fft = n_fft  # FFT window size
        self.freq_bands = freq_bands  # Frequency resolution for display
        self.nyquist_freq = sample_rate // 2  # Maximum displayable frequency

        # Volume tracking
        self.volume_history: deque[float] = deque(maxlen=history_length)
        self.volume_history.extend([0] * history_length)  # Initialize with silence

        # FFT processing state
        self.prev_fft = np.zeros(freq_bands)
        self.smoothing_factor = 0.3  # EMA smoothing factor for FFT data
        self.noise_gate = 0.05  # Minimum volume threshold for visualization

        # Timing and animation
        self.start_time = time.time()

    def draw_timestamp(self, canvas: np.ndarray, fps: Optional[float] = None):
        """Draw current timestamp and FPS counter at top center of canvas.
        
        Args:
            canvas: RGBA numpy array to draw on (modified in-place)
            fps: Optional frames-per-second value to display
        """
        height, width = canvas.shape[:2]
        text = f"{time.time() - self.start_time:.1f}s"
        if fps is not None:
            text = f"{text} @ {fps:.1f}fps"
            
        # Configure text styling
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        thickness = 2
        color = (0, 0, 0)  # Black text

        # Calculate text positioning
        (text_width, text_height), baseline = cv2.getTextSize(
            text, font_face, font_scale, thickness
        )
        x = (width - text_width) // 2
        y = int((height - text_height) * 0.2 + baseline)
        
        # Render text to canvas
        cv2.putText(canvas, text, (x, y), font_face, font_scale, color, thickness)

    def draw_current_wave(self, canvas: np.ndarray, audio_samples: np.ndarray) -> float:
        """Process audio and draw frequency spectrum visualization.
        
        Args:
            canvas: RGBA numpy array to draw on (modified in-place)
            audio_samples: Input audio data as numpy array
            
        Returns:
            Current RMS volume (0-1 scale)
        """
        height, width = canvas.shape[:2]
        center_y = int(height * 0.6)  # Vertical center for waveform

        # Convert to mono and normalize
        normalized_samples = audio_samples.astype(np.float32) / 32767.0
        normalized_samples = normalized_samples.mean(axis=1)

        volume = 0.0
        if len(normalized_samples) >= self.n_fft:
            # Apply FFT to analyze frequency content
            window = np.hanning(self.n_fft)
            fft_data = np.abs(np.fft.rfft(normalized_samples[:self.n_fft] * window))

            # Calculate RMS volume from frequency data
            volume = np.sqrt(np.mean(np.square(fft_data)))
            volume = np.clip(volume * 0.5, 0, 1)  # Scale to 0-1 range

            # Convert to dB scale and normalize
            fft_data = 20 * np.log10(fft_data + 1e-10)  # Avoid log(0)
            fft_data = (fft_data + 80) / 80  # Normalize to 0-1 range
            fft_data = np.clip(fft_data, 0, 1)

            # Group frequencies into bands for visualization
            bands = np.array_split(fft_data, self.freq_bands)
            plot_data = np.array([band.mean() for band in bands])

            # Apply noise gate and temporal smoothing
            if volume < self.noise_gate:
                volume = 0
                plot_data = np.zeros_like(plot_data)
                self.prev_fft *= 0.5  # Fast decay when silent

            # Exponential moving average for smooth animation
            self.prev_fft = (
                self.prev_fft * (1 - self.smoothing_factor)
                + plot_data * self.smoothing_factor
            )
        else:
            # Insufficient samples - fade existing visualization
            volume = 0
            self.prev_fft *= 0.5

        # Generate smooth waveform curve
        x_coords = np.linspace(0, width, self.freq_bands)
        y_coords = center_y - self.prev_fft * 150  # Scale amplitude for display

        # Create interpolated curve points
        x_smooth = np.linspace(0, width, width)
        y_smooth = np.interp(x_smooth, x_coords, y_coords)
        points = np.column_stack((x_smooth, y_smooth)).astype(np.int32)

        # Draw filled waveform area
        bottom_points = np.column_stack(
            (x_smooth, np.full_like(x_smooth, center_y))
        ).astype(np.int32)
        wave_points = np.vstack((points, bottom_points[::-1]))
        
        # Create translucent overlay
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [wave_points.astype(np.int32)], (0, 255, 0, 50))
        cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)

        # Draw waveform outline
        for i in range(len(points) - 1):
            cv2.line(canvas, tuple(points[i]), tuple(points[i + 1]), (0, 255, 0), 2)

        # Draw frequency scale markers
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        freq_points = [0, 1000, 2000, 4000, 8000, self.nyquist_freq]
        
        for freq in freq_points:
            x_pos = int(width * freq / self.nyquist_freq)
            cv2.putText(
                canvas,
                f"{freq}Hz",
                (x_pos, center_y + 20),
                font_face,
                font_scale,
                (100, 100, 100),
                thickness,
            )

        return volume

    def draw_volume_history(self, canvas: np.ndarray, current_volume: float):
        """Draw volume history graph at bottom of canvas.
        
        Args:
            canvas: RGBA numpy array to draw on (modified in-place)
            current_volume: Current volume value (0-1 scale)
        """
        height, width = canvas.shape[:2]
        bottom_y = int(height * 0.95)  # Position for volume graph
        
        # Apply noise gate to current volume
        current_volume = current_volume if current_volume > self.noise_gate else 0
        self.volume_history.append(current_volume)

        # Draw baseline
        cv2.line(canvas, (0, bottom_y), (width, bottom_y), (200, 200, 200), 1)

        # Generate volume curve points
        volume_x = np.linspace(0, width, len(self.volume_history), dtype=int)
        volume_y = bottom_y - (np.array(self.volume_history) * 100)
        points = np.column_stack((volume_x, volume_y)).astype(np.int32)

        # Create filled area below curve
        pts = np.vstack((points, [[width, bottom_y], [0, bottom_y]])).astype(np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], (255, 0, 0, 30))
        cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)

        # Draw volume curve
        for i in range(len(points) - 1):
            cv2.line(canvas, tuple(points[i]), tuple(points[i + 1]), (255, 0, 0), 2)

        # Add volume label
        cv2.putText(
            canvas,
            "Volume",
            (10, bottom_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (100, 100, 100),
            1,
        )

    def draw(
        self,
        canvas: np.ndarray,
        audio_samples: np.ndarray,
        fps: Optional[float] = None,
    ):
        """Main drawing method combining all visualization elements.
        
        Args:
            canvas: RGBA numpy array to draw on (modified in-place)
            audio_samples: Input audio data as numpy array
            fps: Optional frames-per-second value to display
        """
        self.draw_timestamp(canvas, fps)
        volume = self.draw_current_wave(canvas, audio_samples)
        self.draw_volume_history(canvas, volume)
