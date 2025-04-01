import time
from collections import deque
from typing import Optional

import cv2
import numpy as np


class WaveformVisualizer:
    def __init__(
        self,
        history_length: int = 500,
        sample_rate: int = 24000,
        n_fft: int = 512,
        freq_bands: int = 128,
    ):
        """Initialize the waveform visualizer"""
        self.history_length = history_length
        self.sample_rate = sample_rate
        self.n_fft = n_fft  # FFT window size
        self.freq_bands = freq_bands  # Number of frequency bands to display
        self.nyquist_freq = sample_rate // 2  # Highest frequency we can analyze

        # Initialize volume history buffer
        self.volume_history: deque[float] = deque(maxlen=history_length)
        for _ in range(history_length):
            self.volume_history.append(0)

        # For FFT smoothing
        self.prev_fft = np.zeros(freq_bands)
        self.smoothing_factor = 0.3
        self.noise_gate = 0.05  # Values below this are considered silence

        self.start_time = time.time()

    def draw_timestamp(self, canvas: np.ndarray, fps: Optional[float] = None):
        height, width = canvas.shape[:2]
        text = f"{time.time() - self.start_time:.1f}s"
        if fps is not None:
            text = f"{text} @ {fps:.1f}fps"
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2.0
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(
            text, font_face, font_scale, thickness
        )
        x = (width - text_width) // 2
        y = int((height - text_height) * 0.2 + baseline)
        cv2.putText(canvas, text, (x, y), font_face, font_scale, (0, 0, 0), thickness)

    def draw_current_wave(self, canvas: np.ndarray, audio_samples: np.ndarray) -> float:
        height, width = canvas.shape[:2]
        center_y = int(height * 0.6)

        # Convert audio to frequency domain using FFT
        normalized_samples = audio_samples.astype(np.float32) / 32767.0
        normalized_samples = normalized_samples.mean(axis=1)

        if len(normalized_samples) >= self.n_fft:
            window = np.hanning(self.n_fft)
            fft_data = np.abs(np.fft.rfft(normalized_samples[: self.n_fft] * window))

            # Compute RMS volume from frequency domain
            volume = np.sqrt(np.mean(np.square(fft_data)))
            volume = np.clip(volume * 0.5, 0, 1)  # Scale and clip

            # Rest of FFT processing for visualization
            fft_data = 20 * np.log10(fft_data + 1e-10)
            fft_data = (fft_data + 80) / 80
            fft_data = np.clip(fft_data, 0, 1)

            bands = np.array_split(fft_data, self.freq_bands)
            plot_data = np.array([band.mean() for band in bands])

            # Apply noise gate
            if volume < self.noise_gate:
                volume = 0
                plot_data = np.zeros_like(plot_data)
                self.prev_fft *= 0.5

            # Apply temporal smoothing
            self.prev_fft = (
                self.prev_fft * (1 - self.smoothing_factor) + plot_data * self.smoothing_factor
            )
        else:
            volume = 0
            self.prev_fft *= 0.5

        # Create smooth interpolated curve
        x_coords = np.linspace(0, width, self.freq_bands)
        y_coords = center_y - self.prev_fft * 150

        x_smooth = np.linspace(0, width, width)
        y_smooth = np.interp(x_smooth, x_coords, y_coords)

        # Draw the spectrum visualization
        points = np.column_stack((x_smooth, y_smooth)).astype(np.int32)
        bottom_points = np.column_stack((x_smooth, np.full_like(x_smooth, center_y))).astype(
            np.int32
        )
        wave_points = np.vstack((points, bottom_points[::-1]))

        # Draw filled area with transparency
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [wave_points.astype(np.int32)], (0, 255, 0, 50))
        cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)

        # Draw outline
        for i in range(len(points) - 1):
            cv2.line(canvas, tuple(points[i]), tuple(points[i + 1]), (0, 255, 0), 2)

        return volume

    def draw_volume_history(self, canvas: np.ndarray, current_volume: float):
        height, width = canvas.shape[:2]
        bottom_y = int(height * 0.95)
        # Apply noise gate to volume
        current_volume = current_volume if current_volume > self.noise_gate else 0

        self.volume_history.append(current_volume)
        cv2.line(canvas, (0, bottom_y), (width, bottom_y), (200, 200, 200), 1)

        volume_x = np.linspace(0, width, len(self.volume_history), dtype=int)
        volume_y = bottom_y - (np.array(self.volume_history) * 100)
        points = np.column_stack((volume_x, volume_y)).astype(np.int32)

        pts = np.vstack((points, [[width, bottom_y], [0, bottom_y]])).astype(np.int32)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], (255, 0, 0, 30))
        cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)

        for i in range(len(points) - 1):
            cv2.line(canvas, tuple(points[i]), tuple(points[i + 1]), (255, 0, 0), 2)

        # Draw volume label
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
        self.draw_timestamp(canvas, fps)
        volume = self.draw_current_wave(canvas, audio_samples)
        self.draw_volume_history(canvas, volume)
