from livekit import rtc


def resample_audio(audio_bytes: bytes, original_sample_rate: int, target_sample_rate: int) -> bytes:
    if len(audio_bytes) % 2:
        raise ValueError("16-bit PCM data must contain complete samples")
    if not audio_bytes or original_sample_rate == target_sample_rate:
        return audio_bytes
    resampler = rtc.AudioResampler(original_sample_rate, target_sample_rate, num_channels=1)
    frames = resampler.push(bytearray(audio_bytes))
    frames.extend(resampler.flush())
    return b"".join(frame.data.tobytes() for frame in frames)
