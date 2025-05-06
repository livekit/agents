import argparse
import queue
import threading
import time

import numpy as np
import pyaudio
import torch
from loguru import logger
from transformers.pipelines import pipeline

try:
    import accelerate

    _accelerate_available = True
except ImportError:
    _accelerate_available = False

SAMPLE_RATE = 16_000  # Hz (match Whisper)
CHANNELS = 1  # mono
SAMPLE_FORMAT = pyaudio.paInt16  # 16-bit signed
CHUNK_MS = 100  # microphone chunk size in ms
SEGMENT_SEC = 5  # number of seconds between sending audio to Whisper
SILENCE_THRESHOLD = 5000  # maximum absolute value (int16) considered silence
CHUNK_LENGTH = 10


def main():
    parser = argparse.ArgumentParser(
        description="Streaming transcription with Whisper (mic input)"
    )
    parser.add_argument(
        "--mic-index",
        type=int,
        default=None,
        help="Mic device index (default: system default)",
    )
    args = parser.parse_args()

    # --- Load ASR pipeline ---
    if torch.cuda.is_available():
        device = "cuda:0"
        device_map = "auto" if _accelerate_available else None
    else:
        device = "cpu"
        device_map = None

    logger.info("Loading ASR pipeline 'TalTechNLP/whisper-large-et'...")
    if device_map:
        transcriber = pipeline(
            task="automatic-speech-recognition",
            model="TalTechNLP/whisper-large-et",
            device_map=device_map,
            chunk_length_s=CHUNK_LENGTH,
        )
    else:
        transcriber = pipeline(
            task="automatic-speech-recognition",
            model="TalTechNLP/whisper-large-et",
            device=0 if device == "cuda:0" else -1,
            chunk_length_s=CHUNK_LENGTH,
        )
    logger.info("ASR pipeline loaded.")

    audio_q: queue.Queue = queue.Queue()
    stop_event = threading.Event()

    def mic_reader():
        pa = pyaudio.PyAudio()
        frames_per_buffer = int(SAMPLE_RATE * CHUNK_MS / 1000)
        stream = pa.open(
            format=SAMPLE_FORMAT,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=frames_per_buffer,
            input_device_index=args.mic_index,
        )
        while not stop_event.is_set():
            item = stream.read(frames_per_buffer, exception_on_overflow=False)
            # logger.debug(len(item))
            audio_q.put(item)
        stream.stop_stream()
        stream.close()
        pa.terminate()

    threading.Thread(target=mic_reader, daemon=True).start()

    frames = []
    next_send = time.time() + SEGMENT_SEC

    try:
        while True:
            try:
                frames.append(audio_q.get(timeout=0.1))
            except queue.Empty:
                pass
            if time.time() >= next_send:
                raw = b"".join(frames)
                frames.clear()
                next_send += SEGMENT_SEC
                if not raw:
                    continue

                # --- silence detection ---
                audio_int16 = np.frombuffer(raw, dtype=np.int16)
                noise = np.max(np.abs(audio_int16))
                if noise < SILENCE_THRESHOLD:
                    logger.info(f"silence {noise}")
                    continue  # don't send anything to the model
                logger.info(noise)
                # Convert raw bytes (int16) to normalized NumPy float32
                audio_np = audio_int16.astype(np.float32) / 32768.0

                # Transcribe
                try:
                    result = transcriber(audio_np)
                    logger.info(f"{noise}: {result.get('text', '')}")
                except Exception as e:
                    logger.error(f"Error en la transcripción: {e}")
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        time.sleep(0.2)


if __name__ == "__main__":
    main()