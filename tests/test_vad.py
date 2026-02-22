import time

import numpy as np
import pytest

from livekit.agents import vad
from livekit.plugins import silero
from livekit.plugins.silero import onnx_model, vad as silero_vad

try:
    from .utils import make_test_speech, make_wav_file
except ImportError:
    from utils import make_test_speech, make_wav_file

SAMPLE_RATES = [16000, 44100]  # test multiple input sample rates


VAD = silero.VAD.load(
    min_speech_duration=0.5,
    min_silence_duration=0.75,
)


def test_vad_load_without_warmup_does_not_run_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    warmup_calls = {"count": 0}

    def _fake_new_inference_session(_force_cpu: bool, onnx_file_path=None) -> object:
        del onnx_file_path
        return object()

    class FakeOnnxModel:
        window_size_samples = 512

        def __init__(self, *, onnx_session: object, sample_rate: int) -> None:
            del onnx_session, sample_rate

        def __call__(self, _x) -> float:
            warmup_calls["count"] += 1
            return 0.0

    monkeypatch.setattr(onnx_model, "new_inference_session", _fake_new_inference_session)
    monkeypatch.setattr(onnx_model, "OnnxModel", FakeOnnxModel)

    silero_vad.VAD.load(warmup=0)
    assert warmup_calls["count"] == 0


def test_vad_load_with_warmup_runs_n_inferences(monkeypatch: pytest.MonkeyPatch) -> None:
    warmup_calls = {"count": 0}

    def _fake_new_inference_session(_force_cpu: bool, onnx_file_path=None) -> object:
        del onnx_file_path
        return object()

    class FakeOnnxModel:
        window_size_samples = 512

        def __init__(self, *, onnx_session: object, sample_rate: int) -> None:
            del onnx_session, sample_rate

        def __call__(self, _x) -> float:
            warmup_calls["count"] += 1
            return 0.0

    monkeypatch.setattr(onnx_model, "new_inference_session", _fake_new_inference_session)
    monkeypatch.setattr(onnx_model, "OnnxModel", FakeOnnxModel)

    silero_vad.VAD.load(warmup=3)
    assert warmup_calls["count"] == 3


@pytest.mark.parametrize(
    ("sample_rate", "expected_window_size"),
    [(8000, 256), (16000, 512)],
)
def test_vad_load_with_warmup_uses_expected_window_size(
    monkeypatch: pytest.MonkeyPatch,
    sample_rate: int,
    expected_window_size: int,
) -> None:
    warmup_calls = {"shape": None}

    def _fake_new_inference_session(_force_cpu: bool, onnx_file_path=None) -> object:
        del onnx_file_path
        return object()

    class FakeOnnxModel:
        window_size_samples = expected_window_size

        def __init__(self, *, onnx_session: object, sample_rate: int) -> None:
            del onnx_session, sample_rate

        def __call__(self, x) -> float:
            warmup_calls["shape"] = x.shape
            return 0.0

    monkeypatch.setattr(onnx_model, "new_inference_session", _fake_new_inference_session)
    monkeypatch.setattr(onnx_model, "OnnxModel", FakeOnnxModel)

    silero_vad.VAD.load(sample_rate=sample_rate, warmup=1)
    assert warmup_calls["shape"] == (expected_window_size,)


def test_vad_load_with_warmup_propagates_warmup_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_new_inference_session(_force_cpu: bool, onnx_file_path=None) -> object:
        del onnx_file_path
        return object()

    class FakeOnnxModel:
        window_size_samples = 512

        def __init__(self, *, onnx_session: object, sample_rate: int) -> None:
            del onnx_session, sample_rate

        def __call__(self, _x) -> float:
            raise RuntimeError("warmup failed")

    monkeypatch.setattr(onnx_model, "new_inference_session", _fake_new_inference_session)
    monkeypatch.setattr(onnx_model, "OnnxModel", FakeOnnxModel)

    with pytest.raises(RuntimeError, match="warmup failed"):
        silero_vad.VAD.load(warmup=1)


def test_vad_load_with_negative_warmup_raises() -> None:
    with pytest.raises(ValueError, match="warmup must be greater than or equal to 0"):
        silero_vad.VAD.load(warmup=-1)


def test_vad_load_with_non_integer_warmup_raises() -> None:
    with pytest.raises(TypeError, match="warmup must be an integer"):
        silero_vad.VAD.load(warmup=1.5)  # type: ignore[arg-type]


@pytest.mark.parametrize("sample_rate", SAMPLE_RATES)
async def test_chunks_vad(sample_rate) -> None:
    frames, *_ = await make_test_speech(chunk_duration_ms=10, sample_rate=sample_rate)
    assert len(frames) > 1, "frames aren't chunked"

    stream = VAD.stream()

    for frame in frames:
        stream.push_frame(frame)

    stream.end_input()

    start_of_speech_i = 0
    end_of_speech_i = 0

    inference_frames = []

    async for ev in stream:
        if ev.type == vad.VADEventType.START_OF_SPEECH:
            with open(
                f"test_vad.{sample_rate}.start_of_speech_frames_{start_of_speech_i}.wav",
                "wb",
            ) as f:
                f.write(make_wav_file(ev.frames))

            start_of_speech_i += 1

        if ev.type == vad.VADEventType.INFERENCE_DONE:
            inference_frames.extend(ev.frames)

        if ev.type == vad.VADEventType.END_OF_SPEECH:
            with open(
                f"test_vad.{sample_rate}.end_of_speech_frames_{end_of_speech_i}.wav",
                "wb",
            ) as f:
                f.write(make_wav_file(ev.frames))

            end_of_speech_i += 1

    assert start_of_speech_i > 0, "no start of speech detected"
    assert start_of_speech_i == end_of_speech_i, "start and end of speech mismatch"

    with open(f"test_vad.{sample_rate}.inference_frames.wav", "wb") as f:
        f.write(make_wav_file(inference_frames))


@pytest.mark.parametrize("sample_rate", SAMPLE_RATES)
async def test_file_vad(sample_rate) -> None:
    frames, *_ = await make_test_speech(sample_rate=sample_rate)
    assert len(frames) == 1, "one frame should be the whole audio"

    stream = VAD.stream()

    for frame in frames:
        stream.push_frame(frame)

    stream.end_input()

    start_of_speech_i = 0
    end_of_speech_i = 0
    async for ev in stream:
        if ev.type == vad.VADEventType.START_OF_SPEECH:
            start_of_speech_i += 1

        if ev.type == vad.VADEventType.END_OF_SPEECH:
            end_of_speech_i += 1

    assert start_of_speech_i > 0, "no start of speech detected"
    assert start_of_speech_i == end_of_speech_i, "start and end of speech mismatch"


def _benchmark_inference(*, warmup: int, n: int = 10) -> list[float]:
    vad_instance = silero_vad.VAD.load(warmup=warmup)
    model = onnx_model.OnnxModel(
        onnx_session=vad_instance._onnx_session,
        sample_rate=vad_instance._opts.sample_rate,
    )
    input_data = np.zeros(model.window_size_samples, dtype=np.float32)

    durations_ms: list[float] = []
    for _ in range(n):
        start_time = time.perf_counter()
        model(input_data)
        durations_ms.append((time.perf_counter() - start_time) * 1000)

    return durations_ms


def _print_benchmark_results(name: str, durations_ms: list[float]) -> None:
    avg = sum(durations_ms) / len(durations_ms)
    print(f"{name}:")
    print(f"  calls={len(durations_ms)}")
    print(f"  avg_ms={avg:.3f}")
    print(f"  min_ms={min(durations_ms):.3f}")
    print(f"  max_ms={max(durations_ms):.3f}")
    print(f"  per_call_ms={[round(v, 3) for v in durations_ms]}")


if __name__ == "__main__":
    n = 10 # no of iterations to benchmark
    m = 3 # warmup calls
    print(f"Silero VAD inference benchmark (N={n})")
    with_warmup = _benchmark_inference(warmup=m, n=n)
    # Allow for a cooldown for a fair comparison
    time.sleep(60)
    without_warmup = _benchmark_inference(warmup=0, n=n)
    _print_benchmark_results("without_warmup", without_warmup)
    _print_benchmark_results(f"with_warmup({m})", with_warmup)
