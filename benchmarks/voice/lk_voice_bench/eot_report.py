"""Where the end-of-turn delay goes, from the spans in agent logs (``LK_BENCH_TRACE=1`` runs).

    python -m lk_voice_bench.eot_report <run-dir> [<run-dir> ...] [--replay-vad]

Per committed user turn it reads the ``eou_detection`` span (what triggered the end-of-turn check,
the EOT model's probability against its "unlikely" threshold, the endpointing delay that chose)
and the ``user_turn`` span (transcription and end-of-turn delays). ``--replay-vad`` re-runs the
VAD over each session's tapped caller audio (``LK_BENCH_SAVE_AUDIO=1``) at full speed and
compares how much speech it finds with what the live VAD reported during the call.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import wave
from collections import Counter
from pathlib import Path
from typing import Any


def _median(xs: list[float]) -> str:
    return f"{statistics.median(xs):.2f}" if xs else "-"


def _sessions(run_dir: Path) -> list[tuple[Path, list[dict[str, Any]]]]:
    out = []
    for path in sorted((run_dir / "agent_logs").glob("*.jsonl")):
        out.append((path, [json.loads(line) for line in path.read_text().splitlines()]))
    return out


def eot_breakdown(run_dir: Path) -> dict[str, Any]:
    source: Counter[str] = Counter()
    verdict: Counter[str] = Counter()
    probs: list[float] = []
    delay_by_verdict: dict[str, list[float]] = {"likely": [], "unlikely": []}
    transcription_delay: list[float] = []
    end_of_turn_delay: list[float] = []
    turns_without_vad = 0
    turns = 0
    live_vad_speech_s = 0.0
    for _, events in _sessions(run_dir):
        spans = [e for e in events if e["kind"] == "span"]
        speaking = [s for s in spans if s["name"] == "user_speaking"]
        live_vad_speech_s += sum(s["end"] - s["start"] for s in speaking)
        for s in spans:
            a = s["attributes"]
            if s["name"] == "eou_detection":
                source[a.get("lk.eou.source", "?")] += 1
                p, th = a.get("lk.eou.probability"), a.get("lk.eou.unlikely_threshold")
                if p is not None and th is not None:
                    probs.append(p)
                    v = "likely" if p >= th else "unlikely"
                    verdict[v] += 1
                    delay_by_verdict[v].append(a.get("lk.eou.endpointing_delay", 0.0))
            elif s["name"] == "user_turn":
                turns += 1
                if "lk.transcription_delay" in a:
                    transcription_delay.append(a["lk.transcription_delay"])
                if "lk.end_of_turn_delay" in a:
                    end_of_turn_delay.append(a["lk.end_of_turn_delay"])
                # a turn the VAD heard overlaps a user_speaking span
                if not any(sp["start"] < s["end"] and sp["end"] > s["start"] for sp in speaking):
                    turns_without_vad += 1
    return {
        "user_turns": turns,
        "turns_the_live_vad_never_heard": turns_without_vad,
        "live_vad_speech_s": round(live_vad_speech_s, 1),
        "eot_checks_by_trigger": dict(source),
        "eot_verdicts": dict(verdict),
        "eot_probability_p50": _median(probs),
        "endpointing_delay_when_likely_p50": _median(delay_by_verdict["likely"]),
        "endpointing_delay_when_unlikely_p50": _median(delay_by_verdict["unlikely"]),
        "transcription_delay_p50": _median(transcription_delay),
        "end_of_turn_delay_p50": _median(end_of_turn_delay),
        "end_of_turn_delay_measured_turns": len(end_of_turn_delay),
    }


async def _replay_vad(wav_path: Path) -> float:
    """Seconds of speech the VAD finds in a recording when run as fast as it can."""
    from livekit import rtc
    from livekit.agents import inference, vad

    with wave.open(str(wav_path)) as w:
        rate, data = w.getframerate(), w.readframes(w.getnframes())
    stream = inference.VAD().stream()
    step = rate // 50 * 2
    resampler = rtc.AudioResampler(rate, 16000, num_channels=1)
    for i in range(0, len(data), step):
        chunk = data[i : i + step]
        frame = rtc.AudioFrame(chunk, rate, 1, len(chunk) // 2)
        for f in resampler.push(frame):
            stream.push_frame(f)
    for f in resampler.flush():
        stream.push_frame(f)
    stream.end_input()
    speech_s = 0.0
    async for ev in stream:
        if ev.type == vad.VADEventType.END_OF_SPEECH:
            speech_s += ev.speech_duration
    await stream.aclose()
    return speech_s


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--replay-vad", action="store_true")
    args = parser.parse_args()
    for run in args.runs:
        out = eot_breakdown(run)
        if args.replay_vad:
            wavs = sorted((run / "agent_logs").glob("*.caller.wav"))
            out["replayed_vad_speech_s"] = round(sum(asyncio.run(_replay_vad(w)) for w in wavs), 1)
            out["replayed_sessions"] = len(wavs)
        print(f"== {run}")
        for k, v in out.items():
            print(f"  {k:40s} {v}")


if __name__ == "__main__":
    main()
