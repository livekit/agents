"""Render deterministic listening fixtures using Rust, untreated PCM, and actual libwebrtc.

Requires cargo, ffmpeg, a C++20 compiler, and an extracted libwebrtc distribution.
No downloads or credentials. Run from any directory; all outputs go to --output.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import wave
from pathlib import Path

CRATE = Path(__file__).resolve().parents[1]
WEBRTC_TAG = "webrtc-a545de4"
SCENARIOS = (
    "clean",
    "late_20ms",
    "late_20ms_no_reserve",
    "repeated_stalls",
    "burst_recovery",
    "no_catchup",
    "long_outage",
    "variable_chunks",
    "clock_drift",
)


def run(*args: str | Path) -> str:
    completed = subprocess.run([str(arg) for arg in args], text=True, capture_output=True)
    if completed.returncode:
        sys.stderr.write(completed.stderr)
        sys.stderr.write(completed.stdout)
        completed.check_returncode()
    return completed.stdout


def encode(raw: Path, destination: Path, rate: int = 24000) -> None:
    with wave.open(str(destination), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(raw.read_bytes())


def schedule(name: str, count: int) -> list[tuple[int, int, int]]:
    """Ordered complete source delivery. Delays hold subsequent chunks, never reorder them."""
    if name == "late_20ms_no_reserve":
        name = "late_20ms"
    result = []
    start = 0
    previous = 0
    index = 0
    while start < count:
        duration_ms = (20, 100, 40, 200, 60)[index % 5] if name == "variable_chunks" else 100
        length = min(duration_ms * 24, count - start)
        nominal = start * 1_000_000 // 24000
        delay = 0
        if name == "late_20ms" and index == 20:
            delay = 20_000
        elif name == "repeated_stalls" and index % 9 == 4:
            delay = 40_000
        elif name == "burst_recovery" and 20 <= index <= 23:
            delay = (24 - index) * 100_000
        elif name == "no_catchup" and index >= 20:
            delay = 40_000
        elif name == "long_outage" and 20 <= index <= 29:
            delay = (30 - index) * 100_000
        elif name == "clock_drift":
            delay = nominal // 1000  # sender is 1000 ppm slower than the receiver
        arrival = max(previous, nominal + delay)
        result.append((arrival, start, length))
        previous = arrival
        start += length
        index += 1
    return result


def compile_reference(root: Path, destination: Path) -> None:
    source = CRATE / "tools/neteq_reference.cc"
    include = root / "include"
    library = root / "lib/libwebrtc.a"
    if not library.is_file() or not (include / "api/neteq/neteq.h").is_file():
        raise ValueError("--webrtc must contain include/api/neteq/neteq.h and lib/libwebrtc.a")
    arguments = [
        "clang++",
        "-std=c++20",
        "-O2",
        "-DNDEBUG",
        "-DWEBRTC_POSIX",
        "-Wno-nullability-completeness",
        f"-I{include}",
        f"-I{include / 'third_party/abseil-cpp'}",
    ]
    if platform.system() == "Darwin":
        arguments.append("-DWEBRTC_MAC")
    elif platform.system() == "Linux":
        arguments.append("-DWEBRTC_LINUX")
    else:
        raise ValueError("the reference build helper supports macOS and Linux")
    arguments.extend([str(source), str(library)])
    if platform.system() == "Darwin":
        for framework in (
            "Foundation",
            "AudioToolbox",
            "CoreAudio",
            "CoreMedia",
            "CoreVideo",
            "VideoToolbox",
            "AVFoundation",
        ):
            arguments.extend(["-framework", framework])
    else:
        arguments.extend(["-lpthread", "-ldl"])
    run(*arguments, "-o", destination)


def render_player(out: Path, report: dict) -> None:
    """One HTML player, with per-cell deep links; serves media inline with no download links."""
    import html

    labels = {
        "clean": ("Clean delivery", "No failure; compare the startup latency of buffering."),
        "late_20ms": ("20 ms late delivery", "A 20 ms reserve can absorb this gap."),
        "late_20ms_no_reserve": (
            "20 ms late, no reserve",
            "Exposes short concealment and recovery at 2 seconds.",
        ),
        "repeated_stalls": ("Repeated 40 ms stalls", "Tests reserve adaptation and recovery."),
        "burst_recovery": (
            "400 ms stall, then burst",
            "Tests backlog recovery and accumulated latency.",
        ),
        "no_catchup": (
            "Permanent +40 ms delay",
            "Delivery resumes at realtime without faster catch-up.",
        ),
        "long_outage": (
            "One-second outage",
            "All small reserves run out; tests fade and recovery.",
        ),
        "variable_chunks": (
            "Chunks change 20–200 ms",
            "Changing chunk size should not cause an underrun.",
        ),
        "clock_drift": ("Sender 1000 ppm slower", "Small rate mismatch accumulates over time."),
    }
    tracks = {
        "baseline": "No NetEq",
        "buffered": "100 ms FIFO",
        "neteq": "libwebrtc NetEq",
        "candidate": "Rust controller",
    }
    rows = []
    menu = []
    for name, case in report["scenarios"].items():
        label, description = labels[name]
        menu.append(f'<option value="{name}">{html.escape(label)}</option>')
        cells = []
        for track, title in tracks.items():
            if track in ("baseline", "buffered"):
                metrics = case["fifo_metrics"][track]
                detail = f"{metrics['missing_samples'] / 24:g} ms gaps · {metrics['startup_ms']} ms startup"
            elif track == "candidate":
                detail = f"{case['rust_metrics']['concealed'] / 24:g} ms concealed · {case.get('delay_ms', report['delay_ms'])} ms startup"
            else:
                detail = "Adaptive delay · reference DSP"
            cells.append(
                f'<td><a class="play" href="?scenario={name}&track={track}#player">Play {title}</a><small>{detail}</small></td>'
            )
        rows.append(
            f"<tr><th>{html.escape(label)}<small>{html.escape(description)}</small></th>{''.join(cells)}</tr>"
        )
    payload = json.dumps(
        {
            "cases": {name: {"label": labels[name][0]} for name in report["scenarios"]},
            "tracks": tracks,
            "source": report.get("source_path", "speech.wav"),
        }
    ).replace("<", "\\u003c")
    (out / "index.html").write_text(
        """<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Streaming audio comparison</title>
<style>
:root{color-scheme:light dark}body{font:16px system-ui;max-width:1250px;margin:32px auto;padding:0 20px;line-height:1.5}h1{font-size:1.8rem;margin-bottom:8px}p{max-width:1000px}.player{position:sticky;top:0;background:Canvas;padding:16px 0;border-bottom:1px solid #888;z-index:1}audio{width:min(100%,700px);display:block;margin-top:12px}select,button{font:inherit;padding:8px;margin:4px 6px 4px 0}table{border-collapse:collapse;width:100%;margin-top:20px}td,th{padding:14px 10px;border-bottom:1px solid #8885;text-align:left;vertical-align:top}th{max-width:250px}small{display:block;font-size:.875rem;font-weight:normal;opacity:.8;margin-top:5px}a{color:LinkText}td a{white-space:nowrap}.scroll{overflow:auto}#status{min-height:1.5em}
</style><h1>Streaming audio comparison</h1>
<p>One sentence generated by <strong>GPTLive · marin</strong>. Each row uses the same audio and ordered delivery schedule. Compare actual libwebrtc NetEq, the Rust controller, untreated playback, and a plain 100 ms FIFO buffer.</p>
<section class="player" id="player"><label for="scenario">Scenario</label><select id="scenario">"""
        + "".join(menu)
        + """</select>
<label for="track">Playback</label><select id="track">"""
        + "".join(f'<option value="{key}">{value}</option>' for key, value in tracks.items())
        + """</select>
<button id="start">Play from start</button><button id="gap">Play around stall</button><button id="original">Original sentence</button>
<audio id="audio" controls preload="metadata"></audio><div id="status" role="status"></div></section>
<p>Start at zero to compare latency. For the short-stall cases, the interruption is at 2 seconds. The Playback menu switches at the same wall-clock position; these are not speech-aligned quality scores. Gaps exclude intentional startup and the trailing observation window. Concealed time can include generated fades. NetEq alone uses offline 24→48→24 kHz conversion; its delay is adaptive.</p>
<div class="scroll"><table><thead><tr><th>Scenario / what to listen for</th><th>No NetEq</th><th>100 ms FIFO</th><th>libwebrtc NetEq</th><th>Rust controller</th></tr></thead><tbody>"""
        + "".join(rows)
        + """</tbody></table></div>
<p>All audio uses native browser playback controls. No downloads are required. <a href="report.json">Trace provenance and metrics</a>.</p>
<script id="data" type="application/json">"""
        + payload
        + """</script><script>
const data=JSON.parse(document.getElementById('data').textContent);
const audio=document.getElementById('audio'), scenario=document.getElementById('scenario'), track=document.getElementById('track'), status=document.getElementById('status');
const query=new URLSearchParams(location.search);
if(data.cases[query.get('scenario')])scenario.value=query.get('scenario');
if(data.tracks[query.get('track')])track.value=query.get('track');
function select(time=0,play=false){
 const name=scenario.value, variant=track.value;
 audio.src=`${name}/${variant}.wav`;
 audio.currentTime=time;
 status.textContent=`${data.cases[name].label} · ${data.tracks[variant]}`;
 history.replaceState(null,'',`?scenario=${name}&track=${variant}#player`);
 if(play)audio.play().catch(()=>{status.textContent+=' · Press the browser Play control.';});
}
scenario.addEventListener('change',()=>select(0,false));
track.addEventListener('change',()=>select(audio.currentTime,!audio.paused));
document.getElementById('start').onclick=()=>select(0,true);
document.getElementById('gap').onclick=()=>select(1.6,true);
document.getElementById('original').onclick=()=>{audio.src=data.source;status.textContent='Original GPTLive sentence';audio.play().catch(()=>{});};
document.querySelectorAll('a.play').forEach(link=>link.onclick=event=>{event.preventDefault();const next=new URL(link.href);scenario.value=next.searchParams.get('scenario');track.value=next.searchParams.get('track');select(0,true);document.getElementById('player').scrollIntoView();});
audio.addEventListener('error',()=>{status.textContent='Audio could not load. Please reload this page.';});
select(0,false);
</script></html>"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--webrtc",
        type=Path,
        required=True,
        help=f"extracted {WEBRTC_TAG} distribution (see fixture README)",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="external mono 24 kHz PCM16 WAV")
    parser.add_argument("--delay-ms", type=int, default=20)
    parser.add_argument("--scenario", choices=SCENARIOS, action="append")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    reference = out / "neteq_reference"
    compile_reference(args.webrtc.resolve(), reference)
    fixture = args.input.resolve()
    with wave.open(str(fixture), "rb") as source:
        if (source.getframerate(), source.getnchannels(), source.getsampwidth()) != (24000, 1, 2):
            raise ValueError("fixture must be mono 24 kHz signed PCM16")
        pcm = source.readframes(source.getnframes())
    raw = out / "source.pcm"
    raw.write_bytes(pcm)
    source48 = out / "source48.pcm"
    run("ffmpeg", "-v", "error", "-y", "-i", fixture, "-ar", "48000", "-f", "s16le", source48)
    duration_ms = len(pcm) * 1000 // 48000 + 2000
    manifest = CRATE.parents[1] / "Cargo.toml"
    # Cargo's JSON output supplies the executable even when CARGO_TARGET_DIR is set.
    built = run(
        "cargo",
        "build",
        "--offline",
        "--manifest-path",
        manifest,
        "-p",
        "lk-agents-media",
        "--example",
        "playout",
        "--release",
        "--message-format=json",
    )
    executables = [
        event["executable"]
        for row in built.splitlines()
        if (event := json.loads(row)).get("executable")
    ]
    if len(executables) != 1:
        raise ValueError("expected one playout example executable")
    executable = executables[0]
    report = {
        "source_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "reference_archive_sha256": hashlib.sha256(
            (args.webrtc / "lib/libwebrtc.a").read_bytes()
        ).hexdigest(),
        "reference_expected_tag": WEBRTC_TAG,
        "delay_ms": args.delay_ms,
        "scenarios": {},
    }
    shutil.copyfile(fixture, out / "speech.wav")
    for name in args.scenario or SCENARIOS:
        case = out / name
        case.mkdir(exist_ok=True)
        arrivals = case / "arrivals.csv"
        with arrivals.open("w", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(("arrival_us", "source_start", "samples"))
            writer.writerows(schedule(name, len(pcm) // 2))
        case_delay = 0 if name == "late_20ms_no_reserve" else args.delay_ms
        run(executable, raw, arrivals, case, str(duration_ms), str(case_delay))
        reference_stats = run(
            reference,
            source48,
            arrivals,
            case / "neteq48.pcm",
            str(duration_ms),
            str(case_delay),
        )
        # NetEq L16 supports 8/16/32/48 kHz, not 24 kHz. Resampling happens offline on the
        # reference branch only; document it rather than claiming sample equivalence.
        run(
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "s16le",
            "-ar",
            "48000",
            "-ac",
            "1",
            "-i",
            case / "neteq48.pcm",
            "-ar",
            "24000",
            "-f",
            "s16le",
            case / "neteq.pcm",
        )
        for label in ("baseline", "buffered", "candidate", "neteq"):
            encode(case / f"{label}.pcm", case / f"{label}.wav")
        shutil.copyfile(fixture, case / "original.wav")
        # Stereo comparison preserves wall-clock alignment: baseline left, candidate right.
        run(
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            case / "baseline.wav",
            "-i",
            case / "candidate.wav",
            "-filter_complex",
            "[0:a][1:a]join=inputs=2:channel_layout=stereo[a]",
            "-map",
            "[a]",
            case / "baseline_vs_candidate.wav",
        )
        run(
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            case / "neteq.wav",
            "-i",
            case / "candidate.wav",
            "-filter_complex",
            "[0:a][1:a]join=inputs=2:channel_layout=stereo[a]",
            "-map",
            "[a]",
            case / "neteq_vs_candidate.wav",
        )
        with (case / "timeline.csv").open() as stream:
            final = list(csv.DictReader(stream))[-1]
        with (case / "fifo_statistics.csv").open() as stream:
            fifo_metrics = {
                row["track"]: {k: int(v) for k, v in row.items() if k != "track"}
                for row in csv.DictReader(stream)
            }
        report["scenarios"][name] = {
            "delay_ms": case_delay,
            "rust_metrics": {k: int(v) for k, v in final.items()},
            "fifo_metrics": fifo_metrics,
            "rust": (case / "statistics.txt").read_text(),
            "neteq": reference_stats,
            "fifo": (case / "fifo_statistics.txt").read_text(),
        }
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    render_player(out, report)
    print(out / "index.html")


if __name__ == "__main__":
    main()
