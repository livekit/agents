import argparse
import csv
import hashlib
import json
import math
import platform
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Literal, NamedTuple, NotRequired, TypedDict

# Compare Rust playout, untreated PCM and actual libwebrtc NetEq using the same speech
# and ordered arrival schedule. This is an optional offline tool, with no Python packages
# or automatic downloads. Requires Python 3.11+, Cargo, clang++ and FFmpeg.
#
# Download the archive for your platform from livekit/rust-sdks and extract it:
# https://github.com/livekit/rust-sdks/releases/tag/webrtc-a545de4
# Pass the extracted directory containing include/ and lib/ to --webrtc.
#
# Example:
#   python3 neteq_compare.py --input speech.wav \
#       --webrtc /path/to/mac-arm64-release --output /tmp/comparison
# A copied standalone script also accepts --crate /path/to/lk-agents-media.
# Input must be a mono 24 kHz PCM16 WAV; choose speech active at the fault onset.
#
# Writes a directory per scenario containing:
#   untreated.wav, rust.wav, neteq.wav: recordings with the original wall-clock alignment.
#   arrivals.csv: actual chunk arrival times and source sample ranges shared by both paths.
#   rust.csv, neteq.csv: per-frame queue, target, concealment and time-scaling counters.
# report.json records settings, source/archive/output hashes, added startup latency and
# operation counters before EOF. Latency uses 1 ms signal-onset windows and excludes
# model, live network and device latency. These measurements are not audio-quality scores.
#
# Only NetEq uses offline 24->48->24 kHz conversion for L16. Its native maximum target is
# unlimited by default; Rust defaults to 100 ms. Native and Rust rate controls differ.
# Temporary C++/Rust bridges are compiled locally and deleted after the run.
#
# Schedule design follows NetEq's separate timing/loss controls and fixed-seed replay:
#   modules/audio_coding/neteq/tools/neteq_quality_test.cc (fixed seed, loss events, drift)
#   modules/audio_coding/neteq/tools/rtp_generator.cc (sample-based clock drift)
#   modules/audio_coding/neteq/tools/rtp_jitter.cc (explicit arrival-time replay)
# Here the transport is reliable and ordered, so loss/reordering models are excluded.
# The report records each preset description, jitter seed and arrival-CSV hash. Retrying
# identical input/settings/toolchain reproduces the same schedule and output audio.

# These bridge sources are written only to the temporary build directory.

CPP_SOURCE = r"""

// Offline reference only. Links the pinned libwebrtc distribution; no product dependency.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#include "api/audio/audio_frame.h"
#include "api/audio_codecs/L16/audio_decoder_L16.h"
#include "api/audio_codecs/audio_decoder_factory_template.h"
#include "api/environment/environment_factory.h"
#include "api/neteq/default_neteq_factory.h"
#include "api/neteq/neteq.h"
#include "api/rtp_headers.h"
#include "api/rtp_packet_info.h"
#include "system_wrappers/include/clock.h"

struct Chunk {
  int64_t arrival_us;
  size_t start;
  size_t count;
};

int main(int argc, char** argv) {
  if (argc != 9) {
    std::cerr << "neteq_reference input48.pcm arrivals.csv output48.pcm duration_ms "
                 "minimum_delay_ms maximum_delay_ms stretch_enabled timeline.csv\n";
    return 2;
  }
  std::ifstream pcm_file(argv[1], std::ios::binary);
  std::ifstream schedule(argv[2]);
  std::ofstream output(argv[3], std::ios::binary);
  if (!pcm_file || !schedule || !output) return 3;
  std::vector<uint8_t> pcm((std::istreambuf_iterator<char>(pcm_file)), {});
  std::vector<Chunk> chunks;
  std::string line;
  std::getline(schedule, line);
  while (std::getline(schedule, line)) {
    for (auto& ch : line)
      if (ch == ',') ch = ' ';
    std::istringstream row(line);
    Chunk chunk;
    if (!(row >> chunk.arrival_us >> chunk.start >> chunk.count)) return 4;
    // Schedule positions are mono 24 kHz source samples. The reference uses 48 kHz L16.
    chunk.start *= 2;
    chunk.count *= 2;
    if ((chunk.start + chunk.count) * 2 > pcm.size()) return 5;
    chunks.push_back(chunk);
  }
  webrtc::SimulatedClock clock(0);
  auto env = webrtc::CreateEnvironment(&clock);
  webrtc::NetEq::Config config;
  config.sample_rate_hz = 48000;
  config.min_delay_ms = std::stoi(argv[5]);
  config.max_delay_ms = std::stoi(argv[6]);
  config.for_test_no_time_stretching = std::stoi(argv[7]) == 0;
  std::ofstream timeline(argv[8]);
  if (!timeline) return 11;
  timeline << "time_us,queued_ms,target_ms,filtered_ms,concealed,concealment_events,stretched,"
              "accelerated,packets_discarded\n";
  auto neteq = webrtc::DefaultNetEqFactory().Create(
      env, config, webrtc::CreateAudioDecoderFactory<webrtc::AudioDecoderL16>());
  if (!neteq->RegisterPayloadType(93, webrtc::SdpAudioFormat("L16", 48000, 1))) return 6;
  size_t next = 0;
  uint16_t sequence = 0;
  int64_t now_us = 0;
  for (int64_t tick = 0; tick < std::stoll(argv[4]) * 1000; tick += 10000) {
    while (next < chunks.size() && chunks[next].arrival_us <= tick) {
      const auto& chunk = chunks[next++];
      if (chunk.arrival_us < now_us) return 7;
      clock.AdvanceTimeMicroseconds(chunk.arrival_us - now_us);
      now_us = chunk.arrival_us;
      // L16 packets of <=20 ms all retain the enclosing WebSocket chunk's arrival timestamp.
      for (size_t offset = 0; offset < chunk.count; offset += 960) {
        size_t count = std::min(size_t(960), chunk.count - offset);
        std::vector<uint8_t> payload(count * 2);
        for (size_t i = 0; i < count; ++i) {
          size_t at = (chunk.start + offset + i) * 2;
          payload[2 * i] = pcm[at + 1];
          payload[2 * i + 1] = pcm[at];
        }
        webrtc::RTPHeader header;
        header.payloadType = 93;
        header.sequenceNumber = sequence++;
        header.timestamp = static_cast<uint32_t>(chunk.start + offset);
        header.ssrc = 1;
        if (neteq->InsertPacket(header, payload,
                                webrtc::RtpPacketInfo(header, webrtc::Timestamp::Micros(now_us))) !=
            0)
          return 8;
      }
    }
    clock.AdvanceTimeMicroseconds(tick - now_us);
    now_us = tick;
    webrtc::AudioFrame frame;
    if (neteq->GetAudio(&frame) != 0 || frame.samples_per_channel_ != 480) return 9;
    auto state = neteq->GetOperationsAndState();
    auto cumulative = neteq->GetLifetimeStatistics();
    timeline << tick << ',' << state.current_buffer_size_ms << ',' << neteq->TargetDelayMs() << ','
             << neteq->FilteredCurrentDelayMs() << ',' << cumulative.concealed_samples << ','
             << cumulative.concealment_events << ',' << cumulative.inserted_samples_for_deceleration
             << ',' << cumulative.removed_samples_for_acceleration << ','
             << cumulative.packets_discarded << '\n';
    for (size_t i = 0; i < 480; ++i) {
      auto sample = static_cast<uint16_t>(frame.data()[i]);
      output.put(static_cast<char>(sample & 255));
      output.put(static_cast<char>(sample >> 8));
    }
  }
  auto stats = neteq->GetLifetimeStatistics();
  std::cout << "concealed_samples=" << stats.concealed_samples
            << " accelerate_samples=" << stats.removed_samples_for_acceleration
            << " expand_samples=" << stats.inserted_samples_for_deceleration << '\n';
  return output ? 0 : 10;
}
"""


RUST_SOURCE = r"""
use lk_agents_media::playout::{Config, Playout};
use std::{
    error::Error,
    fs,
    io::{BufWriter, Write},
    path::Path,
    time::Duration,
};
fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    let source = lk_agents_media::pcm_samples(&fs::read(&args[1])?);
    let schedule = fs::read_to_string(&args[2])?;
    let chunks: Vec<Vec<usize>> = schedule
        .lines()
        .skip(1)
        .map(|line| line.split(',').map(str::parse).collect())
        .collect::<Result<_, _>>()?;
    let out = Path::new(&args[3]);
    let mut pcm = BufWriter::new(fs::File::create(out.join("rust.pcm"))?);
    let mut baseline = BufWriter::new(fs::File::create(out.join("untreated.pcm"))?);
    let mut trace = BufWriter::new(fs::File::create(out.join("rust.csv"))?);
    writeln!(
        trace,
        "time_us,queued_source,target_us,concealed,underruns,inserted,removed"
    )?;
    let mut playout = Playout::continuous(Config {
        minimum_delay: Duration::from_millis(args[5].parse()?),
        startup_delay: Duration::from_millis(args[6].parse()?),
        maximum_delay: Duration::from_millis(args[7].parse()?),
        ..Config::default()
    })?;
    let mut next = 0;
    let mut available = 0;
    let mut read = 0;
    let mut output = [0; 240];
    let mut drained = false;
    for tick in (0..args[4].parse::<u64>()? * 1000).step_by(10_000) {
        while next < chunks.len() && chunks[next][0] as u64 <= tick {
            let chunk = &chunks[next];
            playout.push(
                Duration::from_micros(chunk[0] as u64),
                &source[chunk[1]..chunk[1] + chunk[2]],
            )?;
            available += chunk[2];
            next += 1;
        }
        if next == chunks.len() {
            playout.finish();
        }
        let rendered = playout.render(Duration::from_micros(tick), &mut output)?;
        let stats = playout.statistics();
        assert_eq!(
            stats.source_received,
            stats.source_consumed + rendered.queued_source as u64
        );
        assert_eq!(
            stats.rendered as i128,
            stats.source_consumed as i128 + stats.stretched as i128 - stats.accelerated as i128
                + stats.concealed as i128
                + stats.silent as i128
                - rendered.pending_output as i128
        );
        drained = rendered.drained;
        writeln!(
            trace,
            "{tick},{},{},{},{},{},{}",
            rendered.queued_source,
            rendered.target_delay.as_micros(),
            stats.concealed,
            stats.underruns,
            stats.stretched,
            stats.accelerated
        )?;
        for sample in output {
            pcm.write_all(&sample.to_le_bytes())?;
            let value = if read < available {
                let s = source[read];
                read += 1;
                s
            } else {
                0
            };
            baseline.write_all(&value.to_le_bytes())?;
        }
    }
    assert!(drained && next == chunks.len());
    assert_eq!(playout.statistics().source_consumed, source.len() as u64);
    Ok(())
}
"""


SAMPLE_RATE = 24_000
SAMPLES_PER_MS = SAMPLE_RATE // 1000
BYTES_PER_MS = SAMPLES_PER_MS * 2
REFERENCE_TAG = "webrtc-a545de4"

Scenario = Literal[
    "steady",
    "late-start",
    "bounded-jitter",
    "slow-jitter",
    "slow-clock",
    "fast-clock",
    "stall-400ms",
    "stall-1s",
    "repeated-stalls",
    "catch-up",
    "variable-chunks",
]
Track = Literal["untreated", "rust", "neteq"]

# Each preset isolates a delivery behavior. All preserve source order and every sample.
SCENARIOS: dict[Scenario, str] = {
    "steady": "Evenly paced chunks; clean control.",
    "late-start": "Second chunk is 60 ms late; later chunks return to nominal timing.",
    "bounded-jitter": "Seeded 0–30 ms delay variation for 20 s, with no clock drift.",
    "slow-jitter": "20 s of 1% slower delivery plus seeded 0–30 ms jitter.",
    "slow-clock": "Sender is 1000 ppm (0.1%) slower after the onset, with no jitter.",
    "fast-clock": "Sender is 1000 ppm (0.1%) faster after the onset, with no jitter.",
    "stall-400ms": "400 ms pause, followed by a burst that restores nominal timing.",
    "stall-1s": "1 s pause, followed by a burst that restores nominal timing.",
    "repeated-stalls": "Four 80 ms pauses, 600 ms apart; each delay persists (no catch-up).",
    "catch-up": "1.5 s of source delivered at 3x speed, creating a 1 s backlog.",
    "variable-chunks": "Cycle 20/100/40/200/60 ms chunks after onset, at nominal timing.",
}
JITTER_SEED = 734

TRACKS: tuple[Track, ...] = ("untreated", "rust", "neteq")


class Arrival(NamedTuple):
    arrival_us: int
    source_start: int
    samples: int


class TrackMetrics(TypedDict):
    pcm_sha256: str
    added_startup_ms: int | None
    pre_eof_counters: NotRequired[dict[str, str]]


class ScenarioMetrics(TypedDict):
    description: str
    fault_source_ms: int | None
    schedule_sha256: str
    tracks: dict[Track, TrackMetrics]


class Report(TypedDict):
    settings: dict[str, object]
    reference_api: str
    jitter_seed: int
    archive_sha256: str
    source_pcm_sha256: str
    scenarios: dict[Scenario, ScenarioMetrics]


class Options(argparse.Namespace):
    input: Path
    webrtc: Path
    crate: Path
    output: Path
    scenario: list[Scenario] | None
    chunk_ms: int
    fault_at_ms: int
    minimum_ms: int
    startup_ms: int
    maximum_ms: int
    native_maximum_ms: int


def run(*arguments: str | Path | int) -> str:
    completed = subprocess.run(
        [str(argument) for argument in arguments], capture_output=True, text=True
    )
    if completed.returncode:
        sys.stderr.write(completed.stderr)
        completed.check_returncode()

    return completed.stdout


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def onset_ms(pcm: bytes) -> int | None:
    for offset in range(0, len(pcm) - BYTES_PER_MS + 1, BYTES_PER_MS):
        samples = struct.unpack("<24h", pcm[offset : offset + BYTES_PER_MS])
        energy = sum(sample * sample for sample in samples) / SAMPLES_PER_MS
        if energy >= 150**2:
            return offset // BYTES_PER_MS

    return None


def make_schedule(name: Scenario, sample_count: int, chunk_ms: int, fault_ms: int) -> list[Arrival]:
    schedule: list[Arrival] = []
    previous_arrival_us = 0
    seed = JITTER_SEED
    start = 0
    variable_index = 0
    retained_delay_us = 0
    next_stall_us = 0

    while start < sample_count:
        nominal_us = start * 1_000_000 // SAMPLE_RATE
        elapsed_us = max(0, nominal_us - fault_ms * 1000)
        after_onset = nominal_us >= fault_ms * 1000
        delay_us = 0
        duration_ms = chunk_ms

        if name == "late-start" and start == chunk_ms * SAMPLES_PER_MS:
            delay_us = 60_000
        elif after_onset:
            if name in ("stall-400ms", "stall-1s"):
                stall_us = 400_000 if name == "stall-400ms" else 1_000_000
                delay_us = max(0, stall_us - elapsed_us)
            elif name in ("bounded-jitter", "slow-jitter"):
                # Reset the local PRNG for every run; no wall clock or process randomness.
                seed = (1664525 * seed + 1013904223) & 0xFFFFFFFF
                if name == "slow-jitter":
                    delay_us = min(elapsed_us, 20_000_000) // 100
                if elapsed_us < 20_000_000:
                    delay_us += seed % 30_001
            elif name in ("slow-clock", "fast-clock"):
                direction = 1 if name == "slow-clock" else -1
                delay_us = direction * (elapsed_us // 1000)
            elif name == "repeated-stalls":
                if elapsed_us >= next_stall_us and next_stall_us < 2_400_000:
                    retained_delay_us += 80_000
                    next_stall_us += 600_000
                delay_us = retained_delay_us
            elif name == "catch-up":
                progress_us = min(elapsed_us, 1_500_000)
                delay_us = -(progress_us - progress_us // 3)
            elif name == "variable-chunks":
                duration_ms = (20, 100, 40, 200, 60)[variable_index % 5]
                variable_index += 1

        # A reliable transport may batch chunks but cannot reorder or discard source audio.
        arrival_us = max(previous_arrival_us, nominal_us + delay_us)
        count = min(duration_ms * SAMPLES_PER_MS, sample_count - start)
        schedule.append(Arrival(arrival_us, start, count))
        previous_arrival_us = arrival_us
        start += count

    return schedule


def parse_arguments() -> Options:
    parser = argparse.ArgumentParser(
        description="Compare ordered PCM playout with native NetEq.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Scenarios (timing changes start at --fault-at-ms unless noted):\n"
        + "\n".join(f"  {name}: {description}" for name, description in SCENARIOS.items()),
    )
    parser.add_argument("--input", type=Path, required=True, help="mono 24 kHz PCM16 WAV")
    parser.add_argument(
        "--webrtc", type=Path, required=True, help=f"extracted {REFERENCE_TAG} archive"
    )
    parser.add_argument("--crate", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", action="append", choices=SCENARIOS)
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--fault-at-ms", type=int, default=2200)
    parser.add_argument("--minimum-ms", type=int, default=20)
    parser.add_argument("--startup-ms", type=int, default=80)
    parser.add_argument("--maximum-ms", type=int, default=100)
    parser.add_argument(
        "--native-maximum-ms", type=int, default=0, help="0: native unlimited target"
    )
    options = parser.parse_args(namespace=Options())

    if options.chunk_ms < 10 or options.chunk_ms % 10:
        parser.error("chunk duration must be a positive multiple of 10 ms")
    if options.fault_at_ms < 0:
        parser.error("fault time must be nonnegative")

    return options


def read_input(options: Options) -> bytes:
    with wave.open(str(options.input), "rb") as stream:
        audio_format = (stream.getframerate(), stream.getnchannels(), stream.getsampwidth())
        if audio_format != (SAMPLE_RATE, 1, 2):
            raise ValueError("input must be mono 24 kHz PCM16 WAV")

        pcm = stream.readframes(stream.getnframes())

    if len(pcm) < (options.fault_at_ms + 1500) * BYTES_PER_MS:
        raise ValueError("input must extend at least 1.5 seconds beyond the fault onset")
    if len(pcm) <= options.chunk_ms * BYTES_PER_MS:
        raise ValueError("input must contain at least two chunks")

    return pcm


def build_native(root: Path, temporary: Path) -> Path:
    source = temporary / "reference.cc"
    source.write_text(CPP_SOURCE)
    include = root / "include"
    library = root / "lib/libwebrtc.a"
    flags = [
        "-std=c++20",
        "-O2",
        "-DNDEBUG",
        "-DWEBRTC_POSIX",
        "-Wno-nullability-completeness",
        f"-I{include}",
        f"-I{include / 'third_party/abseil-cpp'}",
    ]
    links: list[str] = []

    if platform.system() == "Darwin":
        flags.append("-DWEBRTC_MAC")
        for framework in (
            "Foundation",
            "AudioToolbox",
            "CoreAudio",
            "CoreMedia",
            "CoreVideo",
            "VideoToolbox",
            "AVFoundation",
        ):
            links.extend(["-framework", framework])
    elif platform.system() == "Linux":
        flags.append("-DWEBRTC_LINUX")
        links.extend(["-lpthread", "-ldl"])
    else:
        raise ValueError("native compilation supports macOS and Linux")

    executable = temporary / "reference"
    run("clang++", *flags, source, library, *links, "-o", executable)
    return executable


def build_rust(crate: Path, temporary: Path) -> Path:
    (temporary / "src").mkdir()
    (temporary / "src/main.rs").write_text(RUST_SOURCE)
    manifest = temporary / "Cargo.toml"
    manifest.write_text(
        '[package]\nname = "playout-comparison"\nversion = "0.0.0"\nedition = "2024"\n\n'
        "[workspace]\n\n[dependencies]\nlk-agents-media = { path = "
        + json.dumps(str(crate.resolve()))
        + " }\n"
    )

    build_output = run("cargo", "build", "--offline", "--release", "--manifest-path", manifest, "--message-format=json")  # fmt: skip
    for line in build_output.splitlines():
        artifact = json.loads(line)
        if executable := artifact.get("executable"):
            return Path(executable)

    raise RuntimeError("Cargo did not report the comparison executable")


def fault_onset_ms(schedule: list[Arrival], name: Scenario, options: Options) -> int | None:
    if name == "steady":
        return None
    if name == "late-start":
        return options.chunk_ms

    anchor = options.fault_at_ms * SAMPLES_PER_MS
    first_fault = next(chunk for chunk in schedule if chunk.source_start >= anchor)
    return first_fault.source_start // SAMPLES_PER_MS


def check_active_speech(pcm: bytes, name: Scenario, fault_ms: int | None) -> None:
    if fault_ms is None:
        return

    window = pcm[fault_ms * BYTES_PER_MS : (fault_ms + 50) * BYTES_PER_MS]
    samples = struct.unpack(f"<{len(window) // 2}h", window)
    energy = sum(sample * sample for sample in samples) / len(samples) if samples else 0
    if math.sqrt(energy) < 150:
        raise ValueError(f"{name}: fault window is silent; select another onset or speech excerpt")


def collect_metrics(directory: Path, source: bytes, cutoff_us: int) -> dict[Track, TrackMetrics]:
    metrics: dict[Track, TrackMetrics] = {}
    source_onset = onset_ms(source)

    for track in TRACKS:
        raw = directory / f"{track}.pcm"
        pcm = raw.read_bytes()
        with wave.open(str(directory / f"{track}.wav"), "wb") as stream:
            stream.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
            stream.writeframes(pcm)

        output_onset = onset_ms(pcm)
        added_startup = None
        if output_onset is not None and source_onset is not None:
            added_startup = output_onset - source_onset

        metrics[track] = {"pcm_sha256": sha256(raw), "added_startup_ms": added_startup}

    # NetEq has no PCM finish signal. Exclude counters after source EOF from comparisons.
    for track in ("rust", "neteq"):
        with (directory / f"{track}.csv").open() as stream:
            rows = [row for row in csv.DictReader(stream) if int(row["time_us"]) < cutoff_us]
        metrics[track]["pre_eof_counters"] = rows[-1]

    return metrics


def render_scenario(
    name: Scenario,
    options: Options,
    source: bytes,
    raw: Path,
    source48: Path,
    rust: Path,
    native: Path,
) -> ScenarioMetrics:
    directory = options.output.resolve() / name
    directory.mkdir(exist_ok=True)
    schedule = make_schedule(name, len(source) // 2, options.chunk_ms, options.fault_at_ms)
    fault_ms = fault_onset_ms(schedule, name, options)
    check_active_speech(source, name, fault_ms)

    arrivals_file = directory / "arrivals.csv"
    with arrivals_file.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("arrival_us", "source_start", "samples"))
        writer.writerows(schedule)

    source_duration_ms = len(source) // BYTES_PER_MS
    last_arrival_us = schedule[-1].arrival_us
    end_ms = max(source_duration_ms, last_arrival_us / 1000) + options.maximum_ms + 2500
    render_duration_ms = math.ceil(end_ms / 10) * 10

    run(rust, raw, arrivals_file, directory, render_duration_ms, options.minimum_ms, options.startup_ms, options.maximum_ms)  # fmt: skip
    run(native, source48, arrivals_file, directory / "neteq48.pcm", render_duration_ms, options.minimum_ms, options.native_maximum_ms, 1, directory / "neteq.csv")  # fmt: skip
    run("ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", "48000", "-ac", "1", "-i", directory / "neteq48.pcm", "-ar", "24000", "-f", "s16le", directory / "neteq.pcm")  # fmt: skip

    cutoff_us = min(20_000_000, source_duration_ms * 1000, last_arrival_us)
    metrics = collect_metrics(directory, source, cutoff_us)
    print(f"{name}: {directory}", flush=True)
    return {
        "description": SCENARIOS[name],
        "fault_source_ms": fault_ms,
        "schedule_sha256": sha256(arrivals_file),
        "tracks": metrics,
    }


def main() -> None:
    options = parse_arguments()
    source = read_input(options)
    root = options.webrtc.resolve()
    library = root / "lib/libwebrtc.a"
    if not library.is_file() or not (root / "include/api/neteq/neteq.h").is_file():
        raise ValueError("webrtc distribution must contain headers and lib/libwebrtc.a")

    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "source.pcm"
    raw.write_bytes(source)

    report: Report = {
        "settings": {
            name: str(value) if isinstance(value, Path) else value
            for name, value in vars(options).items()
        },
        "reference_api": REFERENCE_TAG,
        "jitter_seed": JITTER_SEED,
        "archive_sha256": sha256(library),
        "source_pcm_sha256": sha256(raw),
        "scenarios": {},
    }

    with tempfile.TemporaryDirectory(prefix="neteq-compare-") as directory:
        temporary = Path(directory)
        native = build_native(root, temporary)
        rust = build_rust(options.crate, temporary)
        source48 = temporary / "source48.pcm"
        run("ffmpeg", "-v", "error", "-y", "-i", options.input.resolve(), "-ar", "48000", "-f", "s16le", source48)  # fmt: skip

        for name in options.scenario or SCENARIOS:
            report["scenarios"][name] = render_scenario(
                name, options, source, raw, source48, rust, native
            )

    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
