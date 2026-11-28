"""Optional offline comparison with actual NetEq; no fixtures, downloads or Python packages.

Requires cargo, clang++, ffmpeg and an extracted livekit/rust-sdks webrtc-a545de4 archive.
Example: python3 neteq_compare.py --input speech.wav --webrtc /path/to/mac-arm64-release \
    --output /tmp/comparison
A copied standalone script also accepts --crate /path/to/lk-agents-media.
Both controllers receive the same complete ordered PCM chunks and arrival times. NetEq
alone uses offline 24->48->24 kHz conversion for L16; this is not a sample-exact comparison.
The native default maximum is unlimited; Rust defaults to 100 ms. Reports retain settings,
hashes, per-frame traces and added onset latency. No automatic quality ranking is made.
"""

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

CPP = r"""
// Offline reference only. Links the pinned libwebrtc distribution; no product dependency.
#include <algorithm>
#include <cstdint>
#include <iterator>
#include <fstream>
#include <iostream>
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

struct Chunk { int64_t arrival_us; size_t start; size_t count; };

int main(int argc, char** argv) {
  if (argc != 9) {
    std::cerr << "neteq_reference input48.pcm arrivals.csv output48.pcm duration_ms minimum_delay_ms maximum_delay_ms stretch_enabled timeline.csv\n";
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
    for (auto& ch : line) if (ch == ',') ch = ' ';
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
  timeline << "time_us,queued_ms,target_ms,filtered_ms,concealed,concealment_events,stretched,accelerated,packets_discarded\n";
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
            webrtc::RtpPacketInfo(header, webrtc::Timestamp::Micros(now_us))) != 0) return 8;
      }
    }
    clock.AdvanceTimeMicroseconds(tick - now_us);
    now_us = tick;
    webrtc::AudioFrame frame;
    if (neteq->GetAudio(&frame) != 0 || frame.samples_per_channel_ != 480) return 9;
    auto state = neteq->GetOperationsAndState();
    auto cumulative = neteq->GetLifetimeStatistics();
    timeline << tick << ',' << state.current_buffer_size_ms << ','
             << neteq->TargetDelayMs() << ',' << neteq->FilteredCurrentDelayMs() << ','
             << cumulative.concealed_samples << ',' << cumulative.concealment_events << ','
             << cumulative.inserted_samples_for_deceleration << ','
             << cumulative.removed_samples_for_acceleration << ','
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

RUST = r"""
use std::{error::Error, fs, io::{BufWriter, Write}, path::Path, time::Duration};
use lk_agents_media::playout::{Config, Playout};
fn main() -> Result<(), Box<dyn Error>> {
    let a: Vec<_> = std::env::args().collect();
    let source = lk_agents_media::pcm_samples(&fs::read(&a[1])?);
    let schedule = fs::read_to_string(&a[2])?;
    let chunks: Vec<Vec<usize>> = schedule.lines().skip(1).map(|line|
        line.split(',').map(str::parse).collect()).collect::<Result<_, _>>()?;
    let out = Path::new(&a[3]);
    let mut pcm = BufWriter::new(fs::File::create(out.join("rust.pcm"))?);
    let mut baseline = BufWriter::new(fs::File::create(out.join("untreated.pcm"))?);
    let mut trace = BufWriter::new(fs::File::create(out.join("rust.csv"))?);
    writeln!(trace, "time_us,queued_source,target_us,concealed,underruns,inserted,removed")?;
    let mut p = Playout::new(Config {
        minimum_delay: Duration::from_millis(a[5].parse()?),
        startup_delay: Duration::from_millis(a[6].parse()?),
        maximum_delay: Duration::from_millis(a[7].parse()?),
        ..Config::default()
    })?;
    let mut next = 0;
    let mut available = 0;
    let mut read = 0;
    let mut output = [0; 240];
    let mut drained = false;
    for tick in (0..a[4].parse::<u64>()? * 1000).step_by(10_000) {
        while next < chunks.len() && chunks[next][0] as u64 <= tick {
            let c = &chunks[next];
            p.push(Duration::from_micros(c[0] as u64), &source[c[1]..c[1]+c[2]])?;
            available += c[2];
            next += 1;
        }
        if next == chunks.len() { p.finish(); }
        let r = p.render(Duration::from_micros(tick), &mut output)?;
        let s = p.statistics();
        assert_eq!(s.source_received, s.source_consumed + r.queued_source as u64);
        assert_eq!(s.rendered as i128, s.source_consumed as i128 + s.stretched as i128
            - s.accelerated as i128 + s.concealed as i128 + s.silent as i128
            - r.pending_output as i128);
        drained = r.drained;
        writeln!(trace, "{tick},{},{},{},{},{},{}", r.queued_source,
            r.target_delay.as_micros(), s.concealed, s.underruns, s.stretched, s.accelerated)?;
        for sample in output {
            pcm.write_all(&sample.to_le_bytes())?;
            let value = if read < available { let s = source[read]; read += 1; s } else { 0 };
            baseline.write_all(&value.to_le_bytes())?;
        }
    }
    assert!(drained && next == chunks.len());
    assert_eq!(p.statistics().source_consumed, source.len() as u64);
    Ok(())
}
"""


def run(*args):
    completed = subprocess.run([str(a) for a in args], capture_output=True, text=True)
    if completed.returncode:
        sys.stderr.write(completed.stderr)
        completed.check_returncode()
    return completed.stdout


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def onset(pcm):
    for offset in range(0, len(pcm) - 47, 48):
        values = struct.unpack("<24h", pcm[offset : offset + 48])
        if sum(s * s for s in values) / 24 >= 150**2:
            return offset // 48
    return None


def arrivals(name, samples, chunk_ms, fault_ms):
    rows, previous, seed = [], 0, 734
    anchor = fault_ms * 24
    for start in range(0, samples, chunk_ms * 24):
        nominal = start * 1_000_000 // 24000
        delay = 0
        elapsed = max(0, nominal - fault_ms * 1000)
        if name == "startup" and start == chunk_ms * 24:
            delay = 60_000
        elif start >= anchor:
            if name in ("burst", "outage"):
                delay = max(0, (400_000 if name == "burst" else 1_000_000) - elapsed)
            elif name == "jitter":
                seed = (1664525 * seed + 1013904223) & 0xFFFFFFFF
                delay = min(elapsed, 20_000_000) // 100
                if elapsed < 20_000_000:
                    delay += seed % 30_001
            elif name == "catchup":
                progress = min(elapsed, 1_500_000)
                delay = -(progress - progress // 3)
        previous = max(previous, nominal + delay)
        rows.append((previous, start, min(chunk_ms * 24, samples - start)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="mono 24 kHz PCM16 WAV")
    parser.add_argument("--webrtc", type=Path, required=True, help="webrtc-a545de4 distribution")
    parser.add_argument("--crate", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=("clean", "startup", "jitter", "burst", "outage", "catchup"),
    )
    parser.add_argument("--chunk-ms", type=int, default=100)
    parser.add_argument("--fault-at-ms", type=int, default=2200)
    parser.add_argument("--minimum-ms", type=int, default=20)
    parser.add_argument("--startup-ms", type=int, default=80)
    parser.add_argument("--maximum-ms", type=int, default=100)
    parser.add_argument(
        "--native-maximum-ms", type=int, default=0, help="0: native unlimited target"
    )
    args = parser.parse_args()
    if args.chunk_ms < 10 or args.chunk_ms % 10 or args.fault_at_ms < 0:
        parser.error(
            "chunk duration must be a positive multiple of 10 ms; fault time must be nonnegative"
        )
    with wave.open(str(args.input), "rb") as stream:
        if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth()) != (24000, 1, 2):
            parser.error("input must be mono 24 kHz PCM16 WAV")
        pcm = stream.readframes(stream.getnframes())
    if len(pcm) < (args.fault_at_ms + 1500) * 48:
        parser.error("input must extend at least 1.5 seconds beyond the fault onset")
    if len(pcm) <= args.chunk_ms * 48:
        parser.error("input must contain at least two chunks")
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "source.pcm"
    raw.write_bytes(pcm)
    root = args.webrtc.resolve()
    library = root / "lib/libwebrtc.a"
    if not library.is_file() or not (root / "include/api/neteq/neteq.h").is_file():
        parser.error("webrtc distribution must contain headers and lib/libwebrtc.a")
    cases = args.scenario or ["clean", "startup", "jitter", "burst", "outage", "catchup"]
    report = {
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "reference_api": "webrtc-a545de4",
        "archive_sha256": digest(library),
        "source_pcm_sha256": digest(raw),
        "scenarios": {},
    }
    with tempfile.TemporaryDirectory(prefix="neteq-compare-") as directory:
        temporary = Path(directory)
        source = temporary / "reference.cc"
        source.write_text(CPP)
        include = root / "include"
        links = []
        flags = [
            "-std=c++20",
            "-O2",
            "-DNDEBUG",
            "-DWEBRTC_POSIX",
            "-Wno-nullability-completeness",
            f"-I{include}",
            f"-I{include / 'third_party/abseil-cpp'}",
        ]
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
            parser.error("native compilation supports macOS and Linux")
        native = temporary / "reference"
        run("clang++", *flags, source, library, *links, "-o", native)
        (temporary / "src").mkdir()
        (temporary / "src/main.rs").write_text(RUST)
        (temporary / "Cargo.toml").write_text(
            '[package]\nname="playout-comparison"\nversion="0.0.0"\nedition="2024"\n'
            "[workspace]\n[dependencies]\nlk-agents-media={path="
            + json.dumps(str(args.crate.resolve()))
            + "}\n"
        )
        built = run(
            "cargo",
            "build",
            "--offline",
            "--release",
            "--manifest-path",
            temporary / "Cargo.toml",
            "--message-format=json",
        )
        executable = next(
            event["executable"]
            for line in built.splitlines()
            if (event := json.loads(line)).get("executable")
        )
        source48 = temporary / "source48.pcm"
        run(
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            args.input.resolve(),
            "-ar",
            "48000",
            "-f",
            "s16le",
            source48,
        )
        for name in cases:
            case = out / name
            case.mkdir(exist_ok=True)
            schedule = arrivals(name, len(pcm) // 2, args.chunk_ms, args.fault_at_ms)
            # Fail silent tests rather than claiming a fault was audibly exercised.
            fault = (
                args.chunk_ms
                if name == "startup"
                else schedule[
                    next(i for i, r in enumerate(schedule) if r[1] >= args.fault_at_ms * 24)
                ][1]
                // 24
            )
            if name != "clean":
                window = pcm[fault * 48 : (fault + 50) * 48]
                values = struct.unpack(f"<{len(window) // 2}h", window)
                if not values or math.sqrt(sum(v * v for v in values) / len(values)) < 150:
                    raise ValueError(
                        f"{name}: fault window is silent; select another onset or speech excerpt"
                    )
            path = case / "arrivals.csv"
            with path.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("arrival_us", "source_start", "samples"))
                writer.writerows(schedule)
            duration = (
                math.ceil(
                    (max(len(pcm) // 48, schedule[-1][0] / 1000) + args.maximum_ms + 2500) / 10
                )
                * 10
            )
            run(
                executable,
                raw,
                path,
                case,
                duration,
                args.minimum_ms,
                args.startup_ms,
                args.maximum_ms,
            )
            run(
                native,
                source48,
                path,
                case / "neteq48.pcm",
                duration,
                args.minimum_ms,
                args.native_maximum_ms,
                1,
                case / "neteq.csv",
            )
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
            metrics = {}
            for track in ("untreated", "rust", "neteq"):
                data = (case / f"{track}.pcm").read_bytes()
                with wave.open(str(case / f"{track}.wav"), "wb") as stream:
                    stream.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
                    stream.writeframes(data)
                active, original = onset(data), onset(pcm)
                metrics[track] = {
                    "pcm_sha256": digest(case / f"{track}.pcm"),
                    "added_startup_ms": active - original
                    if active is not None and original is not None
                    else None,
                }
            # Compare operation counters before source EOF; native intentionally has no EOF signal.
            cutoff = min(20_000_000, len(pcm) // 48 * 1000, schedule[-1][0])
            for track in ("rust", "neteq"):
                with (case / f"{track}.csv").open() as stream:
                    rows = [row for row in csv.DictReader(stream) if int(row["time_us"]) < cutoff]
                metrics[track]["pre_eof_counters"] = rows[-1]
            report["scenarios"][name] = {"fault_source_ms": fault, "tracks": metrics}
            print(f"{name}: {case}", flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
