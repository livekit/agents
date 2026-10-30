//! Direct audio data-plane step measurements, without production observation hooks: core
//! instances, not full sessions.

use lk_agents_abi::{AudioFrame, Ts, resources::ResourceId};
use lk_agents_voice::audio::AudioChannel;
use lk_agents_voice::data_plane::{AudioForwarder, ForwarderCommand};
use serde::Serialize;
use std::hint::black_box;
use std::time::Instant;

#[cfg(any(target_os = "macos", target_os = "linux"))]
mod cpu_clock {
    // POSIX time_t and long are 64 bits on the supported 64-bit Unix targets.
    #[repr(C)]
    struct Timespec {
        seconds: i64,
        nanos: i64,
    }
    unsafe extern "C" {
        fn clock_gettime(clock: i32, output: *mut Timespec) -> i32;
    }
    #[cfg(target_os = "macos")]
    const THREAD_CPU: i32 = 16;
    #[cfg(target_os = "linux")]
    const THREAD_CPU: i32 = 3;

    pub fn now() -> u64 {
        let mut time = Timespec { seconds: 0, nanos: 0 };
        // SAFETY: clock_gettime writes exactly one aligned, initialized POSIX timespec.
        // No pointer is retained; this adapter is measurement-only, outside core transitions.
        let status = unsafe { clock_gettime(THREAD_CPU, &mut time) };
        assert_eq!(status, 0, "thread CPU clock unavailable");
        time.seconds as u64 * 1_000_000_000 + time.nanos as u64
    }
}

#[derive(Serialize)]
struct Histogram {
    count: usize,
    p99_us: f64,
    p999_us: f64,
    max_us: f64,
    // Upper bounds in microseconds, exclusive buckets plus overflow.
    bucket_upper_us: [u64; 8],
    buckets: [usize; 9],
}

fn histogram(mut values: Vec<u64>) -> Histogram {
    assert!(!values.is_empty(), "empty step measurement");
    values.sort_unstable();
    let bounds = [1, 10, 100, 500, 1000, 2000, 5000, 10_000];
    let mut buckets = [0; 9];
    for value in &values {
        buckets[bounds.iter().position(|bound| *value <= bound * 1000).unwrap_or(8)] += 1;
    }
    Histogram {
        count: values.len(),
        p99_us: values[(values.len() * 99).div_ceil(100) - 1] as f64 / 1000.,
        p999_us: values[(values.len() * 999).div_ceil(1000) - 1] as f64 / 1000.,
        max_us: *values.last().unwrap() as f64 / 1000.,
        bucket_upper_us: bounds,
        buckets,
    }
}

#[derive(Serialize)]
struct Observation {
    subject: &'static str,
    sessions: usize,
    frame_ms: u64,
    input_rate_hz: u32,
    output_rate_hz: u32,
    step_wall: Histogram,
    step_thread_cpu: Histogram,
    steps_per_second: f64,
    measured_budget_pass: bool,
}

#[cfg(any(target_os = "macos", target_os = "linux"))]
fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    assert!(args.is_empty() || args == ["--check"], "usage: bench_steps [--check]");
    let mut observations = Vec::new();
    for sessions in [1, 8, 32] {
        for frame_ms in [10, 20] {
            let mut cores: Vec<_> = (0..sessions)
                .map(|_| {
                    let mut core = AudioForwarder::new(16000);
                    for (index, channel) in
                        [AudioChannel::Stt, AudioChannel::Vad, AudioChannel::Eot].iter().enumerate()
                    {
                        black_box(core.handle(ForwarderCommand::Install {
                            channel: *channel,
                            sink: ResourceId(index as u64 + 1),
                        }));
                    }
                    core
                })
                .collect();
            let frame = AudioFrame::new(
                vec![1u8; (48 * frame_ms * 2) as usize],
                48000,
                1,
                (48 * frame_ms) as u32,
            );
            // Include warm resampler conversion; separately reported session startup covers cold work.
            for core in &mut cores {
                black_box(core.on_frame(frame.clone(), Ts::ZERO));
            }
            let mut wall = Vec::with_capacity(sessions * 1000);
            let mut cpu = Vec::with_capacity(sessions * 1000);
            let total = Instant::now();
            for tick in 0..1000 {
                for core in &mut cores {
                    let input = frame.clone();
                    let at = Ts::from_nanos(tick * frame_ms * 1_000_000);
                    let started = Instant::now();
                    let cpu_start = cpu_clock::now();
                    let effects = black_box(core.on_frame(input, at));
                    cpu.push(cpu_clock::now() - cpu_start);
                    wall.push(started.elapsed().as_nanos() as u64);
                    black_box(effects);
                }
            }
            let step_thread_cpu = histogram(cpu);
            let measured_budget_pass = step_thread_cpu.p999_us <= frame_ms as f64 * 50.
                && step_thread_cpu.max_us <= frame_ms as f64 * 100.;
            observations.push(Observation {
                subject: "AudioForwarder::on_frame",
                sessions,
                frame_ms,
                input_rate_hz: 48000,
                output_rate_hz: 16000,
                steps_per_second: (sessions * 1000) as f64 / total.elapsed().as_secs_f64(),
                step_wall: histogram(wall),
                step_thread_cpu,
                measured_budget_pass,
            });
        }
    }
    println!("{}", serde_json::to_string_pretty(&observations).unwrap());
    if !args.is_empty() && observations.iter().any(|row| !row.measured_budget_pass) {
        std::process::exit(1);
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn main() {
    panic!("thread CPU measurement currently supports 64-bit macOS/Linux only");
}
