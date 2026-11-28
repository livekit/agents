//! Local throughput/latency measurement; no machine-specific timing assertions in regression tests.
use std::error::Error;
use std::hint::black_box;
use std::io::Write;
use std::time::{Duration, Instant};

use lk_agents_media::playout::{Config, Playout};

#[allow(
    clippy::disallowed_methods,
    reason = "the standalone benchmark measures wall-clock execution cost"
)]
fn main() -> Result<(), Box<dyn Error>> {
    let voice: Vec<_> = (0..2400)
        .map(|i| ((i as f64 * std::f64::consts::TAU / 120.0).sin() * 12_000.0) as i16)
        .collect();
    let mut out = [0; 240];
    let mut stdout = std::io::stdout().lock();
    for mode in ["normal", "stalls", "backlog"] {
        let mut p = Playout::new(Config::default())?;
        let mut timings = Vec::with_capacity(100_000);
        let started = Instant::now();
        for tick in 0..100_000_u64 {
            let now = Duration::from_millis(tick * 10);
            let measure = Instant::now();
            if tick.is_multiple_of(10) && (mode != "stalls" || tick % 100 < 70) {
                p.push(now, &voice)?;
            }
            if mode == "backlog" && tick == 0 {
                for _ in 0..5 {
                    p.push(now, &voice)?;
                }
            }
            p.render(now, black_box(&mut out))?;
            timings.push(measure.elapsed().as_nanos());
        }
        let elapsed = started.elapsed();
        timings.sort_unstable();
        writeln!(
            stdout,
            "{mode}: 1000s audio in {:.3}s; p50={}ns p99={}ns max={}ns; {:?}",
            elapsed.as_secs_f64(),
            timings[50_000],
            timings[99_000],
            timings[99_999],
            p.statistics()
        )?;
    }
    Ok(())
}
