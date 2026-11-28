//! Replay the same generated speech across ten answers with long idle gaps. CSV compares
//! retained timing with a full reset. A 20ms cold prior deliberately exposes the first late
//! second chunk; production defaults use 80ms. No sleeps or wall-clock reads are involved.
use std::error::Error;
use std::fs;
use std::io::Write;
use std::time::Duration;

use lk_agents_media::playout::{Config, Playout};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("usage: playout_turns source.pcm turns.csv".into());
    }
    let bytes = fs::read(&args[1])?;
    let source = lk_agents_media::pcm_samples(&bytes);
    if source.len() < 24_000 {
        return Err("need at least one second of source speech".into());
    }
    let mut csv = fs::File::create(&args[2])?;
    writeln!(
        csv,
        "variant,turn,startup_delay_ms,gate_release_ms,underruns,concealed,stretched,passive_stretched"
    )?;
    let mut out = [0; 240];
    for variant in ["retained", "reset"] {
        let mut p =
            Playout::new(Config { startup_delay: Duration::from_millis(20), ..Config::default() })?;
        for turn in 0..10_u64 {
            if turn > 0 {
                if variant == "retained" {
                    p.begin_utterance();
                } else {
                    p.reset();
                }
            }
            let base = turn * 60_000;
            let mut packet = 0;
            let mut start = None;
            let mut startup = 0;
            for tick in (0..1400_u64).step_by(10) {
                while packet < 10 {
                    let arrival = if packet == 0 { 0 } else { packet * 100 + 60 };
                    if arrival > tick {
                        break;
                    }
                    let offset = packet as usize * 2400;
                    p.push(Duration::from_millis(base + arrival), &source[offset..offset + 2400])?;
                    packet += 1;
                    if packet == 10 {
                        p.finish();
                    }
                }
                let report = p.render(Duration::from_millis(base + tick), &mut out)?;
                startup = report.startup_delay.as_millis();
                if report.started && start.is_none() {
                    start = Some(tick);
                }
            }
            let s = p.statistics();
            if s.source_consumed != 24_000 {
                return Err("answer did not drain".into());
            }
            writeln!(
                csv,
                "{variant},{},{startup},{},{},{},{},{}",
                turn + 1,
                start.ok_or("playback never started")?,
                s.underruns,
                s.concealed,
                s.stretched,
                s.passive_stretched
            )?;
        }
    }
    Ok(())
}
