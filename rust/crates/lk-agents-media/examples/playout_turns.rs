//! Replay three seconds of generated speech across ten answers with long idle gaps. Compare
//! retaining timing with full reset, and show clean-answer adaptation. Optional PCM captures
//! of answers 1/2 isolate the histogram's effect using identical source and delivery schedules.
use std::error::Error;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Duration;

use lk_agents_media::playout::{Config, Playout};

const ANSWER_SAMPLES: usize = 72_000;

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if !(3..=4).contains(&args.len()) {
        return Err("usage: playout_turns source.pcm turns.csv [capture_directory]".into());
    }
    let bytes = fs::read(&args[1])?;
    let source = lk_agents_media::pcm_samples(&bytes);
    if source.len() < ANSWER_SAMPLES {
        return Err("need at least three seconds of speech".into());
    }
    let capture = args.get(3).map(Path::new);
    if let Some(directory) = capture {
        fs::create_dir_all(directory)?;
    }
    let mut csv = fs::File::create(&args[2])?;
    writeln!(
        csv,
        "variant,turn,startup_delay_ms,gate_release_ms,underruns,concealed,stretched,passive_stretched,observations_before,quantile_before_ms,observations_after,quantile_after_ms"
    )?;
    let mut out = [0; 240];
    for variant in ["retained", "reset", "clean"] {
        // A deliberately low prior exposes the same first-answer issue. The clean control
        // starts at the production default and shows how its initial waiting time relaxes.
        let mut p = Playout::new(Config {
            startup_delay: Duration::from_millis(if variant == "clean" { 80 } else { 20 }),
            ..Config::default()
        })?;
        for turn in 0..10_u64 {
            if turn > 0 {
                if variant == "reset" {
                    p.reset();
                } else {
                    p.begin_utterance();
                }
            }
            let mut audio = if turn < 2 {
                capture
                    .map(|directory| {
                        fs::File::create(directory.join(format!("{variant}-{}.pcm", turn + 1)))
                    })
                    .transpose()?
                    .map(BufWriter::new)
            } else {
                None
            };
            let base = turn * 60_000;
            let mut packet = 0;
            let mut start = None;
            let mut startup = 0;
            let mut before = (0, None);
            let mut after = (0, None);
            for tick in (0..3500_u64).step_by(10) {
                while packet < 30 {
                    let arrival =
                        packet * 100 + if packet == 1 && variant != "clean" { 60 } else { 0 };
                    if arrival > tick {
                        break;
                    }
                    let offset = packet as usize * 2400;
                    p.push(Duration::from_millis(base + arrival), &source[offset..offset + 2400])?;
                    packet += 1;
                    if packet == 30 {
                        p.finish();
                    }
                }
                let report = p.render(Duration::from_millis(base + tick), &mut out)?;
                if let Some(audio) = &mut audio {
                    for sample in out {
                        audio.write_all(&sample.to_le_bytes())?;
                    }
                }
                startup = report.startup_delay.as_millis();
                if tick == 0 {
                    before = (report.startup_observations, report.startup_quantile);
                }
                after = (report.startup_observations, report.startup_quantile);
                if report.started && start.is_none() {
                    start = Some(tick);
                }
            }
            if let Some(audio) = &mut audio {
                audio.flush()?;
            }
            let s = p.statistics();
            if s.source_consumed != ANSWER_SAMPLES as u64 {
                return Err("answer did not drain".into());
            }
            writeln!(
                csv,
                "{variant},{},{startup},{},{},{},{},{},{},{},{},{}",
                turn + 1,
                start.ok_or("playback never started")?,
                s.underruns,
                s.concealed,
                s.stretched,
                s.passive_stretched,
                before.0,
                before.1.map(|d| d.as_millis().to_string()).unwrap_or_default(),
                after.0,
                after.1.map(|d| d.as_millis().to_string()).unwrap_or_default()
            )?;
        }
    }
    Ok(())
}
