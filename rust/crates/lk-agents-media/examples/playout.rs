//! Offline fixture renderer. Arguments: input.pcm arrivals.csv output_directory duration_ms delay_ms.
//! PCM is little-endian mono 24 kHz. Schedule rows: arrival_us,source_start,samples.

use std::collections::VecDeque;
use std::error::Error;
use std::fs;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::time::Duration;

use lk_agents_media::playout::{Config, Playout};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 6 {
        return Err(
            "usage: playout input.pcm arrivals.csv output_directory duration_ms delay_ms".into()
        );
    }
    let bytes = fs::read(&args[1])?;
    if !bytes.len().is_multiple_of(2) {
        return Err("incomplete PCM sample".into());
    }
    let pcm = lk_agents_media::pcm_samples(&bytes);
    let schedule = fs::read_to_string(&args[2])?;
    let mut chunks = VecDeque::new();
    let mut previous = 0;
    let mut expected = 0;
    for row in schedule.lines().skip(1) {
        let fields = row.split(',').map(str::parse::<usize>).collect::<Result<Vec<_>, _>>()?;
        if fields.len() != 3
            || fields[0] < previous
            || fields[1] != expected
            || fields[2] == 0
            || fields[1] + fields[2] > pcm.len()
        {
            return Err("invalid ordered PCM schedule".into());
        }
        previous = fields[0];
        expected += fields[2];
        chunks.push_back((fields[0], fields[1], fields[2]));
    }
    if expected != pcm.len() {
        return Err("schedule does not cover the complete source".into());
    }
    let duration_ms: u64 = args[4].parse()?;
    let delay = Duration::from_millis(args[5].parse()?);
    let mut playout = Playout::new(Config {
        minimum_delay: delay,
        maximum_delay: delay.max(Duration::from_millis(100)),
        ..Config::default()
    })?;
    let out = Path::new(&args[3]);
    fs::create_dir_all(out)?;
    let mut candidate = BufWriter::new(fs::File::create(out.join("candidate.pcm"))?);
    let mut baseline = BufWriter::new(fs::File::create(out.join("baseline.pcm"))?);
    let mut timeline = BufWriter::new(fs::File::create(out.join("timeline.csv"))?);
    writeln!(
        timeline,
        "time_us,queued_source,pending_output,target_delay_us,source_consumed,concealed,silent,stretched,accelerated,underruns,fast_accelerated,merge_inserted"
    )?;
    let mut fifo_timeline = BufWriter::new(fs::File::create(out.join("fifo_timeline.csv"))?);
    writeln!(fifo_timeline, "time_us,source_consumed,missing_samples,underruns")?;
    let mut untreated = Fifo::new(0, pcm.len());
    let mut buffered = Fifo::new(100_000, pcm.len());
    let mut buffered_file = BufWriter::new(fs::File::create(out.join("buffered.pcm"))?);
    let mut block = [0; 240];
    let mut final_report = None;
    for at in (0..duration_ms * 1000).step_by(10_000) {
        while chunks.front().is_some_and(|chunk| chunk.0 as u64 <= at) {
            let (arrival, start, count) = chunks.pop_front().expect("front was present");
            let audio = &pcm[start..start + count];
            playout.push(Duration::from_micros(arrival as u64), audio)?;
            untreated.push(arrival as u64, audio);
            buffered.push(arrival as u64, audio);
        }
        if chunks.is_empty() {
            playout.finish();
        }
        let report = playout.render(Duration::from_micros(at), &mut block)?;
        let stats = playout.statistics();
        if stats.source_received != stats.source_consumed + report.queued_source as u64
            || stats.rendered as i128
                != stats.source_consumed as i128 + stats.stretched as i128
                    - stats.accelerated as i128
                    + stats.concealed as i128
                    + stats.silent as i128
                    - report.pending_output as i128
        {
            return Err("sample accounting diverged".into());
        }
        for (index, sample) in block.into_iter().enumerate() {
            let time = at + index as u64 * 1_000_000 / 24_000;
            candidate.write_all(&sample.to_le_bytes())?;
            baseline.write_all(&untreated.sample(time).to_le_bytes())?;
            buffered_file.write_all(&buffered.sample(time).to_le_bytes())?;
        }
        writeln!(
            fifo_timeline,
            "{at},{},{},{}",
            untreated.consumed, untreated.missing, untreated.underruns
        )?;
        writeln!(
            timeline,
            "{at},{},{},{},{},{},{},{},{},{},{},{}",
            report.queued_source,
            report.pending_output,
            report.target_delay.as_micros(),
            stats.source_consumed,
            stats.concealed,
            stats.silent,
            stats.stretched,
            stats.accelerated,
            stats.underruns,
            stats.fast_accelerated,
            stats.merge_inserted
        )?;
        final_report = Some(report);
    }
    if !final_report.is_some_and(|report| report.drained)
        || untreated.consumed != pcm.len()
        || buffered.consumed != pcm.len()
    {
        return Err("simulation ended before accepted audio drained".into());
    }
    fifo_timeline.flush()?;
    candidate.flush()?;
    baseline.flush()?;
    buffered_file.flush()?;
    fs::write(
        out.join("fifo_statistics.txt"),
        format!(
            "baseline: underruns={}, missing_samples={}\nbuffered_100ms: underruns={}, missing_samples={}\n",
            untreated.underruns, untreated.missing, buffered.underruns, buffered.missing
        ),
    )?;
    fs::write(
        out.join("fifo_statistics.csv"),
        format!(
            "track,startup_ms,underruns,missing_samples\nbaseline,0,{},{}\nbuffered,100,{},{}\n",
            untreated.underruns, untreated.missing, buffered.underruns, buffered.missing
        ),
    )?;
    timeline.flush()?;
    fs::write(out.join("statistics.txt"), format!("{:#?}\n", playout.statistics()))?;
    Ok(())
}

/// Offline FIFO reference: fixed startup delay, then real PCM or abrupt silence. No DSP.
struct Fifo {
    queue: VecDeque<i16>,
    delay_us: u64,
    first: Option<u64>,
    total: usize,
    consumed: usize,
    missing: usize,
    underruns: usize,
    gap: bool,
}

impl Fifo {
    fn new(delay_us: u64, total: usize) -> Self {
        Self {
            queue: VecDeque::new(),
            delay_us,
            first: None,
            total,
            consumed: 0,
            missing: 0,
            underruns: 0,
            gap: false,
        }
    }

    fn push(&mut self, arrival: u64, pcm: &[i16]) {
        self.first.get_or_insert(arrival);
        self.queue.extend(pcm.iter().copied());
    }

    fn sample(&mut self, now: u64) -> i16 {
        if !self.first.is_some_and(|first| now >= first + self.delay_us) {
            return 0;
        }
        if let Some(sample) = self.queue.pop_front() {
            self.consumed += 1;
            self.gap = false;
            return sample;
        }
        if self.consumed < self.total {
            self.missing += 1;
            if !self.gap {
                self.underruns += 1;
                self.gap = true;
            }
        }
        0
    }
}

#[cfg(test)]
mod tests {
    use super::Fifo;

    #[test]
    fn fixed_buffer_delays_without_modifying_samples_and_absorbs_a_short_stall() {
        let mut fifo = Fifo::new(100_000, 4800);
        fifo.push(0, &[1234; 2400]);
        for sample in 0..7200_u64 {
            let time = sample * 1_000_000 / 24_000;
            if sample == 2880 {
                fifo.push(time, &[5678; 2400]);
            }
            let expected = match sample {
                0..2400 => 0,
                2400..4800 => 1234,
                _ => 5678,
            };
            assert_eq!(fifo.sample(time), expected);
        }
        assert_eq!(fifo.underruns, 0);
        assert_eq!(fifo.missing, 0);
        assert_eq!(fifo.consumed, 4800);
    }
}
