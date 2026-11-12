//! The silences in a voice's audio, found as it arrives: what anchors its transcript's timing.
//! Silence is a 10 ms window quieter than a fraction of the reply's own loud level (its 90th
//! percentile so far), so it holds for any voice, gain or codec. A pause is a silence of at least
//! 150 ms; one before the first speech and one after the last are pauses too, so a reader learns
//! where the speech starts and ends from the same report.

/// A silence in the audio, in seconds from its start.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pause {
    pub start: f64,
    pub end: f64,
}

/// Silence is quieter than this share of the loud level.
const RELATIVE: f64 = 0.06;
/// And always below this RMS, whatever the level: a floor for near-silent replies.
const FLOOR: f64 = 30.0;
/// The shortest silence reported between two words, in 10 ms windows.
const MIN_WINDOWS: u64 = 15;
/// Log-RMS histogram bins: `4 * log2(rms + 1)` spans 0 to 60 for 16-bit audio.
const BINS: usize = 64;

pub struct PauseDetector {
    /// Samples per channel in one window: 10 ms, to the sample.
    window: usize,
    /// One window's length in seconds, exactly: 220 samples at 22.05 kHz is not 10 ms.
    window_secs: f64,
    channels: usize,
    /// The channel the next sample is for: only the first is measured.
    channel: usize,
    sum: f64,
    count: usize,
    /// Windows seen.
    windows: u64,
    levels: [u32; BINS],
    seen: u32,
    /// The window the current silence began at; `None` while speech plays.
    silent_since: Option<u64>,
    speech: bool,
}

impl PauseDetector {
    pub fn new(sample_rate: u32, channels: u16) -> Self {
        let window = (sample_rate as usize / 100).max(1);
        PauseDetector {
            window,
            window_secs: window as f64 / f64::from(sample_rate.max(1)),
            channels: usize::from(channels.max(1)),
            channel: 0,
            sum: 0.0,
            count: 0,
            windows: 0,
            levels: [0; BINS],
            seen: 0,
            silent_since: Some(0),
            speech: false,
        }
    }

    /// Interleaved PCM: the pauses that ended in it, speech having resumed.
    pub fn push(&mut self, pcm: impl IntoIterator<Item = i16>) -> Vec<Pause> {
        let mut out = Vec::new();
        for sample in pcm {
            let first = self.channel == 0;
            self.channel = (self.channel + 1) % self.channels;
            if !first {
                continue;
            }
            let s = f64::from(sample);
            self.sum += s * s;
            self.count += 1;
            if self.count == self.window {
                let rms = (self.sum / self.count as f64).sqrt();
                (self.sum, self.count) = (0.0, 0);
                out.extend(self.level(rms));
            }
        }
        out
    }

    /// One window's RMS: the pause it ends, if any.
    fn level(&mut self, rms: f64) -> Option<Pause> {
        let at = self.windows;
        self.windows += 1;
        self.levels[bin(rms)] += 1;
        self.seen += 1;
        let silent = rms < self.threshold();
        match (silent, self.silent_since) {
            (true, None) => {
                self.silent_since = Some(at);
                None
            }
            (false, Some(since)) => {
                self.silent_since = None;
                let first = !std::mem::replace(&mut self.speech, true);
                // leading silence of any length; between words, only a real pause
                ((first && at > since) || at - since >= MIN_WINDOWS).then(|| self.pause(since, at))
            }
            _ => None,
        }
    }

    /// The audio ended: the silence after the last speech, if any.
    pub fn finish(&mut self) -> Option<Pause> {
        let since = self.silent_since.take().filter(|_| self.speech)?;
        (self.windows > since).then(|| self.pause(since, self.windows))
    }

    fn pause(&self, from: u64, to: u64) -> Pause {
        Pause { start: from as f64 * self.window_secs, end: to as f64 * self.window_secs }
    }

    /// The loud level (90th percentile of the windows so far), scaled to the silence threshold.
    fn threshold(&self) -> f64 {
        let target = (f64::from(self.seen) * 0.9).ceil() as u32;
        let mut acc = 0;
        let mut level = 0.0;
        for (i, n) in self.levels.iter().enumerate() {
            acc += n;
            if acc >= target {
                level = 2f64.powf(i as f64 / 4.0) - 1.0;
                break;
            }
        }
        (level * RELATIVE).max(FLOOR)
    }
}

fn bin(rms: f64) -> usize {
    ((4.0 * (rms + 1.0).log2()) as usize).min(BINS - 1)
}

#[cfg(test)]
#[path = "pauses_tests.rs"]
mod tests;
