//! Chunking byte input into fixed-size audio frames.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub data: Vec<u8>,
    pub samples_per_channel: u32,
}

/// Buffers bytes and emits whole frames: fixed-size, or progressive (20 ms first, doubling to
/// the target) to cut TTS time-to-first-audio.
#[crate::class(mutable, module = "livekit.agents.utils.audio")]
#[derive(Debug, Clone)]
pub struct AudioByteStream {
    sample_rate: u32,
    num_channels: u16,
    bytes_per_sample: usize,
    target_bytes_per_frame: usize,
    initial_bytes_per_frame: usize,
    current_bytes_per_frame: usize,
    buf: Vec<u8>,
}

const MIN_PROGRESSIVE_MS: u32 = 20;

// `push`, `write` and `flush` are hand-written below: they build `rtc.AudioFrame`s.
#[crate::host(object(
    pid = false,
    module = "livekit.agents.utils.audio",
    stub = "def push(self, data: bytes | memoryview) -> list[rtc.AudioFrame]: ...",
    stub = "def write(self, data: bytes | memoryview) -> list[rtc.AudioFrame]: ...",
    stub = "def flush(self) -> list[rtc.AudioFrame]: ...",
    stub_import = "from livekit import rtc"
))]
impl AudioByteStream {
    /// `samples_per_channel = None` means 100 ms, as in the reference.
    #[host(new(defaults(progressive = false)))]
    pub fn new(
        sample_rate: u32,
        num_channels: u16,
        samples_per_channel: Option<u32>,
        progressive: bool,
    ) -> Self {
        let samples_per_channel = samples_per_channel.unwrap_or(sample_rate / 10);
        let bytes_per_sample = num_channels as usize * size_of::<i16>();
        let target = samples_per_channel as usize * bytes_per_sample;

        let initial = if progressive {
            let min_samples = (sample_rate * MIN_PROGRESSIVE_MS / 1000) as usize;
            (min_samples * bytes_per_sample).min(target)
        } else {
            target
        };

        AudioByteStream {
            sample_rate,
            num_channels,
            bytes_per_sample,
            target_bytes_per_frame: target,
            initial_bytes_per_frame: initial,
            current_bytes_per_frame: initial,
            buf: Vec::new(),
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
    pub fn num_channels(&self) -> u16 {
        self.num_channels
    }

    /// Buffer `data` and emit every whole frame it completes.
    pub fn push(&mut self, data: &[u8]) -> Vec<Frame> {
        self.buf.extend_from_slice(data);

        let mut frames = Vec::new();
        // `> 0` is a termination guard: a zero frame size (reachable from Python) would loop
        // forever. Its `>= 0` mutant is untested on purpose: under it the test would hang.
        while self.buf.len() >= self.current_bytes_per_frame && self.current_bytes_per_frame > 0 {
            let rest = self.buf.split_off(self.current_bytes_per_frame);
            let data = std::mem::replace(&mut self.buf, rest);
            let samples_per_channel = (data.len() / self.bytes_per_sample) as u32;
            frames.push(Frame { data, samples_per_channel });

            // Double after emitting, so the first frame is the small one. `<` is equivalent to
            // `<=` here (`current` never exceeds `target`).
            if self.current_bytes_per_frame < self.target_bytes_per_frame {
                self.current_bytes_per_frame =
                    (self.current_bytes_per_frame * 2).min(self.target_bytes_per_frame);
            }
        }
        frames
    }

    /// Emit the buffered whole samples, short of a frame or not. A trailing partial sample stays
    /// for the next push: a flush can land mid-stream, inside a sample, and dropping its bytes
    /// would misalign every later sample. Progressive sizing is not reset.
    pub fn flush(&mut self) -> Vec<Frame> {
        let complete = self.buf.len() - self.buf.len() % self.bytes_per_sample;
        if complete == 0 {
            return Vec::new();
        }
        let rest = self.buf.split_off(complete);
        let data = std::mem::replace(&mut self.buf, rest);
        let samples_per_channel = (data.len() / self.bytes_per_sample) as u32;
        vec![Frame { data, samples_per_channel }]
    }

    /// Bytes waiting for the next frame.
    pub fn buffered_bytes(&self) -> usize {
        self.buf.len()
    }

    /// Seconds of audio waiting for the next frame.
    #[host(getter)]
    pub fn buffered_duration(&self) -> f64 {
        self.buf.len() as f64 / self.bytes_per_sample as f64 / f64::from(self.sample_rate)
    }

    /// Restart progressive sizing, keeping the buffered audio: after a mid-stream flush, the
    /// next burst starts with small frames without losing a partial sample.
    #[host(method)]
    pub fn reset_progressive(&mut self) {
        self.current_bytes_per_frame = self.initial_bytes_per_frame;
    }

    /// Drop everything buffered and restart progressive sizing (called on interruption, so the next
    /// word is not a full frame late).
    #[host(method)]
    pub fn clear(&mut self) {
        self.buf.clear();
        self.reset_progressive();
    }
}

// `push()` must return `list[rtc.AudioFrame]` (plugins depend on it).
#[cfg(feature = "python")]
#[pyo3::pymethods]
impl AudioByteStream {
    /// Accepts `bytes | bytearray | memoryview`, as the reference does.
    #[pyo3(name = "push")]
    fn host_push(
        &mut self,
        py: pyo3::Python<'_>,
        data: &[u8],
    ) -> pyo3::PyResult<pyo3::Py<pyo3::types::PyList>> {
        let frames = self.push(data);
        self.to_host_frames(py, frames)
    }

    /// Public alias of `push` in the reference.
    #[pyo3(name = "write")]
    fn host_write(
        &mut self,
        py: pyo3::Python<'_>,
        data: &[u8],
    ) -> pyo3::PyResult<pyo3::Py<pyo3::types::PyList>> {
        self.host_push(py, data)
    }

    #[pyo3(name = "flush")]
    fn host_flush(
        &mut self,
        py: pyo3::Python<'_>,
    ) -> pyo3::PyResult<pyo3::Py<pyo3::types::PyList>> {
        let frames = self.flush();
        self.to_host_frames(py, frames)
    }
}

#[cfg(feature = "python")]
impl AudioByteStream {
    fn to_host_frames(
        &self,
        py: pyo3::Python<'_>,
        frames: Vec<Frame>,
    ) -> pyo3::PyResult<pyo3::Py<pyo3::types::PyList>> {
        let frames = frames.into_iter().map(|f| {
            crate::AudioFrame::new(
                f.data,
                self.sample_rate,
                self.num_channels,
                f.samples_per_channel,
            )
        });
        Ok(pyo3::types::PyList::new(py, frames)?.unbind())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 16_000;
    fn bytes(n: usize) -> Vec<u8> {
        vec![0u8; n]
    }
    fn sizes(frames: &[Frame]) -> Vec<u32> {
        frames.iter().map(|f| f.samples_per_channel).collect()
    }

    // Catches: progressive mode not ramping (a 20 ms first frame).
    #[test]
    fn progressive_starts_at_20ms_and_doubles_to_the_target() {
        let mut s = AudioByteStream::new(SR, 1, Some(3200), true);
        // Plenty of data at once: the ramp must come from the sizing, not from the input shape.
        let frames = s.push(&bytes(3200 * 2 * 4));
        assert_eq!(
            sizes(&frames),
            vec![320, 640, 1280, 2560, 3200, 3200],
            "20/40/80/160/200/200 ms"
        );
    }

    // Catches: the ramp being applied in fixed mode.
    #[test]
    fn fixed_mode_never_ramps() {
        let mut s = AudioByteStream::new(SR, 1, Some(3200), false);
        assert_eq!(sizes(&s.push(&bytes(3200 * 2 * 3))), vec![3200, 3200, 3200]);
    }

    // Catches: `clear` leaving the frame size at the target.
    #[test]
    fn clear_restarts_the_progressive_ramp() {
        let mut s = AudioByteStream::new(SR, 1, Some(3200), true);
        s.push(&bytes(3200 * 2 * 4)); // ramp all the way up
        s.clear();
        assert_eq!(sizes(&s.push(&bytes(3200 * 2))), vec![320, 640, 1280], "back to 20 ms");
    }

    // Catches: bytes lost or duplicated across chunk boundaries.
    #[test]
    fn input_chunking_does_not_change_the_output_bytes() {
        let payload: Vec<u8> = (0..4000u32).map(|i| (i % 251) as u8).collect();
        let one_shot = {
            let mut s = AudioByteStream::new(SR, 1, Some(500), false);
            let mut out: Vec<u8> = s.push(&payload).into_iter().flat_map(|f| f.data).collect();
            out.extend(s.flush().into_iter().flat_map(|f| f.data));
            out
        };
        let dribbled = {
            let mut s = AudioByteStream::new(SR, 1, Some(500), false);
            let mut out = Vec::new();
            for c in payload.chunks(7) {
                out.extend(s.push(c).into_iter().flat_map(|f| f.data));
            }
            out.extend(s.flush().into_iter().flat_map(|f| f.data));
            out
        };
        assert_eq!(one_shot, payload);
        assert_eq!(dribbled, payload, "7-byte chunks must produce identical bytes");
    }

    // Catches: a torn final sample emitted, or dropped (misaligning every later sample).
    #[test]
    fn flush_keeps_a_partial_sample_for_the_next_push() {
        let mut s = AudioByteStream::new(SR, 2, Some(100), false); // 4 bytes per sample
        s.push(&[1, 2, 3, 4, 5, 6]); // 1 whole sample + 2 bytes
        assert_eq!(s.flush(), vec![Frame { data: vec![1, 2, 3, 4], samples_per_channel: 1 }]);
        assert_eq!(s.flush(), vec![], "a lone partial sample is not a frame");
        s.push(&[7, 8]);
        assert_eq!(s.flush(), vec![Frame { data: vec![5, 6, 7, 8], samples_per_channel: 1 }]);
    }

    // Catches: flush emitting an empty frame, which downstream treats as a real frame.
    #[test]
    fn flush_on_an_empty_buffer_emits_nothing() {
        let mut s = AudioByteStream::new(SR, 1, Some(100), false);
        assert_eq!(s.flush(), vec![]);
    }

    // Catches: `reset_progressive` dropping the buffer, or `buffered_duration` counting bytes.
    #[test]
    fn reset_progressive_keeps_the_buffer_and_restarts_the_ramp() {
        let mut s = AudioByteStream::new(SR, 1, Some(3200), true);
        s.push(&bytes(11_200 * 2)); // exactly 20/40/80/160/200/200 ms
        s.push(&bytes(160 * 2 + 1));
        assert_eq!(s.buffered_duration(), 160.5 / f64::from(SR));
        s.reset_progressive();
        assert_eq!(sizes(&s.push(&bytes(320 * 2))), vec![320], "back to 20 ms");
        assert_eq!(s.buffered_duration(), 160.5 / f64::from(SR), "the rest stays buffered");
    }

    // Catches: `flush` miscomputing the tail's sample count (the sink would over-read).
    #[test]
    fn flush_reports_the_sample_count_of_the_short_tail() {
        let mut s = AudioByteStream::new(SR, 1, Some(1000), false);
        s.push(&bytes(300 * 2)); // 300 samples: well short of a 1000-sample frame
        let f = s.flush();
        assert_eq!(sizes(&f), vec![300], "300 samples, not 600 and not 0");
        assert_eq!(f[0].data.len(), 600);

        // Stereo, where "bytes / 2" and "bytes / bytes_per_sample" finally disagree.
        let mut st = AudioByteStream::new(SR, 2, Some(1000), false);
        st.push(&bytes(300 * 4));
        assert_eq!(sizes(&st.flush()), vec![300], "per channel, not per sample slot");
    }

    // Catches: sample size computed per channel rather than per frame.
    #[test]
    fn stereo_counts_samples_per_channel_not_per_frame() {
        let mut s = AudioByteStream::new(SR, 2, Some(480), false);
        let frames = s.push(&bytes(480 * 2 * 2)); // 480 samples x 2ch x 2 bytes
        assert_eq!(sizes(&frames), vec![480]);
        assert_eq!(frames[0].data.len(), 1920);
    }
}
