//! `rtc.AudioMixer` (`livekit/rtc/audio_mixer.py`), sans-IO: live N-stream summing, not the
//! recorder's timeline mixer. The host owns the async pull, per-stream timeout and output queue.
//! Reference rules: a short buffer is zero-padded and then discarded; `blocksize = 0` means
//! `sample_rate / 10`; a round where no stream had data emits nothing; a stream is removed only
//! when exhausted and empty. Sums stay in `f32` and clip.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// The reference keys on the iterator object, a host object; the host keeps the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StreamId(pub u64);

/// The reference's one `logger.warning`, counted so a silently non-playing sound is observable.
#[derive(Debug, Default)]
pub struct MixerCounters {
    /// `AudioMixer: stream {stream} timeout, ignoring`.
    pub stream_timeouts: AtomicU64,
    /// Routine at a sound's end; sustained means sources cannot keep up with the block rate.
    pub padded_contributions: AtomicU64,
    pub empty_rounds: AtomicU64,
}

impl MixerCounters {
    fn bump(counter: &AtomicU64) {
        counter.fetch_add(1, Ordering::Relaxed);
    }

    pub fn stream_timeouts(&self) -> u64 {
        self.stream_timeouts.load(Ordering::Relaxed)
    }

    pub fn padded_contributions(&self) -> u64 {
        self.padded_contributions.load(Ordering::Relaxed)
    }

    pub fn empty_rounds(&self) -> u64 {
        self.empty_rounds.load(Ordering::Relaxed)
    }
}

#[derive(Debug, Default)]
struct Stream {
    /// Always a whole number of frames.
    buffer: Vec<i16>,
    /// The reference's `had_data`: non-empty on entry to this round, or pushed to during it.
    had_data: bool,
    /// `StopAsyncIteration` seen; the stream still contributes what it holds.
    exhausted: bool,
}

#[derive(Debug)]
pub struct Mixer {
    sample_rate: u32,
    num_channels: u16,
    /// **Frames**, not samples.
    blocksize: usize,
    streams: BTreeMap<StreamId, Stream>,
    removals: Vec<StreamId>,
    next_id: u64,
    pub counters: MixerCounters,
}

impl Mixer {
    /// `blocksize` is in frames; zero means `sample_rate / 10`.
    pub fn new(sample_rate: u32, num_channels: u16, blocksize: usize) -> Self {
        let blocksize = if blocksize > 0 { blocksize } else { (sample_rate / 10) as usize };
        Mixer {
            sample_rate,
            num_channels,
            blocksize,
            streams: BTreeMap::new(),
            removals: Vec::new(),
            next_id: 0,
            counters: MixerCounters::default(),
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn num_channels(&self) -> u16 {
        self.num_channels
    }

    pub fn blocksize(&self) -> usize {
        self.blocksize
    }

    pub fn add_stream(&mut self) -> StreamId {
        let id = StreamId(self.next_id);
        self.next_id += 1;
        self.streams.insert(id, Stream::default());
        id
    }

    pub fn remove_stream(&mut self, id: StreamId) {
        self.streams.remove(&id);
    }

    pub fn stream_ids(&self) -> Vec<StreamId> {
        self.streams.keys().copied().collect()
    }

    pub fn is_empty(&self) -> bool {
        self.streams.is_empty()
    }

    /// `while buf.shape[0] < self._chunk_size and not exhausted`; an unknown id answers `false`.
    pub fn wants_more(&self, id: StreamId) -> bool {
        self.streams.get(&id).is_some_and(|s| {
            !s.exhausted && s.buffer.len() < self.blocksize * usize::from(self.num_channels)
        })
    }

    /// A push to an unknown id is dropped: the stream was removed mid-pull.
    pub fn push(&mut self, id: StreamId, pcm: &[i16]) {
        if let Some(s) = self.streams.get_mut(&id) {
            s.buffer.extend_from_slice(pcm);
            s.had_data = true;
        }
    }

    /// Not removed here: it contributes what it holds, padded, for one more round.
    pub fn mark_exhausted(&mut self, id: StreamId) {
        if let Some(s) = self.streams.get_mut(&id) {
            s.exhausted = true;
        }
    }

    pub fn note_timeout(&mut self, _id: StreamId) {
        MixerCounters::bump(&self.counters.stream_timeouts);
    }

    /// `None` when no stream had data: the caller must not emit a frame.
    pub fn mix(&mut self) -> Option<Vec<i16>> {
        if self.streams.is_empty() {
            MixerCounters::bump(&self.counters.empty_rounds);
            return None;
        }

        let width = self.blocksize * usize::from(self.num_channels);
        let mut acc = vec![0.0f32; width];
        let mut any_data = false;
        let mut padded = false;
        self.removals.clear();

        for (id, s) in &mut self.streams {
            if s.had_data {
                any_data = true;
            }
            if s.buffer.len() >= width {
                for (a, v) in acc.iter_mut().zip(s.buffer.drain(..width)) {
                    *a += f32::from(v);
                }
            } else {
                // Padded and discarded; holding them back desynchronises this stream.
                if !s.buffer.is_empty() {
                    padded = true;
                }
                for (a, v) in acc.iter_mut().zip(s.buffer.iter().copied()) {
                    *a += f32::from(v);
                }
                s.buffer.clear();
            }
            if s.exhausted && s.buffer.is_empty() {
                self.removals.push(*id);
            }
            s.had_data = !s.buffer.is_empty();
        }

        for id in std::mem::take(&mut self.removals) {
            self.streams.remove(&id);
            self.removals.push(id);
        }

        if padded {
            MixerCounters::bump(&self.counters.padded_contributions);
        }
        if !any_data {
            MixerCounters::bump(&self.counters.empty_rounds);
            return None;
        }

        // The clamp pins that the sum stays in `f32`: an integer accumulator would wrap into noise.
        Some(acc.into_iter().map(|v| v.clamp(-32768.0, 32767.0) as i16).collect())
    }

    /// Cleared by the next [`mix`](Self::mix).
    pub fn drain_removals(&mut self) -> Vec<StreamId> {
        std::mem::take(&mut self.removals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mixer(blocksize: usize) -> Mixer {
        Mixer::new(48_000, 1, blocksize)
    }

    // Catches: `blocksize=0` being taken literally, which makes every block empty and the track
    // silent. The reference documents `0` as "default to 100 ms" and `background_audio.py` relies
    // on passing an explicit one, so this is the path nobody exercises until someone omits it.
    #[test]
    fn a_blocksize_of_zero_means_a_tenth_of_a_second() {
        assert_eq!(Mixer::new(48_000, 1, 0).blocksize(), 4800);
        assert_eq!(Mixer::new(24_000, 2, 0).blocksize(), 2400);
    }

    // Catches: emitting a block of silence when nothing arrived. The reference sleeps 1 ms and
    // continues; emitting instead would publish a continuous silent track at the block rate,
    // which is not the same as publishing nothing and holds the source open forever.
    #[test]
    fn a_round_in_which_no_stream_had_data_emits_nothing() {
        let mut m = mixer(4);
        m.add_stream();
        assert_eq!(m.mix(), None);
        assert_eq!(m.counters.empty_rounds(), 1);

        // ...and so does a mixer with no streams at all.
        let mut m = mixer(4);
        assert_eq!(m.mix(), None);
    }

    // Catches: summing in `i16` (or `f32 -> i32 -> i16`), which wraps. A mix of two loud sources
    // becomes noise instead of a clipped peak -- reported as "the audio is corrupted", not as
    // "the mix is too loud". Note `f32 as i16` itself saturates in Rust, so dropping the `clamp`
    // is an equivalent mutation this test cannot and need not see.
    #[test]
    fn a_loud_mix_clips_rather_than_wrapping() {
        let mut m = mixer(2);
        let a = m.add_stream();
        let b = m.add_stream();
        m.push(a, &[30_000, -30_000]);
        m.push(b, &[30_000, -30_000]);
        assert_eq!(m.mix(), Some(vec![32_767, -32_768]));
    }

    // Catches: retaining a short buffer for the next round. It is the obvious "improvement" and
    // it desynchronises that stream from every other one for the rest of the call, because it
    // then runs a fixed offset behind the block clock.
    #[test]
    fn a_short_buffer_is_padded_and_then_discarded() {
        let mut m = mixer(4);
        let a = m.add_stream();
        m.push(a, &[100, 200]);
        assert_eq!(m.mix(), Some(vec![100, 200, 0, 0]));
        assert_eq!(m.counters.padded_contributions(), 1);

        // The two samples were CONSUMED. Nothing is carried into the next round.
        m.push(a, &[300]);
        assert_eq!(m.mix(), Some(vec![300, 0, 0, 0]));
    }

    // Catches: `had_data` being read as "a frame arrived this round". A stream that was pushed
    // more than a block last round still has data to emit, and reading it the other way drops
    // the tail of every sound whose length is not a multiple of the block.
    #[test]
    fn a_buffer_left_over_from_last_round_counts_as_data() {
        let mut m = mixer(2);
        let a = m.add_stream();
        m.push(a, &[1, 2, 3, 4]);
        assert_eq!(m.mix(), Some(vec![1, 2]));
        // No push this round, but the buffer was non-empty on entry.
        assert_eq!(m.mix(), Some(vec![3, 4]));
        assert_eq!(m.mix(), None, "and now it really is empty");
    }

    // Catches: removing an exhausted stream before it has emitted what it holds. That drops the
    // last partial block of every sound -- up to 100 ms of the end of every clip.
    #[test]
    fn an_exhausted_stream_with_a_partial_buffer_gets_one_more_padded_round() {
        let mut m = mixer(4);
        let a = m.add_stream();
        m.push(a, &[7, 8]);
        m.mark_exhausted(a);
        assert!(!m.wants_more(a), "an exhausted stream is not pulled again");

        assert_eq!(m.mix(), Some(vec![7, 8, 0, 0]));
        assert_eq!(m.drain_removals(), vec![a], "and now it goes");
        assert!(m.is_empty());
    }

    // Catches: an exhausted stream with exactly a whole block being removed a round early, and
    // one with more than a block being removed before it has drained.
    #[test]
    fn an_exhausted_stream_drains_whole_blocks_before_it_is_removed() {
        let mut m = mixer(2);
        let a = m.add_stream();
        m.push(a, &[1, 2, 3, 4]);
        m.mark_exhausted(a);

        assert_eq!(m.mix(), Some(vec![1, 2]));
        assert!(m.drain_removals().is_empty(), "it still holds a block");
        assert_eq!(m.mix(), Some(vec![3, 4]));
        assert_eq!(m.drain_removals(), vec![a]);
    }

    // Catches: `wants_more` measuring samples where the buffer is interleaved, which underfills
    // every stereo block by half.
    #[test]
    fn wants_more_counts_frames_not_samples_on_a_stereo_mixer() {
        let mut m = Mixer::new(48_000, 2, 2);
        let a = m.add_stream();
        assert!(m.wants_more(a));
        m.push(a, &[1, 2]); // one stereo frame of the two the block needs
        assert!(m.wants_more(a));
        m.push(a, &[3, 4]);
        assert!(!m.wants_more(a));
        assert_eq!(m.mix(), Some(vec![1, 2, 3, 4]));
    }

    // Catches: a stream removed while a pull was in flight taking the whole mixer down with it.
    // The host cannot prevent that race -- it awaits the source outside the lock.
    #[test]
    fn pushing_to_a_removed_stream_is_dropped_rather_than_a_panic() {
        let mut m = mixer(2);
        let a = m.add_stream();
        m.remove_stream(a);
        m.push(a, &[1, 2]);
        m.mark_exhausted(a);
        m.note_timeout(a);
        assert!(!m.wants_more(a));
        assert_eq!(m.mix(), None);
        assert_eq!(m.counters.stream_timeouts(), 1);
    }

    // Catches: a timeout being logged instead of counted. A source that times out every round is
    // a sound that silently never plays, and a log line is not observable from a test or a metric.
    #[test]
    fn a_timeout_is_a_counter_and_does_not_stop_the_round() {
        let mut m = mixer(2);
        let a = m.add_stream();
        let b = m.add_stream();
        m.push(a, &[10, 20]);
        m.note_timeout(b);
        assert_eq!(m.mix(), Some(vec![10, 20]), "the other stream still plays");
        assert_eq!(m.counters.stream_timeouts(), 1);
    }

    // Catches: a stream added mid-call inheriting the block clock wrongly, or its first empty
    // round being counted as data.
    #[test]
    fn a_stream_added_with_nothing_in_it_does_not_make_a_round_non_empty() {
        let mut m = mixer(2);
        let a = m.add_stream();
        m.push(a, &[5, 6]);
        assert_eq!(m.mix(), Some(vec![5, 6]));

        m.add_stream();
        assert_eq!(m.mix(), None, "the new stream has nothing and neither has the old one");
    }
}
