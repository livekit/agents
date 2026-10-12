//! `InterruptionCore`: every decision `_forward_data` and `recv_task` make, with no socket.
//! `now: Ts` is for anything measured, `wall: f64` (`time.time()`) for anything reported.

use std::num::NonZeroUsize;

use lk_agents_abi::{HostRef, Ts};
use lk_agents_abi::errors::ApiError;
use thiserror::Error;

use crate::interruption::audio::{AudioError, AudioFrame, AudioWindow};
use crate::interruption::cache::BoundedCache;
use crate::interruption::event::{
    CacheEntry, OverlappingSpeechEvent, overlapping_speech_from_cache_entry,
};
use crate::interruption::options::InterruptionOptions;
use crate::interruption::proto::{ServerMessage, Verdict};

/// `BoundedDict(maxsize=10)` (`inference/interruption.py:433`): a leak guard, not protocol.
const CACHE_MAXSIZE: usize = 10;

/// `_OverlapSpeechStartedSentinel` (`inference/interruption.py:231`).
#[derive(Debug, Clone, PartialEq)]
pub struct OverlapStart {
    /// How long the user has already been speaking, per VAD; trims the first window.
    pub speech_duration: f64,
    pub started_at: f64,
    /// The OTel user-speech span, as an opaque host handle (§1.11), handed back once.
    pub user_speaking_span: Option<HostRef>,
}

/// `_OverlapSpeechEndedSentinel` (`inference/interruption.py:242`).
#[derive(Debug, Clone, PartialEq)]
pub struct OverlapEnd {
    pub ended_at: f64,
    pub agent_ended: bool,
}

/// `InterruptionDataFrameType` (`inference/interruption.py:254`) -- one channel carrying audio and
/// four lifecycle sentinels, so ordering between them is guaranteed.
#[derive(Debug, Clone, PartialEq)]
pub enum InputEvent {
    AgentSpeechStarted,
    AgentSpeechEnded,
    OverlapSpeechStarted(OverlapStart),
    OverlapSpeechEnded(OverlapEnd),
    Audio(AudioFrame),
    /// `_FlushSentinel`. Ignored by `_forward_data`; `end_input()` pushes it before closing.
    Flush,
}

#[derive(Debug, Clone, PartialEq)]
pub enum InputOutcome {
    Infer(Vec<i16>),
    /// The overlap ended without a verdict.
    Overlap(Box<OverlappingSpeechEvent>),
}

/// The five values `_update_user_speech_span` writes (`inference/interruption.py:548`).
#[derive(Debug, Clone, PartialEq)]
pub struct SpanAttributes {
    /// `str(entry.is_interruption).lower()` -- a string; "none" is reachable.
    pub is_interruption: &'static str,
    pub probability: f64,
    pub total_duration: f64,
    pub prediction_duration: f64,
    pub detection_delay: f64,
}

impl SpanAttributes {
    fn from_entry(entry: &CacheEntry) -> Self {
        SpanAttributes {
            is_interruption: match entry.is_interruption {
                Some(true) => "true",
                Some(false) => "false",
                None => "none",
            },
            probability: entry.probability(),
            total_duration: entry.total_duration(),
            prediction_duration: entry.prediction_duration(),
            detection_delay: entry.detection_delay(),
        }
    }
}

/// What a server message means. A verdict arriving after the overlap closed is dropped, not
/// applied late.
#[derive(Debug, Clone, PartialEq)]
pub enum RecvOutcome {
    /// `session.created`, `session.closed`, an echoed client message, `inference_done`, or a
    /// verdict outside an open overlap.
    Ignored,
    /// `bargein_detected` inside an open overlap, which closes it.
    Detected {
        /// `Some` only the first time: a second verdict must not re-decorate a span the driver
        /// may have ended.
        span_update: Option<(HostRef, SpanAttributes)>,
        event: Box<OverlappingSpeechEvent>,
    },
}

/// An `APIStatusError` raised out of the stream, plus the server's `body` text (which
/// [`ApiError`] does not carry).
#[derive(Debug, Clone, PartialEq, Error)]
#[error("{error}")]
pub struct StreamError {
    pub error: ApiError,
    pub body: Option<String>,
}

impl StreamError {
    fn status(message: String, status_code: i64, retryable: Option<bool>) -> Self {
        StreamError { error: ApiError::status(message, status_code, None, retryable), body: None }
    }
}

#[derive(Debug)]
pub struct InterruptionCore {
    opts: InterruptionOptions,
    window: AudioWindow,
    cache: BoundedCache<u64, CacheEntry>,
    /// Wall clock seconds. Not cleared by `_reset_state`, as in the reference; every reader also
    /// tests `overlap_started`.
    overlap_started_at: Option<f64>,
    user_speech_span: Option<HostRef>,
    agent_speech_started: bool,
    overlap_started: bool,
    overlap_count: u32,
    accumulated_samples: usize,
    num_requests: u32,
    batch_size: usize,
    prefix_size: usize,
}

impl InterruptionCore {
    pub fn new(opts: InterruptionOptions) -> Self {
        let sample_rate = opts.sample_rate;
        let window =
            AudioWindow::new((opts.max_audio_duration * sample_rate as f64) as usize, sample_rate);
        let batch_size = (opts.detection_interval * sample_rate as f64) as usize;
        let prefix_size = (opts.audio_prefix_duration * sample_rate as f64) as usize;
        InterruptionCore {
            opts,
            window,
            cache: BoundedCache::new(NonZeroUsize::new(CACHE_MAXSIZE).expect("10 != 0")),
            overlap_started_at: None,
            user_speech_span: None,
            agent_speech_started: false,
            overlap_started: false,
            overlap_count: 0,
            accumulated_samples: 0,
            num_requests: 0,
            batch_size,
            prefix_size,
        }
    }

    /// `_forward_data`'s `_reset_state` (`inference/interruption.py:576`).
    fn reset_state(&mut self) {
        self.agent_speech_started = false;
        self.overlap_started = false;
        self.overlap_count = 0;
        self.accumulated_samples = 0;
        self.num_requests = 0;
        self.window.reset();
        self.cache.clear();
        self.user_speech_span = None;
    }

    /// `await self._num_requests.get_and_reset()`.
    fn take_num_requests(&mut self) -> u32 {
        std::mem::take(&mut self.num_requests)
    }

    /// `_forward_data`'s match statement (`inference/interruption.py:585`). `wall` is read only by
    /// the overlap-ended branch. Everything except overlap-ended is gated on the agent speaking:
    /// audio pushed otherwise is dropped, not buffered.
    pub fn on_input(
        &mut self,
        ev: InputEvent,
        wall: f64,
    ) -> Result<Option<InputOutcome>, AudioError> {
        match ev {
            InputEvent::Flush => Ok(None),
            InputEvent::AgentSpeechStarted => {
                self.reset_state();
                self.agent_speech_started = true;
                Ok(None)
            }
            InputEvent::AgentSpeechEnded => {
                self.reset_state();
                Ok(None)
            }
            InputEvent::OverlapSpeechStarted(start) if self.agent_speech_started => {
                self.overlap_started_at = Some(start.started_at);
                self.user_speech_span = start.user_speaking_span;
                self.overlap_started = true;
                self.accumulated_samples = 0;
                self.overlap_count += 1;
                // Only the first overlap of an agent turn trims leading silence; re-trimming would
                // drop the earlier overlap the model needs as context.
                if self.overlap_count == 1 {
                    let keep = (start.speech_duration * self.opts.sample_rate as f64) as i64
                        + self.prefix_size as i64;
                    let shift = (self.window.len() as i64 - keep).max(0) as usize;
                    self.window.shift(shift);
                }
                self.cache.clear();
                Ok(None)
            }
            InputEvent::OverlapSpeechStarted(_) => Ok(None),
            InputEvent::OverlapSpeechEnded(end) => {
                let outcome = if self.overlap_started && self.overlap_started_at.is_some() {
                    self.user_speech_span = None;
                    // "The newest request that actually came back". `> 0` because `created_at` is
                    // echoed by the server, so a bogus stamp yields a non-positive RTT.
                    let last = self.cache.pop_if(|e| e.total_duration.is_some_and(|d| d > 0.0));
                    let entry = last.map(|(_, e)| e).unwrap_or_default();
                    let mut event = overlapping_speech_from_cache_entry(
                        &entry,
                        false,
                        self.overlap_started_at,
                        Some(end.ended_at),
                        end.agent_ended,
                        wall,
                    );
                    event.num_requests = self.take_num_requests();
                    Some(InputOutcome::Overlap(Box::new(event)))
                } else {
                    None
                };
                self.overlap_started = false;
                self.accumulated_samples = 0;
                self.overlap_started_at = None;
                // The cache is not cleared: responses for requests sent during the overlap may
                // still be in flight.
                Ok(outcome)
            }
            InputEvent::Audio(frame) if self.agent_speech_started => {
                let written = self.window.push_frame(&frame)?;
                self.accumulated_samples += written;
                // The window fills regardless (pre-overlap prefix), but only an open overlap turns
                // a full batch into a request.
                if self.accumulated_samples >= self.batch_size && self.overlap_started {
                    self.accumulated_samples = 0;
                    return Ok(Some(InputOutcome::Infer(self.window.read())));
                }
                Ok(None)
            }
            InputEvent::Audio(_) => Ok(None),
        }
    }

    /// `send_task`'s per-chunk preamble (`inference/interruption.py:800`). Scans oldest-first,
    /// skipping answered entries, and stops at the first unanswered one (keys are monotonic).
    /// Not retryable.
    pub fn check_inflight_timeout(&self, now: Ts) -> Result<(), StreamError> {
        let timeout_ns = (self.opts.inference_timeout * 1e9) as u64;
        for (_, entry) in self.cache.iter() {
            if entry.total_duration.is_some() {
                continue;
            }
            let elapsed = now.as_nanos().saturating_sub(entry.created_at);
            if elapsed > timeout_ns {
                return Err(StreamError::status(
                    format!(
                        "interruption inference timed out after {:.1}s (ws)",
                        elapsed as f64 / 1e9
                    ),
                    408,
                    Some(false),
                ));
            }
            break;
        }
        Ok(())
    }

    /// `send_task`'s bookkeeping: bump the request counter and open a cache entry keyed by the
    /// header stamp.
    pub fn record_request(&mut self, created_at: Ts, speech_input: Vec<i16>) {
        self.num_requests += 1;
        let created_at = created_at.as_nanos();
        self.cache.insert(
            created_at,
            CacheEntry { created_at, speech_input: Some(speech_input), ..Default::default() },
        );
    }

    /// `recv_task`'s match statement (`inference/interruption.py:854`).
    pub fn on_server_message(
        &mut self,
        msg: ServerMessage,
        now: Ts,
        wall: f64,
    ) -> Result<RecvOutcome, StreamError> {
        match msg {
            ServerMessage::SessionCreated { default_threshold } => {
                // #6034: "error when no interruption threshold is known".
                if self.opts.threshold.is_none() && default_threshold.is_none() {
                    return Err(StreamError::status(
                        "adaptive interruption session created without a threshold: no user \
                         override and the server did not report a default_threshold"
                            .to_owned(),
                        500,
                        Some(false),
                    ));
                }
                Ok(RecvOutcome::Ignored)
            }
            ServerMessage::SessionClosed
            | ServerMessage::EchoedSessionCreate { .. }
            | ServerMessage::EchoedSessionClose => Ok(RecvOutcome::Ignored),
            ServerMessage::Detected(v) => {
                let Some(started_at) = self.open_overlap() else {
                    return Ok(RecvOutcome::Ignored);
                };
                let entry =
                    Self::apply_verdict(&mut self.cache, &v, true, started_at, now, wall).clone();
                let span_update =
                    self.user_speech_span.take().map(|s| (s, SpanAttributes::from_entry(&entry)));
                let mut event = overlapping_speech_from_cache_entry(
                    &entry,
                    true,
                    Some(started_at),
                    // `ended_at=time.time()`: a barge-in ends the overlap now.
                    Some(wall),
                    false,
                    wall,
                );
                event.num_requests = self.take_num_requests();
                // `overlap_started_at` stays set, as in the reference.
                self.overlap_started = false;
                Ok(RecvOutcome::Detected { span_update, event: Box::new(event) })
            }
            ServerMessage::InferenceDone(v) => {
                let Some(started_at) = self.open_overlap() else {
                    return Ok(RecvOutcome::Ignored);
                };
                Self::apply_verdict(&mut self.cache, &v, false, started_at, now, wall);
                Ok(RecvOutcome::Ignored)
            }
            ServerMessage::Error { message, code, .. } => Err(StreamError {
                error: ApiError::status(
                    format!("LiveKit Adaptive Interruption returned error: {code}"),
                    code,
                    None,
                    None,
                ),
                body: Some(message),
            }),
        }
    }

    /// `if (overlap_started_at := self._overlap_started_at) is not None and self._overlap_started`.
    fn open_overlap(&self) -> Option<f64> {
        self.overlap_started.then_some(self.overlap_started_at).flatten()
    }

    /// The `set_or_update` shared by both verdict branches; associated to avoid a borrow clash.
    fn apply_verdict<'a>(
        cache: &'a mut BoundedCache<u64, CacheEntry>,
        v: &Verdict,
        is_interruption: bool,
        started_at: f64,
        now: Ts,
        wall: f64,
    ) -> &'a mut CacheEntry {
        let created_at = v.created_at;
        cache.set_or_update(
            created_at,
            || CacheEntry::new(created_at),
            |entry| {
                // `saturating_sub`: the server could echo a stamp we never sent.
                entry.total_duration = Some(now.as_nanos().saturating_sub(created_at) as f64 / 1e9);
                entry.prediction_duration = Some(v.prediction_duration);
                // Wall clock, because `started_at` came from the host's `time.time()`.
                entry.detection_delay = Some(wall - started_at);
                entry.probabilities = Some(v.probabilities.clone());
                entry.is_interruption = Some(is_interruption);
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interruption::options::{
        AUDIO_PREFIX_DURATION, DETECTION_INTERVAL, InterruptionOptions, MAX_AUDIO_DURATION,
        MIN_INTERRUPTION_DURATION, REMOTE_INFERENCE_TIMEOUT,
    };

    const SR: u32 = 16_000;

    /// The defaults, with the credentials already resolved.
    fn core_with(threshold: Option<f64>) -> InterruptionCore {
        InterruptionCore::new(
            InterruptionOptions::new(
                threshold,
                MIN_INTERRUPTION_DURATION,
                MAX_AUDIO_DURATION,
                AUDIO_PREFIX_DURATION,
                DETECTION_INTERVAL,
                REMOTE_INFERENCE_TIMEOUT,
                "https://gateway".to_owned(),
                "k".to_owned(),
                "s".to_owned(),
            )
            .unwrap(),
        )
    }

    fn core() -> InterruptionCore {
        core_with(None)
    }

    /// `n` samples of mono 16 kHz, each equal to `value`, so a window's contents are identifiable.
    fn frame(n: usize, value: i16) -> AudioFrame {
        AudioFrame {
            data: std::iter::repeat_n(value, n).flat_map(i16::to_le_bytes).collect(),
            sample_rate: SR,
            num_channels: 1,
            samples_per_channel: n as u32,
        }
    }

    fn ts(secs: f64) -> Ts {
        Ts::from_nanos((secs * 1e9) as u64)
    }

    fn end_overlap(c: &mut InterruptionCore, ended_at: f64, agent_ended: bool, now: f64) -> Option<InputOutcome> {
        c.on_input(InputEvent::OverlapSpeechEnded(OverlapEnd { ended_at, agent_ended }), now).unwrap()
    }

    /// The server's answer to the request stamped `sent`, arriving at `at`.
    fn verdict(c: &mut InterruptionCore, detected: bool, sent: f64, prediction_duration: f64, probabilities: &[f32], at: f64) -> RecvOutcome {
        let v = Verdict { created_at: ts(sent).as_nanos(), prediction_duration, probabilities: probabilities.to_vec() };
        let msg = if detected { ServerMessage::Detected(v) } else { ServerMessage::InferenceDone(v) };
        c.on_server_message(msg, ts(at), at).unwrap()
    }

    fn start_overlap(c: &mut InterruptionCore, speech_duration: f64, at: f64) {
        c.on_input(
            InputEvent::OverlapSpeechStarted(OverlapStart {
                speech_duration,
                started_at: at,
                user_speaking_span: None,
            }),
            at,
        )
        .unwrap();
    }

    // Catches: buffering audio before the agent speaks, or shipping it after it stops.
    #[test]
    fn audio_outside_an_agent_turn_is_dropped_entirely() {
        let mut c = core();
        assert_eq!(c.on_input(InputEvent::Audio(frame(1600, 1)), 0.0).unwrap(), None);
        assert_eq!(c.window.len(), 0, "buffered audio with no agent speaking");

        c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
        c.on_input(InputEvent::Audio(frame(1600, 1)), 0.0).unwrap();
        assert_eq!(c.window.len(), 1600);

        c.on_input(InputEvent::AgentSpeechEnded, 0.0).unwrap();
        assert_eq!(c.window.len(), 0, "the window must be reset when the agent stops");
        c.on_input(InputEvent::Audio(frame(1600, 1)), 0.0).unwrap();
        assert_eq!(c.window.len(), 0);
    }

    // Catches: an overlap-started sentinel opening an overlap while the agent is silent.
    #[test]
    fn an_overlap_cannot_open_while_the_agent_is_silent() {
        let mut c = core();
        start_overlap(&mut c, 0.0, 1.0);
        assert!(!c.overlap_started);
        assert_eq!(c.overlap_started_at, None);
        // ... and no event is produced when it "ends", because it never began.
        let out = end_overlap(&mut c, 2.0, false, 2.0);
        assert_eq!(out, None);
    }

    // Catches: a request on every frame instead of every `detection_interval`.
    #[test]
    fn requests_are_emitted_once_per_detection_interval_not_per_frame() {
        let mut c = core();
        c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
        start_overlap(&mut c, 0.0, 1.0);

        let batch = (DETECTION_INTERVAL_SAMPLES) / 10;
        let mut requests = 0;
        for _ in 0..10 {
            if let Some(InputOutcome::Infer(_)) =
                c.on_input(InputEvent::Audio(frame(batch, 3)), 1.0).unwrap()
            {
                requests += 1;
            }
        }
        assert_eq!(requests, 1, "1600 samples of 160-sample frames is exactly one request");
    }
    const DETECTION_INTERVAL_SAMPLES: usize = 1600;

    // Catches: dropping the audio prefix; the trim keeps the prefix plus the speech already
    // detected.
    #[test]
    fn the_first_request_carries_the_configured_prefix_of_pre_overlap_audio() {
        let mut c = core();
        c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
        // 2 s of agent-only audio, then the user has been talking for 0.5 s.
        c.on_input(InputEvent::Audio(frame(2 * SR as usize, 7)), 0.0).unwrap();
        start_overlap(&mut c, 0.5, 2.0);
        // prefix 1.0 s + speech 0.5 s = 1.5 s kept out of the 2 s buffered.
        assert_eq!(c.window.len(), (1.5 * SR as f64) as usize);

        let out = c.on_input(InputEvent::Audio(frame(1600, 9)), 2.0).unwrap();
        let Some(InputOutcome::Infer(samples)) = out else { panic!("expected a request") };
        assert_eq!(samples.len(), (1.5 * SR as f64) as usize + 1600);
    }

    // Catches: re-trimming on a second overlap within one agent turn.
    #[test]
    fn a_second_overlap_in_the_same_agent_turn_does_not_re_trim_the_window() {
        let mut c = core();
        c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
        c.on_input(InputEvent::Audio(frame(2 * SR as usize, 7)), 0.0).unwrap();
        start_overlap(&mut c, 0.5, 2.0);
        let after_first = c.window.len();

        end_overlap(&mut c, 2.2, false, 2.2);
        start_overlap(&mut c, 0.0, 2.5);
        assert_eq!(c.window.len(), after_first, "the window was trimmed a second time");
        assert_eq!(c.overlap_count, 2);
    }

    // Catches: clearing the cache when the overlap ends.
    #[test]
    fn ending_an_overlap_keeps_in_flight_entries_but_starting_one_clears_them() {
        let mut c = core();
        c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
        start_overlap(&mut c, 0.0, 1.0);
        c.record_request(ts(1.0), vec![1, 2]);
        c.record_request(ts(1.1), vec![3, 4]);

        end_overlap(&mut c, 1.2, false, 1.2);
        assert_eq!(c.cache.iter().count(), 2, "in-flight requests were dropped");

        start_overlap(&mut c, 0.0, 1.3);
        assert_eq!(c.cache.iter().count(), 0, "a new overlap must start from a clean cache");
    }

    // Catches: reporting the first answered inference rather than the last, and counting requests
    // cumulatively instead of per overlap.
    #[test]
    fn overlap_end_reports_the_newest_answered_inference_and_drains_the_counter() {
        let mut c = core();
        c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
        start_overlap(&mut c, 0.0, 1.0);
        c.record_request(ts(1.0), vec![1]);
        c.record_request(ts(1.1), vec![2]);
        // Both come back; the second one is slower.
        verdict(&mut c, false, 1.0, 0.01, &[0.1, 0.1], 1.05);
        verdict(&mut c, false, 1.1, 0.02, &[0.4, 0.3], 1.3);

        let out = end_overlap(&mut c, 1.4, false, 1.45);
        let Some(InputOutcome::Overlap(ev)) = out else { panic!("expected an event") };
        assert!(!ev.is_interruption);
        assert_eq!(ev.num_requests, 2);
        assert_eq!(ev.prediction_duration, 0.02, "reported the older inference");
        assert!((ev.total_duration - 0.2).abs() < 1e-9);
        assert_eq!(ev.overlap_started_at, Some(1.0));
        assert_eq!(ev.detected_at, 1.4);

        // And the counter is drained, not merely read.
        start_overlap(&mut c, 0.0, 2.0);
        let out = end_overlap(&mut c, 2.1, false, 2.1);
        let Some(InputOutcome::Overlap(ev)) = out else { panic!("expected an event") };
        assert_eq!(ev.num_requests, 0);
    }

    // Catches: `pop_if` accepting a zero-RTT entry created by the response path.
    #[test]
    fn an_unanswered_entry_never_describes_the_overlap() {
        let mut c = core();
        c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
        start_overlap(&mut c, 0.0, 1.0);
        c.record_request(ts(1.0), vec![1]);
        c.record_request(ts(1.1), vec![2]);
        verdict(&mut c, false, 1.0, 0.05, &[0.2, 0.2], 1.2);

        let out = end_overlap(&mut c, 1.3, false, 1.3);
        let Some(InputOutcome::Overlap(ev)) = out else { panic!("expected an event") };
        assert_eq!(ev.prediction_duration, 0.05);
        assert_eq!(ev.speech_input, Some(vec![1]));
    }

    // Catches: `pop_if` accepting a non-positive round trip from a bogus echoed stamp.
    #[test]
    fn a_verdict_echoing_a_stamp_we_never_sent_does_not_describe_the_overlap() {
        let mut c = core();
        c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
        start_overlap(&mut c, 0.0, 1.0);
        c.record_request(ts(1.0), vec![1]);
        // The server answers a request stamped in the future.
        verdict(&mut c, false, 99.0, 0.5, &[0.7, 0.7], 1.2);

        let out = end_overlap(&mut c, 1.3, false, 1.3);
        let Some(InputOutcome::Overlap(ev)) = out else { panic!("expected an event") };
        assert_eq!(
            (ev.total_duration, ev.prediction_duration, ev.probability),
            (0.0, 0.0, 0.0),
            "a zero round trip must not be reported as a real inference"
        );
    }

    // Catches: emitting nothing when no request was made, or stale durations instead of 0.0.
    #[test]
    fn an_overlap_with_no_requests_still_emits_a_zeroed_event() {
        let mut c = core();
        c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
        start_overlap(&mut c, 0.0, 1.0);
        let out = end_overlap(&mut c, 1.1, true, 1.1);
        let Some(InputOutcome::Overlap(ev)) = out else { panic!("expected an event") };
        assert!(ev.agent_ended && !ev.is_interruption);
        assert_eq!((ev.total_duration, ev.probability, ev.num_requests), (0.0, 0.0, 0));
        assert_eq!(ev.speech_input, None);
    }

    // Catches: an overlap reported twice, by the verdict and by the end sentinel.
    #[test]
    fn a_detected_verdict_closes_the_overlap_so_the_end_sentinel_emits_nothing() {
        let mut c = core();
        c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
        start_overlap(&mut c, 0.0, 1.0);
        c.record_request(ts(1.0), vec![5]);

        let out = verdict(&mut c, true, 1.0, 0.03, &[0.9, 0.8, 0.1], 1.15);
        let RecvOutcome::Detected { event, span_update } = out else { panic!("expected Detected") };
        assert!(event.is_interruption);
        assert_eq!(event.num_requests, 1);
        assert_eq!(event.probability, 0.8_f32 as f64);
        assert!((event.total_duration - 0.15).abs() < 1e-9);
        assert!((event.detection_delay - 0.15).abs() < 1e-9);
        assert_eq!(span_update, None, "no span was supplied");

        assert_eq!(
            end_overlap(&mut c, 1.2, false, 1.2),
            None
        );
    }

    // Catches: applying a verdict that arrives after the overlap closed.
    #[test]
    fn a_verdict_arriving_after_the_overlap_closed_is_ignored() {
        let mut c = core();
        c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
        start_overlap(&mut c, 0.0, 1.0);
        c.record_request(ts(1.0), vec![5]);
        end_overlap(&mut c, 1.1, false, 1.1);

        let out = verdict(&mut c, true, 1.0, 0.03, &[0.9, 0.9], 1.2);
        assert_eq!(out, RecvOutcome::Ignored);
    }

    // Catches: #6034 regressing (no threshold known must be an error).
    #[test]
    fn a_session_created_with_no_threshold_anywhere_is_fatal() {
        let mut c = core();
        let err = c
            .on_server_message(
                ServerMessage::SessionCreated { default_threshold: None },
                ts(0.0),
                0.0,
            )
            .unwrap_err();
        assert_eq!(err.error.status_code, Some(500));
        assert!(!err.error.retryable, "reconnecting cannot conjure a threshold");

        // Either source alone is enough.
        let user = ServerMessage::SessionCreated { default_threshold: None };
        assert_eq!(core_with(Some(0.9)).on_server_message(user, ts(0.0), 0.0), Ok(RecvOutcome::Ignored));
        let server = ServerMessage::SessionCreated { default_threshold: Some(0.5) };
        assert_eq!(core().on_server_message(server, ts(0.0), 0.0), Ok(RecvOutcome::Ignored));
    }

    // Catches: the in-flight scan stopping at an answered entry, or firing inside the window.
    #[test]
    fn the_inflight_scan_skips_answered_entries_and_trips_only_past_the_timeout() {
        let mut c = core();
        c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
        start_overlap(&mut c, 0.0, 1.0);
        c.record_request(ts(1.0), vec![1]);
        c.record_request(ts(1.1), vec![2]);
        // The oldest is answered; the scan must move past it to the second one.
        verdict(&mut c, false, 1.0, 0.0, &[], 1.05);

        assert!(c.check_inflight_timeout(ts(1.79)).is_ok(), "0.69s is inside the 0.7s window");
        assert!(
            c.check_inflight_timeout(ts(1.8)).is_ok(),
            "exactly 0.7s is not yet past the timeout -- the reference compares with `>`"
        );
        let err = c.check_inflight_timeout(ts(1.9)).unwrap_err();
        assert_eq!(err.error.status_code, Some(408));
        assert!(!err.error.retryable);
        assert_eq!(err.error.message, "interruption inference timed out after 0.8s (ws)");
    }

    // Catches: the scan walking the whole cache instead of stopping at the first unanswered entry.
    #[test]
    fn the_inflight_scan_stops_at_the_oldest_unanswered_entry() {
        let mut c = core();
        c.record_request(ts(10.0), vec![1]);
        c.record_request(ts(0.0), vec![2]); // out of order: only reachable if the keys broke
        assert!(
            c.check_inflight_timeout(ts(10.1)).is_ok(),
            "the scan must not look past the first unanswered entry"
        );
    }

    // Catches: the span being decorated on every verdict instead of once.
    #[test]
    fn the_user_speech_span_is_handed_back_exactly_once() {
        use std::sync::Arc;
        struct Noop;
        impl lk_agents_abi::RefRegistry for Noop {
            fn release(&self, _id: u64) {}
        }
        let registry: Arc<dyn lk_agents_abi::RefRegistry> = Arc::new(Noop);

        let mut c = core();
        c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
        c.on_input(
            InputEvent::OverlapSpeechStarted(OverlapStart {
                speech_duration: 0.0,
                started_at: 1.0,
                user_speaking_span: Some(HostRef::new(3, &registry)),
            }),
            1.0,
        )
        .unwrap();
        c.record_request(ts(1.0), vec![1]);

        let out = verdict(&mut c, true, 1.0, 0.0, &[0.9, 0.7], 1.1);
        let RecvOutcome::Detected { span_update: Some((span, attrs)), .. } = out else {
            panic!("expected a span update")
        };
        assert_eq!(span.id(), 3);
        assert_eq!(attrs.is_interruption, "true");
        assert_eq!(attrs.probability, 0.7_f32 as f64);
        assert!(c.user_speech_span.is_none(), "the span must be released as it is written");
    }

    // Catches: a server `error` frame losing its payload (`body`) or status code.
    #[test]
    fn a_server_error_carries_the_code_as_status_and_the_text_as_body() {
        let mut c = core();
        let err = c
            .on_server_message(
                ServerMessage::Error {
                    message: "model unavailable".to_owned(),
                    code: 503,
                    session_id: "sess".to_owned(),
                },
                ts(0.0),
                0.0,
            )
            .unwrap_err();
        assert_eq!(err.error.status_code, Some(503));
        assert_eq!(err.error.message, "LiveKit Adaptive Interruption returned error: 503");
        assert_eq!(err.body.as_deref(), Some("model unavailable"));
        assert!(err.error.retryable, "a 5xx is worth reconnecting for");

        let err = c
            .on_server_message(
                ServerMessage::Error {
                    message: "bad key".to_owned(),
                    code: 401,
                    session_id: "sess".to_owned(),
                },
                ts(0.0),
                0.0,
            )
            .unwrap_err();
        assert!(!err.error.retryable, "a 401 will keep failing (livekit/agents#5831)");
    }

    // Catches: `_reset_state` missing a field.
    #[test]
    fn agent_speech_start_re_arms_every_field_that_survives_a_turn() {
        let mut c = core();
        c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
        c.on_input(InputEvent::Audio(frame(800, 1)), 0.0).unwrap();
        start_overlap(&mut c, 0.0, 1.0);
        c.record_request(ts(1.0), vec![1]);
        assert!(c.overlap_started && c.num_requests == 1 && c.cache.iter().count() == 1);

        c.on_input(InputEvent::AgentSpeechStarted, 2.0).unwrap();
        assert!(c.agent_speech_started);
        assert!(!c.overlap_started);
        assert_eq!(c.overlap_count, 0);
        assert_eq!(c.accumulated_samples, 0);
        assert_eq!(c.num_requests, 0);
        assert_eq!(c.cache.iter().count(), 0);
        assert_eq!(c.window.len(), 0);
        assert!(c.user_speech_span.is_none());
    }

    // Catches: `Flush` resetting state or emitting an event.
    #[test]
    fn flush_is_inert() {
        let mut c = core();
        c.on_input(InputEvent::AgentSpeechStarted, 0.0).unwrap();
        start_overlap(&mut c, 0.0, 1.0);
        assert_eq!(c.on_input(InputEvent::Flush, 1.0).unwrap(), None);
        assert!(c.overlap_started && c.agent_speech_started);
    }
}
