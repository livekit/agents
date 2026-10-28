//! `InterruptionCore`: every decision `_forward_data` and `recv_task` make, with no socket.
//! `now: Ts` is for anything measured, `wall: f64` (`time.time()`) for anything reported.

use std::num::NonZeroUsize;
use std::time::Duration;

use lk_agents_abi::errors::ApiError;
use lk_agents_abi::turn::detector::ReqId;
use lk_agents_abi::{HostRef, Ts};
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
    OverlapRequest {
        request: ReqId,
        start: OverlapStart,
    },
    OverlapRequestEnded {
        request: ReqId,
        end: OverlapEnd,
    },
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

/// Authorization of one inference window. The wire echoes `request_id`; the other
/// generations are retained locally so an echo cannot cross a lifecycle boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestIdentity {
    pub request_id: u64,
    pub connection: u64,
    pub speech: u64,
    pub overlap: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum InterruptionEvent {
    ConnectionOpened,
    Input(InputEvent),
    ServerMessage { connection: u64, message: ServerMessage },
    Deadline,
}

#[derive(Debug, Clone, PartialEq)]
pub enum InterruptionEffect {
    InterruptionInference { identity: RequestIdentity, samples: Vec<i16> },
    Overlap { request: Option<ReqId>, event: Box<OverlappingSpeechEvent> },
    SpanUpdate { span: HostRef, attrs: SpanAttributes },
}

#[derive(Debug)]
struct Request {
    identity: RequestIdentity,
    deadline: Ts,
    entry: CacheEntry,
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
    pub request: Option<ReqId>,
    pub error: ApiError,
    pub body: Option<String>,
}

impl StreamError {
    fn status(message: String, status_code: i64, retryable: Option<bool>) -> Self {
        StreamError {
            request: None,
            error: ApiError::status(message, status_code, None, retryable),
            body: None,
        }
    }
}

#[derive(Debug)]
pub struct InterruptionCore {
    threshold: Option<f64>,
    sample_rate: u32,
    inference_timeout: Duration,
    connection: u64,
    speech: u64,
    overlap: u64,
    last_request_id: u64,
    window: AudioWindow,
    cache: BoundedCache<u64, Request>,
    /// Wall clock seconds. Not cleared by `_reset_state`, as in the reference; every reader also
    /// tests `overlap_started`.
    overlap_started_at: Option<f64>,
    overlap_request: Option<ReqId>,
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
            threshold: opts.threshold,
            sample_rate,
            inference_timeout: Duration::from_secs_f64(opts.inference_timeout),
            connection: 0,
            speech: 0,
            overlap: 0,
            last_request_id: 0,
            window,
            cache: BoundedCache::new(NonZeroUsize::new(CACHE_MAXSIZE).expect("10 != 0")),
            overlap_started_at: None,
            overlap_request: None,
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

    /// Applies one supplied-time transition. Transport tasks only execute these effects.
    pub fn handle(
        &mut self,
        event: InterruptionEvent,
        now: Ts,
        wall: f64,
    ) -> Result<Vec<InterruptionEffect>, StreamError> {
        match event {
            InterruptionEvent::ConnectionOpened => {
                let pending = self.next_deadline().is_some();
                let request = self.overlap_request;
                self.connection += 1;
                self.cache.clear();
                if pending {
                    self.reset_state();
                    return Err(StreamError {
                        request,
                        error: ApiError::new(
                            "connection replaced with pending interruption inference".to_owned(),
                        )
                        .with_retryable(false),
                        body: None,
                    });
                }
                Ok(Vec::new())
            }
            InterruptionEvent::Input(input) => {
                let request = self.overlap_request;
                let outcome = self.on_input(input, wall).map_err(|error| StreamError {
                    request: None,
                    error: ApiError::new(error.to_string()).with_retryable(false),
                    body: None,
                })?;
                Ok(match outcome {
                    Some(InputOutcome::Infer(samples)) => {
                        if self.cache.iter().count() == CACHE_MAXSIZE
                            && self
                                .cache
                                .iter()
                                .next()
                                .is_some_and(|(_, request)| request.entry.total_duration.is_none())
                        {
                            let request = self.overlap_request;
                            self.reset_state();
                            return Err(StreamError {
                                request,
                                error: ApiError::new(
                                    "too many pending interruption inferences".to_owned(),
                                )
                                .with_retryable(false),
                                body: None,
                            });
                        }
                        let request_id = now.as_nanos().max(self.last_request_id + 1);
                        self.last_request_id = request_id;
                        let identity = self.record_request_at(request_id, now, samples.clone());
                        vec![InterruptionEffect::InterruptionInference { identity, samples }]
                    }
                    Some(InputOutcome::Overlap(event)) => {
                        vec![InterruptionEffect::Overlap { request, event }]
                    }
                    None => Vec::new(),
                })
            }
            InterruptionEvent::ServerMessage { connection, message } => {
                if connection != self.connection {
                    return Ok(Vec::new());
                }
                match self.on_server_message(message, now, wall)? {
                    RecvOutcome::Detected { span_update, event } => {
                        let mut effects = Vec::new();
                        if let Some((span, attrs)) = span_update {
                            effects.push(InterruptionEffect::SpanUpdate { span, attrs });
                        }
                        effects.push(InterruptionEffect::Overlap {
                            request: self.overlap_request,
                            event,
                        });
                        Ok(effects)
                    }
                    RecvOutcome::Ignored => Ok(Vec::new()),
                }
            }
            InterruptionEvent::Deadline => {
                if let Err(mut error) = self.check_inflight_timeout(now) {
                    error.request = self.overlap_request;
                    // A timeout hands interruption back to the caller's fallback policy.
                    // It must not fabricate a verdict for this or a later overlap.
                    self.reset_state();
                    return Err(error);
                }
                Ok(Vec::new())
            }
        }
    }

    pub fn overlap_request(&self) -> Option<ReqId> {
        self.overlap_request
    }

    pub fn connection(&self) -> u64 {
        self.connection
    }

    pub fn next_deadline(&self) -> Option<Ts> {
        self.cache
            .iter()
            .filter(|(_, request)| {
                request.entry.total_duration.is_none() && self.eligible(request.identity)
            })
            .map(|(_, request)| request.deadline)
            .min()
    }

    pub fn request_eligible(&self, identity: RequestIdentity) -> bool {
        self.eligible(identity)
            && self.cache.iter().any(|(_, request)| {
                request.identity == identity && request.entry.total_duration.is_none()
            })
    }

    fn eligible(&self, identity: RequestIdentity) -> bool {
        self.overlap_started
            && self.agent_speech_started
            && identity.connection == self.connection
            && identity.speech == self.speech
            && identity.overlap == self.overlap
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
        self.overlap_request = None;
    }

    /// `await self._num_requests.get_and_reset()`.
    fn take_num_requests(&mut self) -> u32 {
        std::mem::take(&mut self.num_requests)
    }

    /// `_forward_data`'s match statement (`inference/interruption.py:585`). `wall` is read only by
    /// the overlap-ended branch. Everything except overlap-ended is gated on the agent speaking:
    /// audio pushed otherwise is dropped, not buffered.
    fn on_input(&mut self, ev: InputEvent, wall: f64) -> Result<Option<InputOutcome>, AudioError> {
        match ev {
            InputEvent::OverlapRequest { request, start } => {
                let outcome = self.on_input(InputEvent::OverlapSpeechStarted(start), wall)?;
                self.overlap_request = self.overlap_started.then_some(request);
                Ok(outcome)
            }
            InputEvent::OverlapRequestEnded { request, end } => {
                if self.overlap_request == Some(request) {
                    self.on_input(InputEvent::OverlapSpeechEnded(end), wall)
                } else {
                    Ok(None)
                }
            }
            InputEvent::Flush => Ok(None),
            InputEvent::AgentSpeechStarted => {
                self.reset_state();
                self.speech += 1;
                self.agent_speech_started = true;
                Ok(None)
            }
            InputEvent::AgentSpeechEnded => {
                self.reset_state();
                Ok(None)
            }
            InputEvent::OverlapSpeechStarted(start) if self.agent_speech_started => {
                self.overlap += 1;
                self.overlap_request = None;
                self.overlap_started_at = Some(start.started_at);
                self.user_speech_span = start.user_speaking_span;
                self.overlap_started = true;
                self.accumulated_samples = 0;
                self.overlap_count += 1;
                // Only the first overlap of an agent turn trims leading silence; re-trimming would
                // drop the earlier overlap the model needs as context.
                if self.overlap_count == 1 {
                    let keep = (start.speech_duration * self.sample_rate as f64) as i64
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
                    let last = self.cache.pop_if(|request| request.entry.total_duration.is_some());
                    let entry = last.map(|(_, request)| request.entry).unwrap_or_default();
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

    pub fn check_inflight_timeout(&self, now: Ts) -> Result<(), StreamError> {
        if self.next_deadline().is_some_and(|deadline| now >= deadline) {
            return Err(StreamError::status(
                format!(
                    "interruption inference timed out after {:.1}s (ws)",
                    self.inference_timeout.as_secs_f64()
                ),
                408,
                Some(false),
            ));
        }
        Ok(())
    }

    /// Register before execution, so even an immediate provider reply has an owner.
    #[cfg(test)]
    fn record_request(&mut self, created_at: Ts, speech_input: Vec<i16>) {
        self.last_request_id = self.last_request_id.max(created_at.as_nanos());
        self.record_request_at(created_at.as_nanos(), created_at, speech_input);
    }

    fn record_request_at(
        &mut self,
        request_id: u64,
        now: Ts,
        speech_input: Vec<i16>,
    ) -> RequestIdentity {
        self.num_requests += 1;
        let identity = RequestIdentity {
            request_id,
            connection: self.connection,
            speech: self.speech,
            overlap: self.overlap,
        };
        self.cache.insert(
            request_id,
            Request {
                identity,
                deadline: now.saturating_add(self.inference_timeout),
                entry: CacheEntry {
                    created_at: now.as_nanos(),
                    speech_input: Some(speech_input),
                    ..Default::default()
                },
            },
        );
        identity
    }

    /// `recv_task`'s match statement (`inference/interruption.py:854`).
    fn on_server_message(
        &mut self,
        msg: ServerMessage,
        now: Ts,
        wall: f64,
    ) -> Result<RecvOutcome, StreamError> {
        match msg {
            ServerMessage::SessionCreated { default_threshold } => {
                // #6034: "error when no interruption threshold is known".
                if self.threshold.is_none() && default_threshold.is_none() {
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
                let entry = self.apply_verdict(&v, true, started_at, now, wall);
                let Some(entry) = entry else { return Ok(RecvOutcome::Ignored) };
                let span_update =
                    self.user_speech_span.take().map(|s| (s, SpanAttributes::from_entry(&entry)));
                let mut event = overlapping_speech_from_cache_entry(
                    &entry,
                    true,
                    Some(started_at),
                    // `ended_at=time.time()`: a interruption ends the overlap now.
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
                self.apply_verdict(&v, false, started_at, now, wall);
                Ok(RecvOutcome::Ignored)
            }
            ServerMessage::Error { message, code, .. } => Err(StreamError {
                request: None,
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

    fn apply_verdict(
        &mut self,
        verdict: &Verdict,
        is_interruption: bool,
        started_at: f64,
        now: Ts,
        wall: f64,
    ) -> Option<CacheEntry> {
        let identity = self.cache.iter().find(|(id, _)| *id == verdict.created_at)?.1.identity;
        if !self.eligible(identity) {
            return None;
        }
        let request = self.cache.get_mut(&verdict.created_at)?;
        if request.entry.total_duration.is_some() || now >= request.deadline {
            return None;
        }
        let entry = &mut request.entry;
        entry.total_duration = Some(now.as_nanos().saturating_sub(entry.created_at) as f64 / 1e9);
        entry.prediction_duration = Some(verdict.prediction_duration);
        entry.detection_delay = Some(wall - started_at);
        entry.probabilities = Some(verdict.probabilities.clone());
        entry.is_interruption = Some(is_interruption);
        Some(entry.clone())
    }
}

#[cfg(test)]
mod tests;
