use lk_agents_abi::voice::rtc::RtcError;
use lk_agents_abi::voice::rtc::capabilities::AudioSource;
use lk_agents_abi::voice::rtc::{PlaybackInfo, TrackPublishOptions, frame_duration};
type Result<T> = std::result::Result<T, RtcError>;
use lk_agents_abi::AudioFrame;
use lk_agents_abi::{Clock, Ts};
use lk_agents_voice::room_io::machine::playout::PlayoutLedger;
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Notify;
use tokio::sync::futures::Notified;

/// `rtc.AudioSource`, minus the speaker.
///
/// A capture longer than the queue goes in as the SDK feeds it (`libwebrtc`'s
/// `NativeAudioSource::capture_frame`): queue-sized pieces, each once the last has drained, and
/// `clear_queue` cuts only what is buffered, so the pieces still to come refill the speaker after
/// it. Nor does dropping the capture stop them: the SDK runs it as its own task.
pub struct FakeAudioSource {
    clock: Arc<dyn Clock>,
    queue_size: Duration,
    close_started: tokio::sync::watch::Sender<bool>,
    close_flow: tokio::sync::watch::Sender<bool>,
    state: Mutex<SourceState>,
    /// Woken by `settle`, `clear_queue` and `close`; parked on by `capture` and `wait_for_playout`.
    wake: Notify,
}

#[derive(Default)]
struct SourceState {
    closed: bool,
    queue: VecDeque<Queued>,
    /// The pieces of a capture longer than the queue, not buffered yet.
    pending: VecDeque<Pending>,
    next_capture: u64,
    /// Never behind `now` once settled.
    drain_at: Ts,
    captured: Vec<AudioFrame>,
    played: Vec<AudioFrame>,
    discarded: Vec<AudioFrame>,
    /// The host shim's ledger: a frame cut mid-play counts only its unplayed part.
    ledger: PlayoutLedger,
    speaker: Speaker,
}

struct Queued {
    frame: AudioFrame,
    ends_at: Ts,
    /// How many clears preceded its capture call.
    clears_before: usize,
}

struct Pending {
    frame: AudioFrame,
    capture: u64,
    called: Ts,
    clears_before: usize,
}

/// What reached the speaker, against the source's clears.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Speaker {
    pub clears: Vec<Ts>,
    pub heard: Vec<HeardFrame>,
}

/// A frame that played, entirely or until a clear cut it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeardFrame {
    /// How many of [`Speaker::clears`] preceded its capture call.
    pub clears_before: usize,
    pub until: Ts,
}

impl FakeAudioSource {
    pub(super) fn new(clock: Arc<dyn Clock>, opts: &TrackPublishOptions) -> Self {
        let now = clock.now();
        FakeAudioSource {
            clock,
            queue_size: Duration::from_millis(opts.queue_size_ms as u64),
            close_started: tokio::sync::watch::Sender::new(false),
            close_flow: tokio::sync::watch::channel(true).0,
            state: Mutex::new(SourceState { drain_at: now, ..SourceState::default() }),
            wake: Notify::new(),
        }
    }

    /// Stall SDK closure independently from publication unpublish and audio playback.
    pub fn flow_closes(&self, flowing: bool) {
        self.close_flow.send_replace(flowing);
    }

    pub async fn wait_for_close_started(&self) {
        let _ = self.close_started.subscribe().wait_for(|started| *started).await;
    }

    pub fn captured(&self) -> Vec<AudioFrame> {
        self.state.lock().unwrap().captured.clone()
    }

    pub fn played(&self) -> Vec<AudioFrame> {
        self.settle();
        self.state.lock().unwrap().played.clone()
    }

    /// `captured().len() == played().len() + discarded().len()` once drained.
    pub fn discarded(&self) -> Vec<AudioFrame> {
        self.settle();
        self.state.lock().unwrap().discarded.clone()
    }

    pub fn speaker(&self) -> Speaker {
        self.settle();
        self.state.lock().unwrap().speaker.clone()
    }

    pub fn is_closed(&self) -> bool {
        self.state.lock().unwrap().closed
    }

    pub fn settle(&self) {
        let now = self.clock.now();
        {
            let mut state = self.state.lock().unwrap();
            self.settle_locked(&mut state, now);
        }
        self.wake.notify_waiters();
    }

    fn settle_locked(&self, state: &mut SourceState, now: Ts) {
        while let Some(at) = self.admits_at(state)
            && at <= now
        {
            let piece = state.pending.pop_front().expect("admits_at() saw a piece");
            let ends_at = state.drain_at.max(at).saturating_add(frame_duration(&piece.frame));
            Self::buffer_locked(state, piece.frame, ends_at, piece.clears_before);
        }
        while state.queue.front().is_some_and(|q| q.ends_at <= now) {
            let q = state.queue.pop_front().expect("front() just matched");
            state
                .speaker
                .heard
                .push(HeardFrame { clears_before: q.clears_before, until: q.ends_at });
            state.played.push(q.frame);
        }
        // `drain_at` is deliberately not clamped to `now`: its readers handle a stale value.
    }

    /// When the next pending piece fits: once what is buffered is down to the room it needs.
    fn admits_at(&self, state: &SourceState) -> Option<Ts> {
        let piece = state.pending.front()?;
        let room = self.queue_size.saturating_sub(frame_duration(&piece.frame));
        Some(piece.called.max(state.drain_at.saturating_sub(room)))
    }

    fn buffer_locked(
        state: &mut SourceState,
        frame: AudioFrame,
        ends_at: Ts,
        clears_before: usize,
    ) {
        state.drain_at = ends_at;
        state.ledger.add_pushed(frame_duration(&frame));
        state.captured.push(frame.clone());
        state.queue.push_back(Queued { frame, ends_at, clears_before });
    }

    /// Discards what is buffered; a frame already playing is heard up to `now`.
    fn cut_locked(state: &mut SourceState, now: Ts) {
        let unplayed = Self::buffered_locked(state, now);
        state.ledger.add_discarded(unplayed);
        for q in state.queue.drain(..) {
            if q.ends_at.saturating_sub(frame_duration(&q.frame)) < now {
                state.speaker.heard.push(HeardFrame { clears_before: q.clears_before, until: now });
            }
            state.discarded.push(q.frame);
        }
        state.drain_at = now;
    }

    /// Wait for a wake or for the clock to reach `at`, whichever comes first.
    async fn parked_until(&self, parked: Pin<&mut Notified<'_>>, at: Option<Ts>) {
        match at {
            Some(at) => {
                tokio::select! {
                    () = parked => {}
                    () = self.clock.wait_until(at) => {}
                }
            }
            None => parked.await,
        }
    }

    fn buffered_locked(state: &SourceState, now: Ts) -> Duration {
        state.drain_at.saturating_duration_since(now)
    }

    /// What is buffered and the pieces still to come.
    fn queued_locked(state: &SourceState, now: Ts) -> Duration {
        state.pending.iter().map(|p| frame_duration(&p.frame)).sum::<Duration>()
            + Self::buffered_locked(state, now)
    }

    /// When the speaker next frees room or takes a piece.
    fn next_change(&self, state: &SourceState) -> Option<Ts> {
        let played = state.queue.front().map(|q| q.ends_at);
        played.into_iter().chain(self.admits_at(state)).min()
    }

    /// Hands a capture longer than the queue to the speaker in queue-sized pieces, then waits
    /// until the last is buffered.
    async fn capture_in_pieces(&self, frame: &AudioFrame) -> Result<()> {
        let sample_bytes = usize::from(frame.num_channels) * 2;
        let piece_samples =
            (u128::from(frame.sample_rate) * self.queue_size.as_millis() / 1000).max(1) as usize;
        let capture = {
            let now = self.clock.now();
            let mut state = self.state.lock().unwrap();
            if state.closed {
                return Err(RtcError::Closed("audio source"));
            }
            let capture = state.next_capture;
            state.next_capture += 1;
            let clears_before = state.speaker.clears.len();
            for bytes in frame.data.chunks(piece_samples * sample_bytes) {
                let piece = AudioFrame::new(
                    bytes.to_vec(),
                    frame.sample_rate,
                    frame.num_channels,
                    (bytes.len() / sample_bytes) as u32,
                );
                state.pending.push_back(Pending {
                    frame: piece,
                    capture,
                    called: now,
                    clears_before,
                });
            }
            capture
        };
        self.wake.notify_waiters();
        loop {
            let parked = self.wake.notified();
            tokio::pin!(parked);
            parked.as_mut().enable();
            let at = {
                let now = self.clock.now();
                let mut state = self.state.lock().unwrap();
                self.settle_locked(&mut state, now);
                if state.closed {
                    return Err(RtcError::Closed("audio source"));
                }
                if !state.pending.iter().any(|p| p.capture == capture) {
                    return Ok(());
                }
                self.next_change(&state)
            };
            self.parked_until(parked, at).await;
        }
    }
}

impl AudioSource for FakeAudioSource {
    fn capture<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        frame: &'a AudioFrame,
    ) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            if !self.queue_size.is_zero() && frame_duration(frame) > self.queue_size {
                return self.capture_in_pieces(frame).await;
            }
            let clears_before = self.state.lock().unwrap().speaker.clears.len();
            loop {
                let parked = self.wake.notified();
                tokio::pin!(parked);
                parked.as_mut().enable();
                let room = {
                    let now = self.clock.now();
                    let mut state = self.state.lock().unwrap();
                    if state.closed {
                        return Err(RtcError::Closed("audio source"));
                    }
                    self.settle_locked(&mut state, now);
                    let queued = Self::queued_locked(&state, now);
                    // An empty queue always accepts: nothing else can drain it, so the bound would
                    // deadlock. The SDK takes one capture at a time: none while pieces are to come.
                    if state.pending.is_empty()
                        && (state.queue.is_empty()
                            || queued + frame_duration(frame) <= self.queue_size)
                    {
                        let ends_at = state.drain_at.max(now).saturating_add(frame_duration(frame));
                        Self::buffer_locked(&mut state, frame.clone(), ends_at, clears_before);
                        drop(state);
                        self.wake.notify_waiters();
                        return Ok(());
                    }
                    self.next_change(&state)
                };
                // Room frees up as the head frame finishes playing, on the clock: a speaker drains
                // whether or not anyone settles it.
                self.parked_until(parked, room).await;
            }
        })
    }

    fn clear_queue(&self) -> Result<()> {
        let now = self.clock.now();
        {
            let mut state = self.state.lock().unwrap();
            self.settle_locked(&mut state, now);
            // Time-based, so a frame cut halfway counts only its unplayed half. The pieces still
            // to come stay: the SDK's clear does not reach its capture in flight.
            Self::cut_locked(&mut state, now);
            state.speaker.clears.push(now);
        }
        self.wake.notify_waiters();
        Ok(())
    }

    fn queued_seconds(&self) -> Result<f64> {
        let now = self.clock.now();
        let mut state = self.state.lock().unwrap();
        self.settle_locked(&mut state, now);
        Ok(Self::queued_locked(&state, now).as_secs_f64())
    }

    fn playback_info(&self) -> Result<PlaybackInfo> {
        let now = self.clock.now();
        let mut state = self.state.lock().unwrap();
        self.settle_locked(&mut state, now);
        Ok(state.ledger.snapshot(Self::buffered_locked(&state, now)))
    }

    fn wait_for_playout<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<'a, Result<PlaybackInfo>> {
        Box::pin(async move {
            loop {
                let parked = self.wake.notified();
                tokio::pin!(parked);
                parked.as_mut().enable();
                let drained = {
                    let now = self.clock.now();
                    let mut state = self.state.lock().unwrap();
                    self.settle_locked(&mut state, now);
                    if state.closed || Self::queued_locked(&state, now).is_zero() {
                        return Ok(state.ledger.settle());
                    }
                    self.next_change(&state).or(Some(state.drain_at))
                };
                self.parked_until(parked, drained).await;
            }
        })
    }

    fn close<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
    ) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            self.close_started.send_replace(true);
            let mut flow = self.close_flow.subscribe();
            let _ = flow.wait_for(|flowing| *flowing).await;
            // Closing discards the queue as a truncation, so `played + discarded` stays `captured`.
            let now = self.clock.now();
            {
                let mut state = self.state.lock().unwrap();
                self.settle_locked(&mut state, now);
                Self::cut_locked(&mut state, now);
                state.pending.clear();
                state.closed = true;
            }
            self.wake.notify_waiters();
            Ok(())
        })
    }
}
