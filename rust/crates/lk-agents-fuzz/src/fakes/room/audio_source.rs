//! `rtc.AudioSource` with playout modelled on the injected clock, and a synthesiser of speech frames.

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
    /// Never behind `now` once settled.
    drain_at: Ts,
    captured: Vec<AudioFrame>,
    played: Vec<AudioFrame>,
    discarded: Vec<AudioFrame>,
    /// The same [`PlayoutLedger`] the host shim keeps. `played` is derived (`captured - discarded`),
    /// so a frame cut mid-play counts only its unplayed part.
    ledger: PlayoutLedger,
}

struct Queued {
    frame: AudioFrame,
    ends_at: Ts,
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

    pub fn is_closed(&self) -> bool {
        self.state.lock().unwrap().closed
    }

    pub fn settle(&self) {
        let now = self.clock.now();
        {
            let mut state = self.state.lock().unwrap();
            Self::settle_locked(&mut state, now);
        }
        self.wake.notify_waiters();
    }

    fn settle_locked(state: &mut SourceState, now: Ts) {
        while state.queue.front().is_some_and(|q| q.ends_at <= now) {
            let q = state.queue.pop_front().expect("front() just matched");
            state.played.push(q.frame);
        }
        // `drain_at` is deliberately not clamped to `now`: its readers handle a stale value.
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

    fn queued_locked(state: &SourceState, now: Ts) -> Duration {
        state.drain_at.saturating_duration_since(now)
    }
}

impl AudioSource for FakeAudioSource {
    fn capture<'a>(&'a self, frame: &'a AudioFrame) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move {
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
                    Self::settle_locked(&mut state, now);
                    let queued = Self::queued_locked(&state, now);
                    // An empty queue always accepts: nothing else can drain it, so the bound would deadlock.
                    if state.queue.is_empty() || queued + frame_duration(frame) <= self.queue_size {
                        let base = state.drain_at.max(now);
                        let ends_at = base.saturating_add(frame_duration(frame));
                        state.drain_at = ends_at;
                        state.queue.push_back(Queued { frame: frame.clone(), ends_at });
                        state.captured.push(frame.clone());
                        state.ledger.add_pushed(frame_duration(frame));
                        drop(state);
                        self.wake.notify_waiters();
                        return Ok(());
                    }
                    state.queue.front().map(|q| q.ends_at)
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
            Self::settle_locked(&mut state, now);
            // Time-based, so a frame cut halfway counts only its unplayed half.
            let unplayed = Self::queued_locked(&state, now);
            state.ledger.add_discarded(unplayed);
            let cut: Vec<AudioFrame> = state.queue.drain(..).map(|q| q.frame).collect();
            state.discarded.extend(cut);
            state.drain_at = now;
        }
        self.wake.notify_waiters();
        Ok(())
    }

    fn queued_duration(&self) -> Result<Duration> {
        let now = self.clock.now();
        let mut state = self.state.lock().unwrap();
        Self::settle_locked(&mut state, now);
        Ok(Self::queued_locked(&state, now))
    }

    fn playback_info(&self) -> Result<PlaybackInfo> {
        let now = self.clock.now();
        let mut state = self.state.lock().unwrap();
        Self::settle_locked(&mut state, now);
        Ok(state.ledger.snapshot(Self::queued_locked(&state, now)))
    }

    fn wait_for_playout<'a>(&'a self) -> lk_agents_abi::BoxFuture<'a, Result<PlaybackInfo>> {
        Box::pin(async move {
            loop {
                let parked = self.wake.notified();
                tokio::pin!(parked);
                parked.as_mut().enable();
                let drained = {
                    let now = self.clock.now();
                    let mut state = self.state.lock().unwrap();
                    Self::settle_locked(&mut state, now);
                    if state.closed || Self::queued_locked(&state, now).is_zero() {
                        return Ok(state.ledger.settle());
                    }
                    state.drain_at
                };
                self.parked_until(parked, Some(drained)).await;
            }
        })
    }

    fn close<'a>(&'a self) -> lk_agents_abi::BoxFuture<'a, Result<()>> {
        Box::pin(async move {
            self.close_started.send_replace(true);
            let mut flow = self.close_flow.subscribe();
            let _ = flow.wait_for(|flowing| *flowing).await;
            // Closing discards the queue as a truncation, so `played + discarded` stays `captured`.
            let now = self.clock.now();
            {
                let mut state = self.state.lock().unwrap();
                Self::settle_locked(&mut state, now);
                let unplayed = Self::queued_locked(&state, now);
                state.ledger.add_discarded(unplayed);
                let cut: Vec<AudioFrame> = state.queue.drain(..).map(|q| q.frame).collect();
                state.discarded.extend(cut);
                state.drain_at = now;
                state.closed = true;
            }
            self.wake.notify_waiters();
            Ok(())
        })
    }
}
