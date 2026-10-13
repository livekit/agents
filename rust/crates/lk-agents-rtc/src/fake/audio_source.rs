//! `rtc.AudioSource` with playout modelled on the injected clock, and a synthesiser of speech frames.

use crate::playout::PlayoutLedger;
use crate::traits::AudioSource;
use crate::{PlaybackInfo, TrackPublishOptions, frame_duration};
use crate::{Result, RtcError};
use async_trait::async_trait;
use lk_agents_abi::AudioFrame;
use lk_agents_abi::{Clock, Ts};
use std::collections::VecDeque;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::Notify;

/// `rtc.AudioSource`, minus the speaker.
pub struct FakeAudioSource {
    clock: Arc<dyn Clock>,
    queue_size: Duration,
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
            state: Mutex::new(SourceState { drain_at: now, ..SourceState::default() }),
            wake: Notify::new(),
        }
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

    fn queued_locked(state: &SourceState, now: Ts) -> Duration {
        state.drain_at.saturating_duration_since(now)
    }
}

#[async_trait]
impl AudioSource for FakeAudioSource {
    async fn capture(&self, frame: &AudioFrame) -> Result<()> {
        loop {
            let parked = self.wake.notified();
            tokio::pin!(parked);
            parked.as_mut().enable();
            {
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
            }
            parked.await;
        }
    }

    fn clear_queue(&self) {
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
    }

    fn queued_duration(&self) -> Duration {
        let now = self.clock.now();
        let mut state = self.state.lock().unwrap();
        Self::settle_locked(&mut state, now);
        Self::queued_locked(&state, now)
    }

    fn playback_info(&self) -> PlaybackInfo {
        let now = self.clock.now();
        let mut state = self.state.lock().unwrap();
        Self::settle_locked(&mut state, now);
        state.ledger.snapshot(Self::queued_locked(&state, now))
    }

    async fn wait_for_playout(&self) -> PlaybackInfo {
        loop {
            let parked = self.wake.notified();
            tokio::pin!(parked);
            parked.as_mut().enable();
            {
                let now = self.clock.now();
                let mut state = self.state.lock().unwrap();
                Self::settle_locked(&mut state, now);
                if state.closed || Self::queued_locked(&state, now).is_zero() {
                    return state.ledger.settle();
                }
            }
            parked.await;
        }
    }

    async fn close(&self) -> Result<()> {
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
    }
}

/// Speech frames have distinct samples frame to frame, so dropped or duplicated audio is visible.
pub struct FrameSynth {
    sample_rate: u32,
    samples: u32,
    next: std::sync::atomic::AtomicUsize,
}

impl FrameSynth {
    /// A rate that does not divide into whole frames truncates rather than panics.
    pub fn new(sample_rate: u32, frame_ms: u32) -> Self {
        FrameSynth {
            sample_rate,
            samples: sample_rate / (1_000 / frame_ms.max(1)).max(1),
            next: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    pub fn frame(&self, silent: bool) -> AudioFrame {
        if silent {
            return crate::silence_frame(self.samples, self.sample_rate, 1);
        }
        let base = self.next.fetch_add(1, Ordering::SeqCst) as i16;
        let data: Vec<u8> = (0..self.samples)
            // `| 1` keeps every sample non-zero, so no frame can be mistaken for silence.
            .flat_map(|i| (base.wrapping_add(i as i16).wrapping_mul(31) | 1).to_le_bytes())
            .collect();
        AudioFrame::new(data, self.sample_rate, 1, self.samples)
    }
}
