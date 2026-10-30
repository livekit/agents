//! Bounded realtime ingress and event delivery with one reserved terminal observation.

use super::session::{Cmd, Shared};
use lk_agents_abi::errors::ApiError;
use lk_agents_abi::llm::realtime::{Event, RealtimeModelError};
use lk_agents_abi::{Clock, Ts};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::{mpsc, watch};

pub(super) const COMMAND_BACKLOG: usize = 128;
pub(crate) const EVENT_BACKLOG: usize = 256;
/// Queued audio bytes. Text and tool-argument events are bounded only by `EVENT_BACKLOG` times
/// the WebSocket message limit.
const AUDIO_BYTES: usize = 32 * 1024 * 1024;

#[derive(Clone)]
enum Ending {
    Open,
    Closed,
    Failed(Event),
}

#[derive(Clone)]
pub(crate) struct Delivery {
    events: mpsc::Sender<QueuedEvent>,
    retained: Arc<AtomicUsize>,
    terminal: watch::Sender<Ending>,
}

pub(crate) struct EventReceiver {
    events: mpsc::Receiver<QueuedEvent>,
    terminal: watch::Receiver<Ending>,
    ended: bool,
}

pub(crate) fn events() -> (Delivery, EventReceiver) {
    let (events, receiver) = mpsc::channel(EVENT_BACKLOG);
    let (terminal, failure) = watch::channel(Ending::Open);
    (
        Delivery { events, terminal, retained: Arc::new(AtomicUsize::new(0)) },
        EventReceiver { events: receiver, terminal: failure, ended: false },
    )
}

struct QueuedEvent {
    event: Option<Event>,
    bytes: usize,
    retained: Arc<AtomicUsize>,
}

impl QueuedEvent {
    fn into_event(mut self) -> Event {
        self.event.take().expect("queued event consumed once")
    }
}

impl Drop for QueuedEvent {
    fn drop(&mut self) {
        self.retained.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

impl Delivery {
    pub(crate) fn send(&self, event: Event) -> Result<(), ApiError> {
        let bytes = match &event {
            Event::AudioDelta(delta) => delta.frame.data.len(),
            _ => 0,
        };
        let mut result = Err(ApiError::new("realtime event stream ended").with_retryable(false));
        let mut event = Some(event);
        // Serialize admission with terminal publication so a producer cannot keep filling
        // freshly drained capacity and postpone the required completion indefinitely.
        self.terminal.send_if_modified(|terminal| {
            if !matches!(terminal, Ending::Open) {
                return false;
            }
            result = (|| {
                self.retained
                    .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |retained| {
                        retained.checked_add(bytes).filter(|total| *total <= AUDIO_BYTES)
                    })
                    .map_err(|_| {
                        ApiError::new("realtime event audio limit exceeded").with_retryable(false)
                    })?;
                let queued =
                    QueuedEvent { event: event.take(), bytes, retained: self.retained.clone() };
                self.events.try_send(queued).map_err(|_| {
                    ApiError::new("realtime event backlog exceeded").with_retryable(false)
                })
            })();
            false
        });
        result
    }

    pub(crate) fn fail(&self, now: Ts, label: &str, error: ApiError) {
        // Keep the first terminal failure even if teardown or a simultaneous overload also fails.
        self.terminal.send_if_modified(|terminal| {
            if !matches!(terminal, Ending::Open) {
                return false;
            }
            *terminal = Ending::Failed(Event::Error(RealtimeModelError {
                timestamp: now,
                label: label.to_owned(),
                message: error.message.clone(),
                recoverable: false,
                req_id: error.request_id.clone(),
                code: Some("realtime_backlog_exceeded".into()),
                provider_error: Some(error.clone()),
            }));
            true
        });
    }

    pub(crate) fn finish(&self) {
        self.terminal.send_if_modified(|terminal| {
            if !matches!(terminal, Ending::Open) {
                return false;
            }
            *terminal = Ending::Closed;
            true
        });
    }
}

impl EventReceiver {
    pub(crate) async fn next(&mut self) -> Option<Event> {
        loop {
            if self.ended {
                return None;
            }
            if let Ok(event) = self.events.try_recv() {
                return Some(event.into_event());
            }
            match self.terminal.borrow_and_update().clone() {
                Ending::Failed(event) => {
                    self.ended = true;
                    self.events.close();
                    return Some(event);
                }
                Ending::Closed => {
                    self.ended = true;
                    self.events.close();
                    return None;
                }
                Ending::Open => {}
            }
            tokio::select! {
                biased;
                changed = self.terminal.changed() => {
                    if changed.is_err() {
                        return self.events.recv().await.map(QueuedEvent::into_event);
                    }
                }
                event = self.events.recv() => return event.map(QueuedEvent::into_event),
            }
        }
    }
}

pub(super) struct Ingress {
    pub(super) commands: mpsc::Sender<Cmd>,
    pub(super) shared: Arc<Shared>,
    pub(super) events: Delivery,
    pub(super) clock: Arc<dyn Clock>,
    pub(super) label: String,
}

impl Ingress {
    pub(super) fn send(&self, command: Cmd) {
        if let Err(mpsc::error::TrySendError::Full(command)) = self.commands.try_send(command) {
            let mut error =
                ApiError::new("realtime command backlog exceeded").with_retryable(false);
            if let Cmd::Core(command) = command {
                error.request_id = command.req_id().map(str::to_owned);
            }
            self.events.fail(self.clock.now(), &self.label, error);
            self.shared.closing.notify_one();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_saturated_event_queue_retains_one_terminal_failure_and_then_ends() {
        let (delivery, mut receiver) = events();
        for _ in 0..EVENT_BACKLOG {
            delivery
                .send(Event::InputSpeechStarted(
                    lk_agents_abi::llm::realtime::InputSpeechStarted {},
                ))
                .unwrap();
        }
        let error = delivery
            .send(Event::InputSpeechStarted(lk_agents_abi::llm::realtime::InputSpeechStarted {}))
            .unwrap_err();
        delivery.fail(Ts::ZERO, "model", error);
        delivery.finish();
        for _ in 0..EVENT_BACKLOG {
            assert!(matches!(receiver.next().await, Some(Event::InputSpeechStarted(_))));
        }
        assert!(matches!(receiver.next().await, Some(Event::Error(error))
            if !error.recoverable && error.provider_error.is_some()));
        assert!(receiver.next().await.is_none());
    }

    #[tokio::test]
    async fn producers_cannot_delay_terminal_delivery_after_overload() {
        let (delivery, mut receiver) = events();
        delivery
            .send(Event::InputSpeechStarted(lk_agents_abi::llm::realtime::InputSpeechStarted {}))
            .unwrap();
        delivery.fail(Ts::ZERO, "model", ApiError::new("overloaded"));
        assert!(matches!(receiver.next().await, Some(Event::InputSpeechStarted(_))));
        assert!(
            delivery
                .send(Event::InputSpeechStarted(
                    lk_agents_abi::llm::realtime::InputSpeechStarted {},
                ))
                .is_err(),
            "a producer admitted more data after the terminal outcome"
        );
        assert!(matches!(receiver.next().await, Some(Event::Error(_))));
        assert!(receiver.next().await.is_none());
    }

    fn audio(bytes: usize) -> Event {
        Event::AudioDelta(lk_agents_abi::llm::realtime::AudioDelta {
            message_id: "message".into(),
            response_id: None,
            frame: lk_agents_abi::AudioFrame::new(vec![0u8; bytes], 24_000, 1, bytes as u32 / 2),
        })
    }

    #[tokio::test]
    async fn event_backlog_bounds_queued_audio_and_releases_read_capacity() {
        let (delivery, mut receiver) = events();
        assert!(
            delivery.send(audio(33 * 1024 * 1024)).is_err(),
            "one frame exceeded the audio budget"
        );
        delivery.send(audio(12 * 1024 * 1024)).unwrap();
        delivery.send(audio(12 * 1024 * 1024)).unwrap();
        assert!(
            delivery.send(audio(12 * 1024 * 1024)).is_err(),
            "several individually admissible events exceeded the shared byte budget"
        );
        assert!(matches!(receiver.next().await, Some(Event::AudioDelta(_))));
        delivery.send(audio(12 * 1024 * 1024)).unwrap();
        delivery.finish();
        assert!(receiver.next().await.is_some());
        assert!(receiver.next().await.is_some());
        assert!(receiver.next().await.is_none());
    }

    #[tokio::test]
    async fn a_clean_pump_end_closes_delivery_while_ingress_still_exists() {
        let (delivery, mut receiver) = events();
        let ingress_delivery = delivery.clone();
        delivery.finish();
        assert!(receiver.next().await.is_none());
        drop(ingress_delivery);
    }
}
