//! Realtime ingress over a media queue, and bounded event delivery with one reserved terminal observation.

use super::session::Cmd;
use lk_agents_abi::RealtimeCommand as Command;
use lk_agents_abi::errors::ApiError;
use lk_agents_abi::llm::realtime::{
    Event, GenerateReply, GenerateReplyFailed, PushAudio, RealtimeError, RealtimeModelError, Say,
};
use lk_agents_abi::{Clock, Ts};
use lk_agents_media::media_queue;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::{mpsc, watch};

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
    /// Audio and every command, in one ordered queue: a commit, clear or reply is never
    /// reordered against the audio before it.
    pub(super) input: media_queue::AudioSender<Cmd>,
    pub(super) drops: lk_agents_net::audio_drops::AudioDrops,
    pub(super) events: Delivery,
    pub(super) clock: Arc<dyn Clock>,
    pub(super) label: String,
}

impl Ingress {
    pub(super) fn send(&self, command: Cmd) {
        if let Cmd::Core(Command::PushAudio(PushAudio { frame })) = command {
            return self.drops.record(self.input.push(frame));
        }
        let reply = match &command {
            Cmd::Core(Command::GenerateReply(GenerateReply { req_id, .. }))
            | Cmd::Core(Command::Say(Say { req_id, .. })) => Some(req_id.clone()),
            _ => None,
        };
        let sent = match command {
            Cmd::Core(Command::ClearAudio(clear)) => {
                self.input.flush(Cmd::Core(Command::ClearAudio(clear)))
            }
            command => self.input.event(command),
        };
        if let Err(stuck) = sent {
            self.stuck(stuck, reply);
        }
    }

    /// Pre-connect audio, kept in full.
    pub(super) fn head(&self, frame: lk_agents_abi::AudioFrame) {
        self.input.head(frame);
    }

    /// The pump stopped draining: the command is lost, reported as a recoverable error, and a
    /// reply it asked for fails instead of waiting forever.
    fn stuck(&self, stuck: media_queue::Stuck, reply: Option<String>) {
        let now = self.clock.now();
        let error = ApiError::new(stuck.to_string()).with_retryable(true);
        let mut events = vec![Event::Error(RealtimeModelError {
            timestamp: now,
            label: self.label.clone(),
            message: error.message.clone(),
            recoverable: true,
            req_id: reply.clone(),
            code: None,
            provider_error: Some(error),
        })];
        events.extend(reply.map(|req_id| {
            Event::GenerateReplyFailed(GenerateReplyFailed {
                req_id,
                error: RealtimeError::new("realtime input stopped draining"),
            })
        }));
        for event in events {
            if let Err(error) = self.events.send(event) {
                self.events.fail(now, &self.label, error);
            }
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

    // Catches: pre-connect audio pushed as live audio, so the input policy drops the start of the
    // first turn before the socket is up.
    #[tokio::test]
    async fn ingress_keeps_pre_connect_audio_in_full() {
        let policy = media_queue::Policy::CatchUp(std::time::Duration::from_millis(100));
        let (input, mut rx) = media_queue::audio(policy);
        let (events, _receiver) = events();
        let ingress = Ingress {
            input,
            drops: lk_agents_net::audio_drops::AudioDrops::new(
                lk_agents_net::telemetry::AudioConsumer::Realtime,
                policy,
            ),
            events,
            clock: Arc::new(lk_agents_net::clock::RealClock::default()),
            label: "model".into(),
        };
        let frame = || lk_agents_abi::AudioFrame::new(vec![0u8; 480], 24_000, 1, 240);
        for _ in 0..30 {
            ingress.head(frame());
        }
        drop(ingress);
        let mut samples = 0;
        while let Some(media_queue::Item::Media(frame)) = rx.recv().await {
            samples += frame.samples_per_channel;
        }
        assert_eq!(samples, 30 * 240, "the head was cut by the input policy");
        assert_eq!(rx.head().received, std::time::Duration::from_millis(300));
        assert_eq!(rx.dropped(), 0);
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
