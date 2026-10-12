//! The realtime model: one socket, and a server that answers every request the same way.
//!
//! The turn itself is scripted: a realtime turn is the server's to run and the harness has no id
//! to name one by before it exists, so the answer is produced here rather than delivered by the
//! case. The script is fixed, so a seed still reproduces. What the case does control is the
//! socket: the server-side events that are nobody's answer -- a drop, a reconnect, a stray delta,
//! a second generation -- arrive as `RealtimeEvent` on the registered socket.
//!
//! The generation is text only: audio would reach the sink untagged, and `audio_conservation`
//! is right to call a sample nobody produced a finding.
use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicU64, Ordering},
    },
};

use lk_agents_abi::{
    ChatContext, Realtime, RealtimeCommand, RealtimeSession,
    errors::ProviderError,
    llm::realtime::{
        Event, GenerationCreated, GenerationDone, InputTranscription, MessageDone, MessageStarted,
        RealtimeCapabilities, TextDelta,
    },
};
use tokio::sync::mpsc;

use super::{Control, Registry, Slot};
use crate::{
    input::{Input, RtEvent},
    record::{Detail, Kind},
};

/// What the fake server can do. Text output and a mutable session, so the driver sends the
/// opening frame and every later update over the one socket rather than skipping them.
fn capabilities(server_turns: bool) -> RealtimeCapabilities {
    RealtimeCapabilities {
        message_truncation: true,
        turn_detection: server_turns,
        user_transcription: true,
        auto_tool_reply_generation: false,
        audio_output: false,
        manual_function_calls: false,
        can_disable_turn_detection: true,
        auto_turn_reply_generation: false,
        mutable_chat_context: true,
        mutable_instructions: true,
        mutable_tools: true,
        per_response_tool_choice: true,
        supports_say: true,
    }
}

pub struct FakeRealtime {
    registry: Arc<Registry>,
    /// Whether the server detects turns itself, which is what decides whether the driver asks it
    /// to stop doing so.
    server_turns: bool,
}

impl FakeRealtime {
    pub fn new(registry: Arc<Registry>, server_turns: bool) -> Arc<Self> {
        Arc::new(FakeRealtime { registry, server_turns })
    }
}

#[async_trait::async_trait]
impl Realtime for FakeRealtime {
    fn label(&self) -> String {
        "fake::realtime".into()
    }

    fn capabilities(&self) -> RealtimeCapabilities {
        capabilities(self.server_turns)
    }

    async fn session(&self, _disabled: bool) -> Result<Arc<dyn RealtimeSession>, ProviderError> {
        let (sender, events) = mpsc::unbounded_channel();
        let session = Arc::new(Session {
            sender: Mutex::new(Some(sender)),
            events: tokio::sync::Mutex::new(events),
            next: AtomicU64::new(1),
            server_turns: self.server_turns,
            slot: OnceLock::new(),
            registry: self.registry.clone(),
            closed: tokio::sync::Notify::new(),
            open: Mutex::new(VecDeque::new()),
        });
        let slot = self.registry.register(
            Kind::Realtime,
            Detail::None,
            Control::Realtime(session.clone()),
        );
        let _ = session.slot.set(slot);
        Ok(session)
    }
}

/// One server-side event onto a registered socket. True when the driver still reads it.
pub(super) fn deliver(socket: &Session, what: &Input) -> bool {
    let Input::RealtimeEvent { event, .. } = what else { return false };
    let Some(sender) = socket.sender.lock().unwrap().clone() else { return false };
    if let RtEvent::Disconnected = event {
        // The server dropped the socket: the disconnect is the last thing the stream carries, and
        // then the stream ends -- which is the one way the driver sees a stream end while it is
        // still running, rather than behind its own close.
        let sent =
            sender.send(Event::Disconnected { reason: "injected socket drop".into() }).is_ok();
        socket.sender.lock().unwrap().take();
        for (slot, _) in socket.open.lock().unwrap().drain(..) {
            socket.registry.mark_closed(slot);
        }
        socket.closed.notify_one();
        return sent;
    }
    if let RtEvent::GenerationDone = event {
        // The oldest turn still open finishes; with none open there is nothing to finish.
        let Some((slot, done)) = socket.open.lock().unwrap().pop_front() else { return false };
        let sent = sender.send(Event::GenerationDone(done)).is_ok();
        socket.registry.mark_closed(slot);
        return sent;
    }
    static EXTRA: AtomicU64 = AtomicU64::new(1);
    let nth = EXTRA.fetch_add(1, Ordering::Relaxed);
    let event = match event {
        RtEvent::GenerationDone => unreachable!("answered above"),
        RtEvent::Disconnected => unreachable!("answered above"),
        RtEvent::Reconnected => Event::SessionReconnected(lk_agents_abi::llm::realtime::SessionReconnected {}),
        // A delta for a generation the driver never saw created.
        RtEvent::StrayDelta => Event::TextDelta(TextDelta {
            message_id: format!("rt-stray-msg-{nth}"),
            response_id: Some(format!("rt-stray-{nth}")),
            delta: "...".into(),
            start_time: None,
        }),
        // The server opens a generation of its own, over whatever is in progress.
        RtEvent::SecondGeneration => Event::GenerationCreated(GenerationCreated {
            response_id: format!("rt-extra-{nth}"),
            req_id: None,
        }),
    };
    sender.send(event).is_ok()
}

/// One socket: commands in, the scripted answer straight back out -- all of it but the end,
/// which is the case's to say.
pub struct Session {
    /// Dropped by `aclose`, which is what ends the event stream.
    sender: Mutex<Option<mpsc::UnboundedSender<Event>>>,
    events: tokio::sync::Mutex<mpsc::UnboundedReceiver<Event>>,
    next: AtomicU64,
    server_turns: bool,
    slot: OnceLock<Slot>,
    registry: Arc<Registry>,
    /// Set by `aclose`; a permit, so a `next_event` that comes after the close sees it too.
    closed: tokio::sync::Notify,
    /// Turns the server has started and not finished, oldest first, each with the end it will
    /// report when the case says so. A turn is a resource the case can see and must end.
    open: Mutex<VecDeque<(Slot, GenerationDone)>>,
}

impl Session {
    /// A turn, answered the instant it was asked for: created, one message, its words -- and
    /// then open, speaking, until the case ends it with `GenerationDone`.
    fn answer(&self, req_id: Option<String>, text: String) {
        let Some(sender) = self.sender.lock().unwrap().clone() else { return };
        let nth = self.next.fetch_add(1, Ordering::Relaxed);
        let (response_id, message_id) = (format!("rt-{nth}"), format!("rt-msg-{nth}"));
        let of = |id: &str| Some(id.to_owned());
        let events = [
            Event::GenerationCreated(GenerationCreated {
                response_id: response_id.clone(),
                req_id,
            }),
            Event::MessageStarted(MessageStarted {
                message_id: message_id.clone(),
                response_id: of(&response_id),
            }),
            Event::TextDelta(TextDelta {
                message_id: message_id.clone(),
                response_id: of(&response_id),
                delta: text,
                start_time: None,
            }),
            Event::MessageDone(MessageDone { message_id, response_id: of(&response_id) }),
        ];
        for event in events {
            let _ = sender.send(event);
        }
        let done = GenerationDone {
            response_id,
            status: lk_agents_abi::llm::realtime::GenerationStatus::Completed,
            usage: Default::default(),
        };
        let slot = self.registry.register(Kind::Generation, Detail::None, Control::Generation);
        self.open.lock().unwrap().push_back((slot, done));
    }

    /// The user's turn, as the server heard it.
    fn transcribe(&self) {
        let Some(sender) = self.sender.lock().unwrap().clone() else { return };
        let nth = self.next.fetch_add(1, Ordering::Relaxed);
        let _ = sender.send(Event::InputTranscription(InputTranscription {
            item_id: format!("rt-user-{nth}"),
            transcript: "yeah hi um I'm calling about my order".into(),
            is_final: true,
            confidence: None,
            turn_started_at: None,
        }));
    }
}

#[async_trait::async_trait]
impl RealtimeSession for Session {
    fn capabilities(&self) -> RealtimeCapabilities {
        capabilities(self.server_turns)
    }

    fn send(&self, cmd: RealtimeCommand) {
        match cmd {
            RealtimeCommand::GenerateReply(c) => {
                self.answer(Some(c.req_id), "Sure, I can help with that.".into())
            }
            RealtimeCommand::Say(c) => self.answer(Some(c.req_id), c.text),
            // The turn was ended by hand, so the server reports what it heard -- the capabilities
            // say this session transcribes the user, and a server that says so and never does is
            // a fake the driver cannot be tested against.
            RealtimeCommand::CommitAudio(_) => self.transcribe(),
            // Everything else the server takes silently: the driver's own journal is the record
            // that it was sent, and inventing a reply to a `Truncate` would be inventing a server.
            _ => {}
        }
    }

    async fn next_event(&self) -> Option<Event> {
        let mut events = self.events.lock().await;
        tokio::select! {
            biased;
            _ = self.closed.notified() => None,
            event = events.recv() => event,
        }
    }

    async fn chat_ctx(&self) -> ChatContext {
        ChatContext::default()
    }

    async fn aclose(&self) {
        self.sender.lock().unwrap().take();
        for (slot, _) in self.open.lock().unwrap().drain(..) {
            self.registry.mark_closed(slot);
        }
        // The case holds a sender of its own, so dropping ours would not end the stream, and the
        // driver's pump holds the receiver across its `recv`, so it cannot be closed from here.
        // The socket says it is closed instead, and `next_event` answers the end on that.
        self.closed.notify_one();
        if let Some(slot) = self.slot.get() {
            self.registry.mark_closed(*slot);
        }
    }
}
