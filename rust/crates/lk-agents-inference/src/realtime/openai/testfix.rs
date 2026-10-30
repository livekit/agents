//! Realtime feature fixtures exercise the core through the runtime encoder and decoder.

use super::wire::{IdGen, Incoming, WireShape};
use crate::realtime::machine::{RealtimeMachine as Core, Step as CoreStep};
use crate::realtime::{Dialect, OptionsPatch, SessionOptions};
use lk_agents_abi::llm::provider_format::ImageResolver;
use lk_agents_abi::llm::realtime::{Event, RealtimeCapabilities, RealtimeModelError};
use lk_agents_abi::{ChatItem, ChatMessage, Role, Ts};
use std::sync::Arc;
use std::time::SystemTime;

pub const TIME: SystemTime = SystemTime::UNIX_EPOCH;

pub fn msg(id: &str, text: &str) -> ChatItem {
    ChatItem::Message(ChatMessage::text(id, Role::User, text, TIME))
}

pub fn machine(dialect: Dialect) -> RealtimeMachine {
    let opts = SessionOptions::from_dialect(&dialect, false);
    let caps = dialect.capabilities(opts.turn_detection.as_ref(), false, true, true);
    RealtimeMachine::new(
        dialect,
        opts,
        caps,
        Arc::new(super::wire::SeqIds::default()),
        Arc::new(crate::testkit::NoImages),
    )
}

pub struct RealtimeMachine {
    core: Core,
    encoder: super::writer::Encoder,
    images: Arc<dyn ImageResolver + Send + Sync>,
}

#[derive(Debug)]
pub struct Step {
    pub frames: Vec<serde_json::Value>,
    pub events: Vec<Event>,
    pub recycle: bool,
    pub fatal: Option<RealtimeModelError>,
}

impl Step {
    pub fn merge(&mut self, other: Step) {
        self.frames.extend(other.frames);
        self.events.extend(other.events);
        self.recycle |= other.recycle;
        self.fatal = self.fatal.take().or(other.fatal);
    }
}

impl std::ops::Deref for RealtimeMachine {
    type Target = Core;
    fn deref(&self) -> &Core {
        &self.core
    }
}

impl RealtimeMachine {
    pub fn new(
        dialect: Dialect,
        opts: SessionOptions,
        caps: RealtimeCapabilities,
        _ids: Arc<dyn IdGen>,
        images: Arc<dyn ImageResolver + Send + Sync>,
    ) -> Self {
        Self {
            core: Core::new(dialect, opts, caps, String::new()),
            encoder: super::writer::Encoder::new(),
            images,
        }
    }
    fn render(&mut self, now: Ts, mut step: CoreStep) -> Step {
        loop {
            while let Some(retirement) = self.core.take_context_retirement() {
                drop(retirement);
            }
            if !self.core.context_work_pending() {
                break;
            }
            let more = self.core.continue_context(now);
            step.frames.extend(more.frames);
            step.events.extend(more.events);
            step.recycle |= more.recycle;
            step.fatal = step.fatal.or(more.fatal);
        }
        let frames = self
            .encoder
            .encode(step.frames, self.images.as_ref(), &WireShape::default())
            .expect("valid fixture effects")
            .into_iter()
            .map(|frame| {
                if let Some((req_id, is_say)) = frame.response {
                    self.core.response_sent(
                        self.core.connection_generation(),
                        now,
                        &req_id,
                        is_say,
                    );
                }
                serde_json::from_str(&frame.text).expect("encoded JSON")
            })
            .collect();
        Step { frames, events: step.events, recycle: step.recycle, fatal: step.fatal }
    }
    pub fn open(&mut self, now: Ts) -> Step {
        let step = self.core.open(now);
        self.render(now, step)
    }
    pub fn reconnect(&mut self, now: Ts) -> Step {
        self.encoder = super::writer::Encoder::new();
        let step = self.core.reconnect(now);
        self.render(now, step)
    }
    pub fn command(&mut self, now: Ts, cmd: lk_agents_abi::RealtimeCommand) -> Step {
        let step = self.core.command(now, cmd);
        self.render(now, step)
    }
    pub fn incoming(&mut self, now: Ts, input: Incoming) -> Step {
        let step = self.core.incoming_on(self.core.connection_generation(), now, input);
        self.render(now, step)
    }
    pub fn update_options(&mut self, patch: OptionsPatch) -> Step {
        let step = self.core.update_options(patch);
        self.render(Ts::ZERO, step)
    }
    pub fn tick(&mut self, now: Ts) -> Step {
        let step = self.core.tick(now);
        self.render(now, step)
    }
    pub fn close(&mut self, now: Ts) -> Step {
        let step = self.core.close(now);
        self.render(now, step)
    }
    pub fn with_provider(mut self, provider: String) -> Self {
        self.core = self.core.with_provider(provider);
        self
    }
}
