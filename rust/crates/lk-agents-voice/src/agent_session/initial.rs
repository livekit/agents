//! The session and opening activity's synchronous initialization.

use super::commands::Origin;
use super::{Driver, NextId, Now, SessionParts, SessionView, VoiceSession};
use crate::agent_activity::false_interruption::PausedSpeech;
use crate::agent_activity::{AgentActivity, DriverConfig};
use crate::audio::AudioChannel;
use crate::audio_recognition::TurnDetectionMode;
use crate::generation::{ExpressiveLayers, Instructions};
use crate::trace::{CoreStep, RealtimeStep, Step};
use lk_agents_abi::resources::{RealtimeRef, ResourceId};
use lk_agents_abi::{TimerSet, Ts};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

/// What the core is built from: its options, the providers by id, the opening agent and the
/// sinks the microphone feeds.
pub struct InitialSession {
    pub config: DriverConfig,
    pub parts: SessionParts,
    pub instructions: Instructions,
    pub dynamic_instructions: Option<lk_agents_abi::resources::ResourceId>,
    pub output_nodes: super::nodes::OutputNodes,
    pub toolset: crate::tool_executor::Toolset,
    pub hook_bound: Option<Duration>,
    pub expressive: ExpressiveLayers,
    pub sinks: BTreeMap<AudioChannel, ResourceId>,
    pub video_instance: Option<lk_agents_abi::resources::VideoInputRef>,
    pub video_sampler: Option<crate::agent_activity::video_sampler::VideoSampler>,
    pub eot_thresholds: Option<lk_agents_abi::tokenize::language::ThresholdOptions>,
    pub eot_provider: Option<lk_agents_abi::ProviderId>,
    pub rt_session: Option<RealtimeRef>,
    pub rt_caps: lk_agents_abi::llm::realtime::RealtimeCapabilities,
    pub rt_disabled: bool,
    pub rt_resolved: TurnDetectionMode,
    pub view: Arc<SessionView>,
    pub next_id: NextId,
    pub now: Now,
    pub built_at: f64,
    pub journalled: bool,
    pub forward_chat_context: bool,
    pub keyterms: Option<(crate::keyterm_detection::KeytermSetup, Option<ResourceId>)>,
}

impl Driver {
    pub fn new(mut initial: InitialSession) -> Self {
        let (sink, audio_enabled) = initial.parts.outputs.pause_facts();
        let paused_speech = PausedSpeech::new(&initial.config.effective, sink, audio_enabled);
        let realtime_disabled = initial.rt_session.as_ref().map(|_| initial.rt_disabled);
        let activity = AgentActivity::from_initial(&mut initial, paused_speech);
        let mut session = VoiceSession::new(initial.config.clone(), initial.parts, initial.next_id);
        session.expressive = initial.expressive;
        let input = crate::agent_activity::io::AgentInput::new(
            initial.sinks,
            initial.video_instance,
            initial.video_sampler,
        );
        let mut outbox = Vec::new();
        if initial.journalled {
            outbox.push(super::effects::Effect::Journal(Box::new(Step::Core(
                CoreStep::ConfigApplied {
                    layer: crate::agent_activity::options::ConfigLayer::Session,
                    effective: Box::new(initial.config.effective),
                },
            ))));
            let since_zero = initial.now.at.saturating_duration_since(Ts::ZERO);
            outbox.push(super::effects::Effect::Journal(Box::new(Step::Core(
                CoreStep::WallEpoch {
                    at: initial.now.wall.checked_sub(since_zero).unwrap_or(std::time::UNIX_EPOCH),
                },
            ))));
            if let Some(disabled) = realtime_disabled {
                outbox.push(super::effects::Effect::Journal(Box::new(Step::Realtime(
                    RealtimeStep::Opened { turn_detection_disabled: disabled },
                ))));
            }
        }
        let mut driver = Driver {
            activity_changed: Default::default(),
            outbox,
            view: initial.view,
            session,
            activity,
            input,
            timers: TimerSet::new(),
            journalled: initial.journalled,
            command_origin: Origin::Host,
            step: super::time::StepClock::new(),
            stopped: false,
            opened: false,
        };
        driver.session.trace.built = initial.built_at;
        if let Some(stream) = driver.input.live_stt {
            driver.stt_stream_opened(stream);
        }
        driver.session.providers = driver.providers_held();
        driver.step.set(initial.now);
        driver.publish_activity();
        driver.flush_activity();
        driver.open_session(initial.forward_chat_context, initial.keyterms);
        driver
    }
}

impl Driver {
    /// The opening cycles: the first facts, in order, the gate, then the opening agent's records
    /// (the handoff from nobody, the starting config), and the opening agent on the realtime
    /// socket; then the static keyterms.
    fn open_session(
        &mut self,
        forward_chat_context: bool,
        keyterms: Option<(
            crate::keyterm_detection::KeytermSetup,
            Option<lk_agents_abi::resources::ResourceId>,
        )>,
    ) {
        self.publish_gate();
        {
            let mut out = self.out();
            self.record_handoff(None, &mut out);
            if self.activity.realtime().is_some() {
                let opening = self.session.chat_ctx.for_realtime();
                self.push_activity(Some(opening), &mut out);
            }
            let effects = self.settle_step(out);
            self.outbox.extend(effects);
        }
        self.session.forward_chat_context = forward_chat_context;
        // `KeytermDetector.start`: the static terms reach the recogniser now.
        if let Some((setup, llm)) = keyterms {
            let mut out = self.out();
            self.keyterms_start(setup, llm, &mut out);
            let effects = self.settle_step(out);
            self.outbox.extend(effects);
        }
    }
}
