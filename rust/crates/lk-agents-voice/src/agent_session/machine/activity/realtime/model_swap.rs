//! An agent switch onto another realtime model. The arriving agent's model opens its session while
//! the leaving agent's stays connected; the swap settles everything the old session owed, closes
//! it, and hands the new one the agent and the whole conversation, rebuilt from the current history
//! and options, never replayed from the old socket.

use super::Settled;
use crate::agent_session::effects::{Timer, VoiceEffect};
use crate::agent_session::{HandoffRefused, VoiceDriver};
use crate::audio::AudioChannel;
use crate::data_plane::AudioInputRouterCommand;
use crate::trace::InterruptCause;
use lk_agents_abi::Ts;
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::llm::realtime::RealtimeCapabilities;
use lk_agents_abi::productstats::models::ModelKind;
use lk_agents_abi::resources::{RealtimeModelRef, RealtimeRef, ResourceId};
use lk_agents_abi::voice::session::ModelSwapRefused;

/// The session a switch's model opened, the sink the caller's audio reaches it through, and the
/// model.
pub struct OpenedSession {
    pub session: RealtimeRef,
    pub sink: ResourceId,
    pub model: RealtimeModelRef,
}

/// Whether an agent on a model with `agent`'s capabilities (`None`: a pipeline LLM) can take over
/// a session opened on `session`'s. The kind of model, and the capabilities deciding who ends the
/// turn and how the agent is heard, shaped the session's audio pipeline when it opened.
pub fn model_swap_allowed(
    session: Option<&RealtimeCapabilities>,
    agent: Option<&RealtimeCapabilities>,
) -> Result<(), ModelSwapRefused> {
    match (session, agent) {
        (None, None) => Ok(()),
        (Some(_), None) | (None, Some(_)) => Err(ModelSwapRefused::KindChanged),
        (Some(session), Some(agent)) => {
            let shape = |c: &RealtimeCapabilities| {
                (c.audio_output, c.turn_detection, c.can_disable_turn_detection)
            };
            if shape(session) == shape(agent) {
                Ok(())
            } else {
                Err(ModelSwapRefused::PipelineChanged)
            }
        }
    }
}

impl VoiceDriver {
    /// Whether `model` is another than the one the realtime session runs on.
    pub(crate) fn realtime_model_changes(&self, model: Option<&RealtimeModelRef>) -> bool {
        let (Some(model), Some(rt)) = (model, self.activity.realtime_activity()) else {
            return false;
        };
        rt.model.as_ref().is_none_or(|current| current.instance != model.instance)
    }

    /// A drained switch's model, when another than the one connected, opens its session with
    /// the session's turn-detection choice; the open's number comes back with it.
    pub(crate) fn model_swap_open(&mut self, model: Option<&RealtimeModelRef>) -> Option<u64> {
        if !self.realtime_model_changes(model) {
            return None;
        }
        let model = model?.id;
        let span = self.activity_span();
        let rt = self.activity.realtime_activity_mut()?;
        rt.opens += 1;
        let (open, turn_detection_disabled) = (rt.opens, rt.turn_detection_disabled);
        self.record_effect(VoiceEffect::OpenRealtime {
            open,
            model,
            turn_detection_disabled,
            span,
        });
        Some(open)
    }

    /// The switch waiting on `open` lands on the session its model opened, or is refused with the
    /// leaving agent still connected. An open nobody waits for any more is closed unused.
    pub(crate) fn on_realtime_opened(
        &mut self,
        now: Ts,
        open: u64,
        outcome: Result<(RealtimeRef, ResourceId), ProviderError>,
    ) {
        let waiting = self.activity.handoff.opening(open);
        let update = if waiting { self.activity.handoff.take_in_progress() } else { None };
        match (outcome, update) {
            (Ok((session, sink)), Some(update)) => match update.realtime.clone() {
                Some(model) => {
                    self.land_handoff(now, update, Some(OpenedSession { session, sink, model }))
                }
                None => {
                    self.discard_realtime(session.id, sink);
                    self.land_handoff(now, update, None);
                }
            },
            (Ok((session, sink)), None) => self.discard_realtime(session.id, sink),
            (Err(error), Some(update)) => {
                self.report_error(now, crate::events::ErrorSource::Realtime, true, error, None);
                self.refuse_drained_handoff(now, update, HandoffRefused::SetupFailed);
            }
            (Err(_), None) => {}
        }
    }

    /// A session opened for nobody: closed, and let go with its sink.
    pub(crate) fn discard_realtime(&mut self, session: ResourceId, sink: ResourceId) {
        self.record_effect(VoiceEffect::CloseRealtimeSession(session));
        self.record_effect(VoiceEffect::ProviderRelease(sink));
        self.record_effect(VoiceEffect::ProviderRelease(session));
    }

    /// The old session's generations are cancelled and every request it owed settles as cut by
    /// the swap, its players end, its socket closes; the new session takes the caller's audio and
    /// owes its frame, which the switch's `update_agent` sends.
    pub(crate) fn swap_realtime_session(&mut self, opened: OpenedSession) {
        let OpenedSession { session, sink, model } = opened;
        let installed = crate::agent_session::product_stats::realtime_model(&session);
        self.count_model_installed(ModelKind::Realtime, Some(installed));
        let Some(rt) = self.activity.realtime_activity_mut() else {
            return self.discard_realtime(session.id, sink);
        };
        for command in rt.state.cancel_where(|_| true) {
            self.rt_send(command);
        }
        self.settle_all(Settled::Interrupted(InterruptCause::ModelSwapped), false);
        let Some(rt) = self.activity.realtime_activity_mut() else { return };
        rt.players.clear();
        let timed = rt.state.replace(session.capabilities);
        let closed = std::mem::replace(&mut rt.session, session);
        rt.model = Some(model);
        rt.owes_frame = true;
        for key in timed {
            self.clear_deadline(Timer::RealtimeRequest(key));
        }
        self.record_effect(VoiceEffect::CloseRealtimePlayers);
        self.record_effect(VoiceEffect::CloseRealtimeSession(closed.id));
        self.input.sinks.insert(AudioChannel::Realtime, sink);
        self.record_effect(VoiceEffect::UpdateAudioInputRouting(
            AudioInputRouterCommand::Install { channel: AudioChannel::Realtime, sink },
        ));
    }
}

#[cfg(test)]
#[path = "model_swap_tests.rs"]
mod tests;
