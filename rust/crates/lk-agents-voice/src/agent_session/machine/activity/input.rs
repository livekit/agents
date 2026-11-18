//! The first frontend request waits on response admission, never on remote business work.

use super::scheduling::SpeechId;
use crate::agent::{Source, Visibility};
use crate::agent_session::{Answer, VoiceDriver};
use lk_agents_abi::session::InputAdmission;
use lk_agents_abi::storage::StorageError;
use lk_agents_abi::{ChatMessage, Modality, Role};
use lk_agents_core::agent_run::RequestId as ModelRequestId;

#[cfg(test)]
#[path = "input_tests.rs"]
mod tests;

#[derive(Debug, Clone)]
pub(crate) enum ReplyInputScope {
    Snapshot(std::collections::BTreeSet<String>),
    AwaitingTranscript,
    Human(Box<ChatMessage>),
}

impl ReplyInputScope {
    pub(crate) fn capture(
        message: Option<&ChatMessage>,
        answers: &std::collections::BTreeSet<String>,
    ) -> Self {
        match message.filter(|message| message.role == Role::User) {
            Some(message) => Self::Human(Box::new(message.clone())),
            None => Self::Snapshot(answers.clone()),
        }
    }
}

pub(super) use lk_agents_core::session::input::same_human_version;

#[derive(Debug)]
pub(super) enum AfterCapture {
    Ready,
    Human,
    HumanEnded,
    Message(Box<ChatMessage>),
}

#[derive(Debug)]
pub(super) enum ReplyInputGate {
    Ready,
    Capturing { request: ModelRequestId, after: AfterCapture },
    AwaitingTranscript,
    Routing { request_id: String, human_id: String },
}

impl VoiceDriver {
    pub(crate) fn input_offers(
        &mut self,
        speech: SpeechId,
        request: ModelRequestId,
        offered: Vec<lk_agents_abi::session::InputCandidate>,
    ) {
        let Some(reply) = self.activity.tools.replies.get_mut(&speech) else { return };
        if !matches!(reply.input_gate, ReplyInputGate::Capturing { request: current, .. } if current == request)
        {
            return;
        }
        let ReplyInputGate::Capturing { after, .. } =
            std::mem::replace(&mut reply.input_gate, ReplyInputGate::Ready)
        else {
            unreachable!("the captured request was checked")
        };
        let pending = !offered.is_empty();
        reply.input_offers(offered);
        match after {
            AfterCapture::Ready => {}
            AfterCapture::Human if pending => reply.input_gate = ReplyInputGate::AwaitingTranscript,
            AfterCapture::HumanEnded if pending => reply.run.input_offers_unavailable(),
            AfterCapture::Human | AfterCapture::HumanEnded => {}
            AfterCapture::Message(message) => self.begin_input_route(speech, *message),
        }
    }

    pub(crate) fn input_capture_pending(&self, speech: SpeechId) -> bool {
        self.activity
            .tools
            .replies
            .get(&speech)
            .is_some_and(|reply| matches!(reply.input_gate, ReplyInputGate::Capturing { .. }))
    }

    pub(crate) fn begin_input_route(&mut self, speech: SpeechId, message: ChatMessage) {
        let request_id = self.session.ids.mint("input_route");
        if let Some(reply) = self.activity.tools.replies.get_mut(&speech) {
            if matches!(reply.input_scope, ReplyInputScope::AwaitingTranscript) {
                reply.input_scope = ReplyInputScope::Human(Box::new(message.clone()));
            }
            reply.input_gate = ReplyInputGate::Routing {
                request_id: request_id.clone(),
                human_id: message.id.clone(),
            };
        }
        self.replies_answered();
        self.record_effect(crate::agent_session::effects::VoiceEffect::RouteInput {
            speech,
            request_id,
            message,
        });
    }

    pub(crate) fn expect_realtime_input(&mut self, speech: SpeechId) {
        let message = self.realtime_input.take();
        let Some(reply) = self.activity.tools.replies.get_mut(&speech) else { return };
        reply.input_scope = ReplyInputScope::AwaitingTranscript;
        match &mut reply.input_gate {
            ReplyInputGate::Capturing { after, .. } => {
                *after = match message {
                    Some(message) => AfterCapture::Message(Box::new(message)),
                    None => AfterCapture::Human,
                };
            }
            _ => {
                if let Some(message) = message {
                    self.begin_input_route(speech, message);
                } else if !reply.run.input_offers().is_empty() {
                    reply.input_gate = ReplyInputGate::AwaitingTranscript;
                }
            }
        }
    }

    /// Realtime input has no responseId. Only a unique live reply awaiting that input can
    /// take it; otherwise retain the canonical words until the provider creates its reply.
    pub(crate) fn route_realtime_input(&mut self, message: ChatMessage) {
        let waiting: Vec<_> = self
            .activity
            .tools
            .replies
            .iter()
            .filter_map(|(speech, reply)| {
                (reply.realtime
                    && matches!(
                        reply.input_gate,
                        ReplyInputGate::AwaitingTranscript
                            | ReplyInputGate::Capturing {
                                after: AfterCapture::Human | AfterCapture::Message(_),
                                ..
                            }
                            | ReplyInputGate::Routing { .. }
                    ))
                .then_some(*speech)
            })
            .filter(|speech| {
                self.activity
                    .speech_scheduler
                    .speech(*speech)
                    .is_some_and(|state| !state.interrupted() && !state.done())
            })
            .collect();
        let [speech] = waiting.as_slice() else {
            let live = self.activity.tools.replies.iter().any(|(speech, reply)| {
                reply.realtime
                    && self
                        .activity
                        .speech_scheduler
                        .speech(*speech)
                        .is_some_and(|state| !state.interrupted() && !state.done())
            });
            if waiting.is_empty() && !live {
                self.realtime_input = Some(message);
            }
            return;
        };
        self.route_realtime_input_for(*speech, message);
    }

    pub(crate) fn route_realtime_input_for(&mut self, speech: SpeechId, message: ChatMessage) {
        let Some(reply) = self.activity.tools.replies.get_mut(&speech) else { return };
        if !matches!(reply.input_scope, ReplyInputScope::Human(_)) {
            reply.input_scope = ReplyInputScope::Human(Box::new(message.clone()));
        }
        if let ReplyInputGate::Capturing { after, .. } = &mut reply.input_gate {
            *after = AfterCapture::Message(Box::new(message));
        } else {
            self.begin_input_route(speech, message);
        }
    }

    /// GenerationDone is an explicit provider boundary. If it never supplied final human
    /// words, release remote callers with refusal feedback; no unowned transcript wait remains.
    pub(crate) fn realtime_input_ended(&mut self, speech: SpeechId) {
        let Some(reply) = self.activity.tools.replies.get_mut(&speech) else { return };
        match &mut reply.input_gate {
            ReplyInputGate::AwaitingTranscript => {
                reply.run.input_offers_unavailable();
                reply.input_gate = ReplyInputGate::Ready;
            }
            ReplyInputGate::Capturing { after: after @ AfterCapture::Human, .. } => {
                *after = AfterCapture::HumanEnded;
            }
            _ => {}
        }
    }

    /// Admission belongs to the conversation. Later replies and agent changes must retain
    /// its destination, even after the assistant reply awaiting the detector was interrupted.
    pub(crate) fn input_reserved(&mut self, message: ChatMessage, admission: InputAdmission) {
        if !admission.observation.is_empty() {
            let observation = ChatMessage::text(
                format!("input_route/{}", message.id),
                Role::System,
                admission.observation,
                self.step.wall(),
            );
            self.commit_item(observation.into(), Source::InputRouting, Visibility::Model);
        }
        self.hold_local_responses(&message, &admission.requests);
        self.reserve_expert_responses(message, admission.requests);
    }

    pub(crate) fn settle_canceled_input_plans(&mut self) -> bool {
        let canceled: Vec<_> = self
            .input_plans
            .keys()
            .copied()
            .filter(|speech| {
                self.session.closing_past_edge()
                    || self
                        .activity
                        .speech_scheduler
                        .speech(*speech)
                        .is_none_or(|state| state.interrupted() || state.done())
            })
            .collect();
        let ready = !canceled.is_empty();
        for speech in canceled {
            if let Some((call, _)) = self.input_plans.remove(&speech) {
                self.answer(Some(call), Answer::ReplyPlan(None));
            }
        }
        ready
    }

    pub(crate) fn input_routed(
        &mut self,
        speech: SpeechId,
        request_id: String,
        result: Option<Result<InputAdmission, StorageError>>,
        usage: lk_agents_abi::usage::Usage,
    ) {
        self.session.usage.add(&usage);
        if self.run_contains_speech(speech)
            && let Some(run) = self.session.runs.usage()
        {
            run.add(&usage);
        }
        if let Some(reply) = self.activity.tools.replies.get_mut(&speech) {
            let ReplyInputGate::Routing { request_id: current, human_id } = &reply.input_gate
            else {
                return;
            };
            if current != &request_id {
                return;
            }
            let offered = match &result {
                Some(Ok(admission)) => {
                    let captured = reply.run.input_offers();
                    let mut offered: Vec<_> = admission
                        .offered
                        .iter()
                        .filter(|candidate| {
                            !reply.realtime
                                || captured
                                    .iter()
                                    .any(|offered| offered.reference == candidate.reference)
                        })
                        .cloned()
                        .collect();
                    for candidate in captured {
                        if candidate
                            .response
                            .as_ref()
                            .is_some_and(|response| &response.id == human_id)
                            && !offered
                                .iter()
                                .any(|offered| offered.reference == candidate.reference)
                        {
                            offered.push(candidate.clone());
                        }
                    }
                    offered
                }
                _ => Vec::new(),
            };
            if result.as_ref().is_some_and(|result| result.is_ok()) {
                reply.input_offers(offered);
            } else {
                reply.run.input_offers_unavailable();
            }
            reply.input_gate = ReplyInputGate::Ready;
        }
        let Some((call, mut plan)) = self.input_plans.remove(&speech) else { return };
        let canceled = self.session.closing_past_edge()
            || self
                .activity
                .speech_scheduler
                .speech(speech)
                .is_none_or(|state| state.interrupted() || state.done());
        if canceled {
            self.answer(Some(call), Answer::ReplyPlan(None));
            return;
        }
        let observation = match result {
            // The checkpoint owner publishes the observation before resolving this waiter.
            Some(Ok(_)) => String::new(),
            Some(Err(_)) | None => "Response routing could not admit this message. Handle the \
                message and clarify naturally. Leave the pending requests open; do not forward \
                this message automatically as an answer."
                .into(),
        };
        if !observation.is_empty() {
            let instructions = plan.spec.turn_instructions.get_or_insert_default();
            instructions.push('\n');
            instructions.push_str(&observation);
        }
        let instructions = self.activity.instructions.render(Some(Modality::Audio));
        let record = self.session.chat_ctx.agent_chat_ctx();
        let request = plan.spec.plan(&record, &instructions, self.step.wall());
        self.answer(
            Some(call),
            Answer::ReplyPlan(Some(Box::new(crate::generation::PlannedReply {
                request,
                tool_choice: plan.tool_choice,
            }))),
        );
    }
}
