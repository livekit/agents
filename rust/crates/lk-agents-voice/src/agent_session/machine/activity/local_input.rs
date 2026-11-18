//! Local questions use the shared occurrence journal; live bodies keep only their native owner.

use std::collections::{BTreeMap, BTreeSet};

use super::input::same_human_version;
use super::scheduling::SpeechId;
use super::voice_task::VoiceTaskId;
use crate::agent::{Source, Visibility};
use crate::agent_session::VoiceDriver;
use crate::agent_session::effects::VoiceEffect;
use crate::experts::Floor;
use crate::trace::ToolStep;
use lk_agents_abi::agent::run::RunInputRequirement;
use lk_agents_abi::session::{
    InputCandidate, InputRequestRef, InputRequestState, SessionInputCommand,
};
use lk_agents_abi::{ChatMessage, Role};
use lk_agents_core::session::input;

#[cfg(test)]
#[path = "local_input_tests.rs"]
mod tests;

#[derive(Debug)]
enum Owner {
    Agent { agent: String, task: Option<VoiceTaskId> },
    Body { output_id: String },
}

#[derive(Debug)]
enum Registration {
    Opening,
    Presented(String),
    Ready(InputRequestRef),
    PresentedReference(InputRequestRef),
}

#[derive(Debug)]
struct Request {
    owner: Owner,
    registration: Registration,
}

#[derive(Debug, Clone)]
enum DeliveryOwner {
    Agent { agent: String, task: Option<VoiceTaskId>, human: Option<ChatMessage> },
    Body(crate::tool_executor::HostedTool),
}

/// A private native receipt carries the owner chosen before the checkpoint, never a new floor.
#[derive(Debug)]
pub struct LocalInputDelivery {
    request: InputRequestRef,
    call_id: Option<String>,
    message: ChatMessage,
    owner: DeliveryOwner,
}

#[derive(Debug, Default)]
pub(crate) struct LocalInputs {
    requests: BTreeMap<String, Request>,
    presentations: BTreeMap<SpeechId, Vec<String>>,
    held: BTreeMap<InputRequestRef, ChatMessage>,
    delivered: BTreeSet<InputRequestRef>,
    presented: BTreeSet<InputRequestRef>,
    answers: BTreeMap<(String, Option<VoiceTaskId>), BTreeMap<String, ChatMessage>>,
    humans: BTreeMap<(String, Option<VoiceTaskId>), ChatMessage>,
}

impl VoiceDriver {
    /// A foreground collector owns its input. Its own local read-backs can still be matched,
    /// but an earlier root or remote question cannot collect the same response.
    pub fn input_routing_candidates(
        &self,
        mut candidates: Vec<InputCandidate>,
    ) -> Vec<InputCandidate> {
        if self.activity.voice_tasks.running() {
            candidates.retain(|candidate| match &candidate.reference {
                InputRequestRef::Agent { request_id, .. } => {
                    self.local_owner_eligible(request_id, &candidate.reference)
                }
                InputRequestRef::RemoteTask { .. } => false,
            });
        }
        candidates
    }

    pub(crate) fn retain_local_requirements(
        &mut self,
        speech: SpeechId,
        requirements: &[RunInputRequirement],
    ) {
        for requirement in requirements {
            let expert = self.activity.tools.running.iter().any(|running| {
                running.call.id == requirement.call.id
                    && self.session.experts.owns(&running.output_id)
            });
            if !expert {
                self.retain_local_question(speech, &requirement.call.id, &requirement.question);
            }
        }
    }

    pub(crate) fn retain_local_question(
        &mut self,
        speech: SpeechId,
        call_id: &str,
        question: &str,
    ) {
        let agent = self.activity.speech_agent(speech).unwrap_or(&self.activity.agent).to_owned();
        let task = self.activity.voice_tasks.of_agent(&agent);
        self.retain_local_input(call_id, call_id, question, Owner::Agent { agent, task });
    }

    pub(crate) fn retain_body_question(
        &mut self,
        output_id: &str,
        request_id: &str,
        call_id: &str,
        question: &str,
    ) {
        self.close_body_inputs(output_id);
        self.retain_local_input(
            request_id,
            call_id,
            question,
            Owner::Body { output_id: output_id.to_owned() },
        );
    }

    fn retain_local_input(
        &mut self,
        request_id: &str,
        call_id: &str,
        question: &str,
        owner: Owner,
    ) {
        if self.session.closing_past_edge()
            || self.session.local_inputs.requests.contains_key(request_id)
        {
            return;
        }
        self.session
            .local_inputs
            .requests
            .insert(request_id.to_owned(), Request { owner, registration: Registration::Opening });
        let source = ChatMessage::text(
            format!("{request_id}/input"),
            Role::Assistant,
            question,
            self.step.wall(),
        );
        self.record_effect(VoiceEffect::RetainLocalInput {
            request_id: request_id.to_owned(),
            call_id: call_id.to_owned(),
            source,
        });
    }

    pub(crate) fn local_input_retained(&mut self, request_id: String, reference: InputRequestRef) {
        let Some(request) = self.session.local_inputs.requests.get_mut(&request_id) else {
            self.record_effect(VoiceEffect::CloseLocalInputs { requests: vec![reference] });
            return;
        };
        let previous =
            std::mem::replace(&mut request.registration, Registration::Ready(reference.clone()));
        if let Owner::Body { output_id } = &request.owner {
            self.session.experts.reference_retained(output_id, &request_id, reference.clone());
        }
        if let Registration::Presented(message_id) = previous {
            if let Some(request) = self.session.local_inputs.requests.get_mut(&request_id) {
                request.registration = Registration::PresentedReference(reference.clone());
            }
            self.record_effect(VoiceEffect::InputPresented {
                requests: vec![reference],
                message_id,
            });
        }
    }

    pub(crate) fn presenting_local_input(&mut self, speech: SpeechId, request_id: String) {
        self.session.local_inputs.presentations.entry(speech).or_default().push(request_id);
    }

    pub(crate) fn local_input_presented(
        &mut self,
        speech: SpeechId,
        message_id: &str,
    ) -> Vec<InputRequestRef> {
        let requests = self.session.local_inputs.presentations.remove(&speech).unwrap_or_default();
        let mut references = Vec::new();
        for request_id in requests {
            let Some(request) = self.session.local_inputs.requests.get_mut(&request_id) else {
                continue;
            };
            match &request.registration {
                Registration::Ready(reference) | Registration::PresentedReference(reference) => {
                    references.push(reference.clone());
                    request.registration = Registration::PresentedReference(reference.clone());
                }
                Registration::Opening | Registration::Presented(_) => {
                    request.registration = Registration::Presented(message_id.to_owned());
                }
            }
        }
        references
    }

    pub(crate) fn local_input_presentation_ended(&mut self, speech: SpeechId) {
        self.session.local_inputs.presentations.remove(&speech);
    }

    fn local_owner_eligible(&self, request_id: &str, reference: &InputRequestRef) -> bool {
        match self.session.local_inputs.requests.get(request_id).map(|request| &request.owner) {
            Some(Owner::Body { output_id }) => {
                let tools = self.activity.expert_tools();
                let floor =
                    Floor { tools: &tools, free: true, task: self.activity.voice_tasks.running() };
                self.session.experts.accepts(reference, floor)
                    && self
                        .activity
                        .tools
                        .running
                        .iter()
                        .any(|running| &running.output_id == output_id)
            }
            Some(Owner::Agent { agent, task }) => {
                agent == &self.activity.agent
                    && *task == self.activity.voice_tasks.of_agent(&self.activity.agent)
            }
            // A saved generic root question has no transient owner in this process.
            None => {
                self.activity.agent == self.session.root_agent
                    && !self.activity.voice_tasks.running()
            }
        }
    }

    /// The serialized executor checkpoints these exact deliveries before releasing planning or
    /// calling a live body. A question from another context remains held for that owner.
    pub fn admit_local_responses(
        &self,
        state: &mut lk_agents_abi::session::snapshot::SessionSnapshot,
    ) -> Result<Vec<LocalInputDelivery>, lk_agents_abi::storage::StorageError> {
        self.readmit_local_responses(state, Vec::new())
    }

    /// A cancelled control scope can retry its checkpoint without changing the already chosen
    /// local owner. The occurrence and full canonical human response are revalidated together.
    pub fn readmit_local_responses(
        &self,
        state: &mut lk_agents_abi::session::snapshot::SessionSnapshot,
        mut delivered: Vec<LocalInputDelivery>,
    ) -> Result<Vec<LocalInputDelivery>, lk_agents_abi::storage::StorageError> {
        delivered.retain(|delivery| self.local_delivery_alive(delivery));
        for delivery in &delivered {
            if !state.chat_history.iter().any(|message| message == &delivery.message) {
                return Err(lk_agents_abi::storage::StorageError::Invalid(
                    "the admitted local response changed before its checkpoint retry".into(),
                ));
            }
        }
        let mut commands: Vec<_> = delivered
            .iter()
            .map(|delivery| SessionInputCommand::Delivered {
                requests: vec![delivery.request.clone()],
                message_id: delivery.message.id.clone(),
            })
            .collect();
        for (agent_id, agent) in &state.frames {
            for (request_id, request) in &agent.input_requests {
                if request.state != InputRequestState::Responding {
                    continue;
                }
                let reference = InputRequestRef::Agent {
                    agent_id: agent_id.value.clone(),
                    request_id: request_id.clone(),
                };
                if delivered.iter().any(|delivery| delivery.request == reference)
                    || !self.local_owner_eligible(request_id, &reference)
                {
                    continue;
                }
                let response = input::response(state, &reference)?;
                commands.push(SessionInputCommand::Delivered {
                    requests: vec![reference.clone()],
                    message_id: response.id.clone(),
                });
                let owner = match self
                    .session
                    .local_inputs
                    .requests
                    .get(request_id)
                    .map(|input| &input.owner)
                {
                    Some(Owner::Body { output_id }) => {
                        let running = self
                            .activity
                            .tools
                            .running
                            .iter()
                            .find(|running| &running.output_id == output_id)
                            .expect("the eligible body was checked in this transition");
                        DeliveryOwner::Body(running.hosted.clone())
                    }
                    Some(Owner::Agent { agent, task }) => DeliveryOwner::Agent {
                        agent: agent.clone(),
                        task: *task,
                        human: self
                            .session
                            .local_inputs
                            .humans
                            .get(&(agent.clone(), *task))
                            .cloned(),
                    },
                    None => {
                        let agent = self.session.root_agent.clone();
                        let human =
                            self.session.local_inputs.humans.get(&(agent.clone(), None)).cloned();
                        DeliveryOwner::Agent { agent, task: None, human }
                    }
                };
                delivered.push(LocalInputDelivery {
                    request: reference,
                    call_id: request.origin.as_ref().map(|origin| origin.call_id.clone()),
                    message: response,
                    owner,
                });
            }
        }
        input::apply(state, commands)?;
        Ok(delivered)
    }

    pub(crate) fn hold_local_responses(
        &mut self,
        message: &ChatMessage,
        requests: &[InputRequestRef],
    ) {
        for reference in requests.iter().filter(|reference| reference.remote().is_none()) {
            if !self.session.local_inputs.delivered.contains(reference) {
                self.session.local_inputs.held.insert(reference.clone(), message.clone());
            }
        }
    }

    pub(crate) fn local_input_candidates(&mut self, requests: BTreeSet<InputRequestRef>) {
        self.session.local_inputs.presented = requests;
    }

    pub(crate) fn needs_local_input_routing(&self) -> bool {
        self.session
            .local_inputs
            .presented
            .iter()
            .any(|reference| !self.session.local_inputs.delivered.contains(reference))
            || self.session.local_inputs.requests.values().any(|request| {
                matches!(&request.registration, Registration::PresentedReference(reference)
                    if !self.session.local_inputs.delivered.contains(reference))
            })
    }

    pub(crate) fn check_local_responses(&mut self) {
        let eligible = self.session.local_inputs.held.keys().any(|reference| {
            let InputRequestRef::Agent { request_id, .. } = reference else { return false };
            self.local_owner_eligible(request_id, reference)
        });
        if eligible && !self.session.closing_past_edge() {
            self.record_effect(VoiceEffect::CheckpointLocalInputs);
        }
    }

    pub(crate) fn local_approval_answers(&self, agent: &str) -> BTreeSet<String> {
        let task = self.activity.voice_tasks.of_agent(agent);
        self.session
            .local_inputs
            .answers
            .get(&(agent.to_owned(), task))
            .into_iter()
            .flat_map(|answers| answers.iter())
            .filter(|(_, response)| self.local_response_current(response))
            .map(|(call_id, _)| call_id.clone())
            .collect()
    }

    pub(crate) fn scoped_local_approval_answers(
        &self,
        agent: &str,
        scope: &super::input::ReplyInputScope,
    ) -> BTreeSet<String> {
        use super::input::ReplyInputScope;
        let task = self.activity.voice_tasks.of_agent(agent);
        self.session
            .local_inputs
            .answers
            .get(&(agent.to_owned(), task))
            .into_iter()
            .flat_map(|answers| answers.iter())
            .filter(|(call_id, response)| {
                self.local_response_current(response)
                    && match scope {
                        ReplyInputScope::Snapshot(captured) => captured.contains(*call_id),
                        ReplyInputScope::AwaitingTranscript => false,
                        ReplyInputScope::Human(captured) => same_human_version(captured, response),
                    }
            })
            .map(|(call_id, _)| call_id.clone())
            .collect()
    }

    fn local_response_current(&self, response: &ChatMessage) -> bool {
        self.session.chat_ctx.items().any(
            |item| matches!(item, lk_agents_abi::ChatItem::Message(current) if same_human_version(current, response)),
        )
    }

    pub(crate) fn local_human_observed(&mut self, agent: &str, message: &ChatMessage) {
        let owner = (agent.to_owned(), self.activity.voice_tasks.of_agent(agent));
        if self
            .session
            .local_inputs
            .humans
            .get(&owner)
            .is_some_and(|current| same_human_version(current, message))
        {
            return;
        }
        // A same-ID transcript revision invalidates every owner's earlier copy as well.
        let revised: Vec<_> = self
            .session
            .local_inputs
            .humans
            .iter()
            .filter_map(|(owner, current)| {
                (current.id == message.id && !same_human_version(current, message))
                    .then_some(owner.clone())
            })
            .collect();
        for owner in revised {
            self.session.local_inputs.answers.remove(&owner);
            self.session.local_inputs.humans.insert(owner, message.clone());
        }
        self.session.local_inputs.answers.remove(&owner);
        self.session.local_inputs.humans.insert(owner, message.clone());
        self.replies_answered();
    }

    pub(crate) fn local_history_observed(&mut self, agent: &str) {
        let latest =
            self.session.chat_ctx.agent_chat_ctx().items.into_iter().rev().find_map(|item| {
                match item {
                    lk_agents_abi::ChatItem::Message(message) if message.role == Role::User => {
                        Some(message)
                    }
                    _ => None,
                }
            });
        if let Some(message) = latest {
            self.local_human_observed(agent, &message);
        }
    }

    fn local_delivery_alive(&self, delivery: &LocalInputDelivery) -> bool {
        match &delivery.owner {
            DeliveryOwner::Body(hosted) => {
                self.activity.tools.running.iter().any(|running| running.hosted == *hosted)
                    && self.session.experts.waits_for(&delivery.request)
            }
            DeliveryOwner::Agent { task, .. } => {
                task.is_none_or(|task| self.activity.voice_tasks.contains(task))
            }
        }
    }

    pub(crate) fn local_input_delivered(&mut self, delivery: LocalInputDelivery) {
        let LocalInputDelivery { request, call_id, message, owner } = delivery;
        self.session.local_inputs.held.remove(&request);
        if self.session.local_inputs.delivered.contains(&request) {
            return;
        }
        match owner {
            DeliveryOwner::Body(hosted) => {
                // An admitted body keeps its response across floor changes. Its death, unlike a
                // floor change, removes the exact invocation and makes a late callback invalid.
                if !self.activity.tools.running.iter().any(|running| running.hosted == hosted)
                    || !self.session.experts.response_queued(&request)
                {
                    return;
                }
                self.forward_expert_reply(hosted, message.clone(), request.clone());
            }
            DeliveryOwner::Agent { agent, task, human } => {
                if task.is_some_and(|task| !self.activity.voice_tasks.contains(task)) {
                    return;
                }
                let current = self.session.local_inputs.humans.get(&(agent.clone(), task));
                let same_turn =
                    match (current, human.as_ref()) {
                        (Some(current), Some(human)) => same_human_version(current, human),
                        (None, None) => true,
                        _ => false,
                    } && human.as_ref().is_none_or(|human| same_human_version(human, &message))
                        && self.local_response_current(&message);
                if same_turn && let Some(call_id) = &call_id {
                    self.session
                        .local_inputs
                        .answers
                        .entry((agent.clone(), task))
                        .or_default()
                        .insert(call_id.clone(), message.clone());
                    self.replies_answered();
                    let notice = self.session.chat_ctx.items().find_map(|item| match item {
                        lk_agents_abi::ChatItem::FunctionCall(call)
                            if &call.id == call_id
                                && lk_agents_core::approval::mark(call)
                                    == Some(lk_agents_abi::llm::tool_executor::Approval::Asked) =>
                        {
                            Some(lk_agents_core::approval::answered_text(
                                &call.name,
                                &message.raw_text_content().unwrap_or_default(),
                            ))
                        }
                        _ => None,
                    });
                    if let Some(notice) = notice {
                        let observation = ChatMessage::text(
                            format!("{}/approval/{call_id}", message.id),
                            Role::System,
                            notice,
                            self.step.wall(),
                        );
                        self.conversation_changed();
                        let committed = self.session.chat_ctx.commit(
                            observation.into(),
                            Source::InputRouting,
                            Visibility::Model,
                            &agent,
                        );
                        self.perform_commit(committed);
                    }
                }
            }
        }
        self.session.local_inputs.delivered.insert(request.clone());
        self.record_trace(ToolStep::InputResponseDelivered {
            request,
            call_id,
            message_id: message.id,
        });
    }

    pub(crate) fn close_body_inputs(&mut self, output_id: &str) {
        self.close_local_inputs(
            |owner| matches!(owner, Owner::Body { output_id: owned } if owned == output_id),
        );
    }

    pub(crate) fn close_task_inputs(&mut self, task: VoiceTaskId) {
        self.session.local_inputs.answers.retain(|(_, owner), _| *owner != Some(task));
        self.session.local_inputs.humans.retain(|(_, owner), _| *owner != Some(task));
        self.close_local_inputs(
            |owner| matches!(owner, Owner::Agent { task: Some(owned), .. } if *owned == task),
        );
    }

    fn close_local_inputs(&mut self, closes: impl Fn(&Owner) -> bool) {
        let ids: BTreeSet<_> = self
            .session
            .local_inputs
            .requests
            .iter()
            .filter(|(_, request)| closes(&request.owner))
            .map(|(id, _)| id.clone())
            .collect();
        let mut requests = Vec::new();
        for id in ids {
            if let Some(request) = self.session.local_inputs.requests.remove(&id)
                && let Registration::Ready(reference) | Registration::PresentedReference(reference) =
                    request.registration
            {
                self.session.local_inputs.held.remove(&reference);
                self.session.local_inputs.presented.remove(&reference);
                requests.push(reference);
            }
        }
        if !requests.is_empty() {
            self.record_effect(VoiceEffect::CloseLocalInputs { requests });
        }
    }
}
