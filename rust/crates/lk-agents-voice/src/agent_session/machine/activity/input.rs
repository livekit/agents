//! The first frontend request waits on response admission, never on remote business work.

use super::scheduling::SpeechId;
use crate::agent::{Source, Visibility};
use crate::agent_session::{Answer, VoiceDriver};
use lk_agents_abi::session::InputAdmission;
use lk_agents_abi::storage::StorageError;
use lk_agents_abi::{ChatMessage, Modality, Role};

impl VoiceDriver {
    /// Admission belongs to the conversation. Later replies and agent changes must retain
    /// its destination, even after the assistant reply awaiting the detector was interrupted.
    pub(crate) fn input_reserved(&mut self, message: ChatMessage, mut admission: InputAdmission) {
        if !admission.requests.is_empty() {
            admission.observation.push_str(
                " The voice session forwards recognized responses to eligible running remote \
                continuations. Do not start another remote call merely to repeat this answer. \
                Continue handling the person's whole message, including any other topic.",
            );
        }
        if !admission.observation.is_empty() {
            let observation = ChatMessage::text(
                format!("input_route/{}", message.id),
                Role::System,
                admission.observation,
                self.step.wall(),
            );
            self.commit_item(observation.into(), Source::InputRouting, Visibility::Model);
        }
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
        result: Option<Result<InputAdmission, StorageError>>,
        usage: lk_agents_abi::usage::Usage,
    ) {
        self.session.usage.add(&usage);
        if self.run_contains_speech(speech)
            && let Some(run) = self.session.runs.usage()
        {
            run.add(&usage);
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
