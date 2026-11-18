use super::{Phase, Session};
use crate::session::{InputCandidate, input};
use lk_agents_abi::session::SessionInputCommand;
use lk_agents_abi::storage::StorageError;
use lk_agents_abi::{ChatMessage, Role};

impl Session {
    pub(super) fn route_input(
        &mut self,
        message: ChatMessage,
        now: lk_agents_abi::Ts,
        effects: &mut Vec<super::SessionEffect>,
    ) {
        let human = crate::session::live::FrozenHuman {
            message,
            candidates: self.input_candidates(),
            offered: crate::delegation::remote::routing::offered(&self.state),
            history: self.state.chat_history.clone(),
        };
        self.route_frozen_input(human, now, effects);
    }

    /// Input received behind a live body keeps the occurrences the person could actually
    /// answer then. A question presented while it waited cannot become a new answer target.
    pub(super) fn route_frozen_input(
        &mut self,
        human: crate::session::live::FrozenHuman,
        now: lk_agents_abi::Ts,
        effects: &mut Vec<super::SessionEffect>,
    ) {
        let crate::session::live::FrozenHuman { message, candidates, offered, history } = human;
        if candidates.is_empty() || message.raw_text_content().unwrap_or_default().trim().is_empty()
        {
            self.invoke_offered(Some(message), Some(offered), Vec::new(), now, effects);
        } else {
            let request_id = self.mint("input_route");
            effects.push(super::SessionEffect::RouteInput(crate::session::InputRoutingRequest {
                request_id: request_id.clone(),
                message: message.clone(),
                history,
                candidates: candidates.clone(),
            }));
            self.phase = Phase::Routing { request_id, message, candidates, offered };
        }
    }

    pub(super) fn input_routed(
        &mut self,
        request_id: String,
        result: Option<Result<super::InputRoute, lk_agents_abi::errors::ProviderError>>,
        usage: lk_agents_abi::usage::Usage,
        now: lk_agents_abi::Ts,
        effects: &mut Vec<super::SessionEffect>,
    ) {
        let Phase::Routing { request_id: current, message, candidates, offered } = &self.phase
        else {
            return;
        };
        if *current != request_id {
            return;
        }
        let message = message.clone();
        let offered = offered.clone();
        self.usage.add(&usage);
        let admission = match input::route(&mut self.state, &message, candidates, result) {
            Ok(admission) => admission,
            Err(error) => {
                let Some(current) = self
                    .state
                    .chat_history
                    .iter()
                    .find(|current| current.id == message.id && current.role == Role::User)
                    .cloned()
                else {
                    self.finish(
                        Err(super::TextSessionError::Routing(super::RoutingError::InvalidState(
                            error.0,
                        ))),
                        effects,
                    );
                    return;
                };
                // Transcript revisions keep their identity. Earlier words cannot authorize
                // the revision, even when the old verdict selected the same occurrence.
                self.route_input(current, now, effects);
                return;
            }
        };
        // The current agent receives the same human message exactly once and a private
        // framework observation. It does not lose the conversational floor to the detector.
        let id = self.mint("input_route");
        let observation =
            ChatMessage::text(id, Role::System, admission.observation, self.stamp(now));
        let frame = self.active_frame_mut();
        frame.history.push(observation.into());
        self.phase = Phase::NotStarted;
        let offered = crate::delegation::remote::routing::refresh_offered(&self.state, &offered);
        self.invoke_offered(Some(message), Some(offered), Vec::new(), now, effects);
    }

    pub fn input(&mut self, command: SessionInputCommand) -> Result<(), StorageError> {
        if matches!(self.phase, Phase::Done) {
            return Err(StorageError::Invalid("the conversation owner has ended".into()));
        }
        if (self.closing || self.persistence_failure.is_some())
            && !matches!(command, SessionInputCommand::Presented { .. })
        {
            return Err(StorageError::Invalid("the conversation owner is closing".into()));
        }
        input::apply(&mut self.state, vec![command])
    }

    pub fn admit_input(
        &mut self,
        request: super::InputRoutingRequest,
        verdict: Option<Result<super::InputRoute, lk_agents_abi::errors::ProviderError>>,
    ) -> Result<super::InputAdmission, StorageError> {
        if self.closing || self.persistence_failure.is_some() || matches!(self.phase, Phase::Done) {
            return Err(StorageError::Invalid("the conversation owner is closing or ended".into()));
        }
        input::route(&mut self.state, &request.message, &request.candidates, verdict)
            .map_err(|error| StorageError::Invalid(error.0))
    }
    pub fn input_candidates(&self) -> Vec<InputCandidate> {
        input::candidates(&self.state)
    }
}
