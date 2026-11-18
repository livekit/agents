//! Execute the configured detector and submit its verdict to the serialized conversation owner.

use super::VoiceRuntime;
use lk_agents_abi::ChatMessage;
use lk_agents_abi::session::{InputAdmission, SessionCommands as _};
use lk_agents_core::session::input;
use lk_agents_voice::agent_session::VoiceEvent;
use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId;

impl VoiceRuntime {
    pub(super) fn route_input(
        &mut self,
        speech: SpeechId,
        request_id: String,
        message: ChatMessage,
    ) {
        let state = self.conversation_snapshot();
        let options = self.delegation.clone();
        let commands = self.conversation_port.clone();
        let shared = self.shared.clone();
        let reports = self.reports.clone();
        let lost = reports.clone();
        let lost_request = request_id.clone();
        self.speeches.callback_feeding(
            speech,
            super::HostCallbackId::InputRouting(request_id.clone()),
            "voice response routing",
            move || {
                let _ = lost.send(VoiceEvent::InputRouted {
                    speech,
                    request_id: lost_request,
                    result: None,
                    usage: Default::default(),
                });
            },
            async move {
                let mut usage = Default::default();
                let result = async {
                    let state = state?;
                    let Some(request) = input::routing_request(&state, request_id.clone(), message)
                    else {
                        return Ok(InputAdmission {
                            offered: lk_agents_core::delegation::remote::routing::offered(&state),
                            ..Default::default()
                        });
                    };
                    let verdict = match options.and_then(|options| options.routing_model) {
                        Some(model) => {
                            let (verdict, consumed) =
                                lk_agents_inference::delegation::input::classify(
                                    model,
                                    &shared.background(),
                                    &shared.trace,
                                    request.clone(),
                                    shared.conn.llm.unwrap_or_default(),
                                )
                                .await;
                            usage = consumed;
                            Some(verdict)
                        }
                        None => Some(Ok(lk_agents_abi::session::InputRoute::NoMatch)),
                    };
                    commands.admit_input(request, verdict).await
                }
                .await;
                let _ = reports.send(VoiceEvent::InputRouted {
                    speech,
                    request_id,
                    result: Some(result),
                    usage,
                });
            },
        );
    }
}
