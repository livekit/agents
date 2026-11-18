//! Execute the configured detector and submit its verdict to the serialized conversation owner.

use super::VoiceRuntime;
use lk_agents_abi::ChatMessage;
use lk_agents_abi::session::{InputAdmission, SessionCommands as _};
use lk_agents_core::session::input;
use lk_agents_voice::agent_session::VoiceEvent;
use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId;

impl VoiceRuntime {
    pub(super) fn route_input(&mut self, speech: SpeechId, message: ChatMessage) {
        let state = self.conversation_snapshot();
        let options = self.delegation.clone();
        let commands = self.conversation_port.clone();
        let shared = self.shared.clone();
        let reports = self.reports.clone();
        let lost = reports.clone();
        self.speeches.callback_feeding(
            speech,
            super::HostCallbackId::InputRouting,
            "voice response routing",
            move || {
                let _ = lost.send(VoiceEvent::InputRouted {
                    speech,
                    result: None,
                    usage: Default::default(),
                });
            },
            async move {
                let mut usage = Default::default();
                let result = async {
                    let state = state?;
                    let Some(request) =
                        input::routing_request(&state, format!("voice-input/{speech}"), message)
                    else {
                        return Ok(InputAdmission::default());
                    };
                    let verdict = match options {
                        Some(options) => {
                            let (verdict, consumed) =
                                lk_agents_inference::delegation::input::classify(
                                    options.routing_model,
                                    &shared.background(),
                                    &shared.trace,
                                    request.clone(),
                                    shared.conn.llm.unwrap_or_default(),
                                )
                                .await;
                            usage = consumed;
                            Some(verdict)
                        }
                        None => None,
                    };
                    commands.admit_input(request, verdict).await
                }
                .await;
                let _ =
                    reports.send(VoiceEvent::InputRouted { speech, result: Some(result), usage });
            },
        );
    }
}
