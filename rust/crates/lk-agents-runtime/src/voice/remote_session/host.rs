//! Request dispatch to the live session port.

use lk_agents_abi::errors::HostError;
use lk_agents_abi::worker::remote_session::SessionPort;
use lk_agents_voice::remote_session::{SessionReply, encode_session_reply, io_update, proto};
use std::sync::Arc;

pub struct SessionHost {
    port: Arc<dyn SessionPort>,
    sdk: (String, String),
}

impl SessionHost {
    pub fn new(port: Arc<dyn SessionPort>) -> Self {
        let sdk = port.framework_info();
        SessionHost { port, sdk }
    }

    /// A failed host call stays local; an unknown request receives no response.
    pub async fn handle_request(
        &self,
        req: &proto::SessionRequest,
    ) -> Result<Option<proto::AgentSessionMessage>, HostError> {
        use proto::session_request::Request;
        let Some(request) = req.request.as_ref() else { return Ok(None) };
        let reply = match request {
            Request::Ping(_) => SessionReply::Pong,
            Request::GetChatHistory(_) => {
                SessionReply::ChatHistory(self.port.chat_history().await?)
            }
            Request::GetAgentInfo(_) => SessionReply::AgentInfo(self.port.agent_info().await?),
            Request::RunInput(input) if input.text.is_empty() => {
                SessionReply::RunInput((None, Some("empty run_input text".to_owned())))
            }
            Request::RunInput(input) => {
                SessionReply::RunInput(self.port.run_input(&input.text).await?)
            }
            Request::GetSessionState(_) => {
                SessionReply::SessionState(self.port.session_state().await?)
            }
            Request::GetRtcStats(_) => SessionReply::RtcStats(self.port.rtc_stats().await?),
            Request::GetSessionUsage(_) => {
                SessionReply::SessionUsage(self.port.session_usage().await?)
            }
            Request::GetFrameworkInfo(_) => {
                SessionReply::FrameworkInfo(self.sdk.0.clone(), self.sdk.1.clone())
            }
            Request::UpdateIo(io) => {
                self.port.update_io(io_update(io)).await?;
                SessionReply::UpdateIo
            }
            Request::FinalizeSimulation(input) => SessionReply::FinalizeSimulation(
                self.port
                    .finalize_simulation((
                        input.provisional_success,
                        input.provisional_reason.clone(),
                    ))
                    .await?,
            ),
        };
        Ok(Some(encode_session_reply(req.request_id.clone(), reply, lk_agents_abi::unix_now())))
    }
}
