//! Request execution and session-clock reply encoding.

use super::reply;
use crate::voice::session::Timebase;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::worker::remote_session::SessionPort;
use lk_agents_voice::remote_session::{encode_session_reply, proto};
use std::sync::Arc;

pub struct SessionHost {
    port: Arc<dyn SessionPort>,
    time: Timebase,
}

impl SessionHost {
    pub fn new(port: Arc<dyn SessionPort>, time: Timebase) -> Self {
        SessionHost { port, time }
    }

    /// A failed host call stays local; an unknown request receives no response.
    pub async fn handle_request(
        &self,
        request: &proto::SessionRequest,
    ) -> Result<Option<proto::AgentSessionMessage>, HostError> {
        Ok(reply(&*self.port, request).await?.map(|reply| {
            encode_session_reply(request.request_id.clone(), reply, self.time.wall_secs())
        }))
    }
}
