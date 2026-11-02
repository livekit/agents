//! The remote session protocol on its live transport.

#[cfg(test)]
mod client;
pub mod driver;
pub mod room;

use crate::voice::session::Timebase;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::worker::remote_session::WireEvent;
use std::sync::Arc;

pub use lk_agents_voice::remote_session::*;

/// Opens the production remote host on the session's selected transport.
pub fn hosting(
    config: crate::voice::session::RemoteSessionConfig,
    time: crate::voice::session::Timebase,
    spawner: std::sync::Arc<dyn lk_agents_abi::Spawner>,
) -> std::sync::Arc<dyn lk_agents_abi::worker::remote_session::SessionHosting> {
    use std::sync::Arc;
    let transport = match &config.console {
        Some(console) => console.clone().as_transport(spawner.clone()),
        None => Arc::new(room::RoomTransport::new(
            config.room.expect("a remote room session has an RTC attachment"),
            time.clock(),
        )) as Arc<dyn SessionTransport>,
    };
    Arc::new(driver::RemoteSessionLink::new(
        config.port,
        transport,
        config.console.map(|console| console.as_console()),
        time,
        spawner,
    ))
}

/// Execute the one operation planned by the voice protocol core.
pub fn reply<'a>(
    port: &'a dyn SessionPort,
    request: &'a proto::SessionRequest,
) -> lk_agents_abi::BoxFuture<'a, Result<Option<SessionReply>, lk_agents_abi::errors::HostError>> {
    Box::pin(async move {
        let Some(plan) = plan_request(request) else { return Ok(None) };
        let answer = match plan {
            RequestPlan::Reply(answer) => *answer,
            RequestPlan::ChatHistory => SessionReply::ChatHistory(port.chat_history().await?),
            RequestPlan::AgentInfo => SessionReply::AgentInfo(port.agent_info().await?),
            RequestPlan::RunInput(text) => SessionReply::RunInput(port.run_input(text).await?),
            RequestPlan::SessionState => SessionReply::SessionState(port.session_state().await?),
            RequestPlan::RtcStats => SessionReply::RtcStats(port.rtc_stats().await?),
            RequestPlan::SessionUsage => SessionReply::SessionUsage(port.session_usage().await?),
            RequestPlan::FrameworkInfo => {
                let (sdk, version) = port.framework_info();
                SessionReply::FrameworkInfo(sdk, version)
            }
            RequestPlan::UpdateIo(update) => {
                port.update_io(update).await?;
                SessionReply::UpdateIo
            }
            RequestPlan::FinalizeSimulation(provisional) => {
                SessionReply::FinalizeSimulation(port.finalize_simulation(provisional).await?)
            }
        };
        Ok(Some(answer))
    })
}

// Request execution and session-clock reply encoding.

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
