//! The remote session protocol on its live transport.

#[cfg(test)]
mod client;
pub mod driver;
mod host;
pub mod room;

pub use host::SessionHost;
use lk_agents_abi::worker::remote_session::WireEvent;
pub use lk_agents_voice::remote_session::*;

/// Opens the production remote host on the session's selected transport.
pub fn hosting(
    config: lk_agents_voice::agent_session::api::RemoteSessionConfig,
    time: lk_agents_voice::agent_session::Timebase,
    spawner: std::sync::Arc<dyn lk_agents_abi::Spawner>,
) -> std::sync::Arc<dyn lk_agents_abi::worker::remote_session::SessionHosting> {
    use std::sync::Arc;
    let transport = match &config.console {
        Some(console) => console.clone().as_transport(spawner.clone()),
        None => Arc::new(room::RoomTransport::new(
            config.room.expect("a remote room session has an RTC attachment"),
            time.clock(),
            spawner.clone(),
        )) as Arc<dyn SessionTransport>,
    };
    Arc::new(driver::Hosting::new(
        config.port,
        transport,
        config.console.map(|console| console.as_console()),
        time,
        spawner,
    ))
}
