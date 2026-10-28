//! Test room performers execute the production voice decision cores against supplied-clock peers.

pub mod output;
pub mod transcription;

pub mod input;
pub mod owner;
pub mod pre_connect_audio;

use lk_agents_abi::voice::{room_io::RoomIoConfig, room_owner, rtc::RtcRoom};
use lk_agents_abi::{Clock, Spawner, VirtualClock};
use lk_agents_voice::agent_session::Timebase;
use std::sync::Arc;
pub struct FixedRoom {
    pub rtc: Arc<dyn RtcRoom>,
    pub time: Timebase,
}
impl room_owner::SessionRoom for FixedRoom {
    fn rtc(&self) -> Option<Arc<dyn RtcRoom>> {
        Some(self.rtc.clone())
    }
    fn clock(&self) -> Arc<dyn Clock> {
        self.time.clock()
    }
    fn virtual_clock(&self) -> Option<Arc<VirtualClock>> {
        self.time.virtual_clock()
    }
    fn room_io(
        &self,
        spawner: Arc<dyn Spawner>,
        config: RoomIoConfig,
    ) -> Arc<dyn room_owner::RoomIo> {
        owner::RoomIo::new(self.rtc.clone(), self.time.clock(), spawner, config)
    }
}

#[cfg(test)]
mod tests;
