//! Supplied ports for the session kernel's media-off test fixtures.

use crate::agent_session::Timebase;
use lk_agents_abi::voice::room_io::RoomIoConfig;
use lk_agents_abi::voice::room_owner::{RoomIo, SessionRoom};
use lk_agents_abi::voice::rtc::RtcRoom;
use lk_agents_abi::{Clock, Spawner, VirtualClock};
use std::sync::Arc;

pub struct FixedRoom {
    pub rtc: Arc<dyn RtcRoom>,
    pub time: Timebase,
}

impl SessionRoom for FixedRoom {
    fn rtc(&self) -> Option<Arc<dyn RtcRoom>> {
        Some(self.rtc.clone())
    }
    fn clock(&self) -> Arc<dyn Clock> {
        self.time.clock()
    }
    fn virtual_clock(&self) -> Option<Arc<VirtualClock>> {
        self.time.virtual_clock()
    }
    fn room_io(&self, _: Arc<dyn Spawner>, _: RoomIoConfig) -> Arc<dyn RoomIo> {
        crate::room_io::detached::room_io()
    }
}

/// A room event fan-out nobody reads: these fixtures' rooms are linked to no room I/O.
pub fn unread_room_events() -> lk_agents_abi::voice::rtc::RoomEventBus {
    struct Unread;
    impl lk_agents_abi::voice::rtc::RoomEventSink for Unread {
        fn subscribe(
            &self,
        ) -> lk_agents_abi::voice::rtc::Receiver<lk_agents_abi::voice::rtc::RoomEvent> {
            unreachable!("no room I/O subscribes in a media-off fixture")
        }
        fn emit(&self, _: lk_agents_abi::voice::rtc::RoomEvent) -> usize {
            0
        }
        fn subscriber_count(&self) -> usize {
            0
        }
    }
    lk_agents_abi::voice::rtc::RoomEventBus(Arc::new(Unread))
}
