//! `_core.FakeRoom` (a whole turn on a virtual clock) and `_core.OfflineRoom` (what `lk agent
//! console` / `daemon` pass as `room`). Both are the Rust room itself where an `Arc<dyn RtcRoom>`
//! is taken (`as_dyn`), never proxied back through Python.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use lk_agents_abi::VirtualClock;
use lk_agents_rtc::{
    DisconnectReason, FakeRoom, FakeRtc, FakeTrackPublication, FrameSynth, ParticipantKind, RtcRoom,
    StreamInfo, UnknownDisconnectReason,
};

#[cfg(feature = "python")]
use crate::agent_session::SessionRoom;
use crate::agent_session::Timebase;

/// 20 ms, the frame the room's own tests push. Not configurable: everything downstream rebuffers.
const FRAME_MS: u32 = 20;

/// What the two rooms refuse, as the `ValueError`s a caller sees.
#[derive(Debug, thiserror::Error)]
#[lk_agents_abi::host(error(kind = "ValueError"))]
pub enum RoomRefused {
    #[error("sample_rate must be a non-zero multiple of 1000 so a 20 ms frame is whole")]
    SampleRate,
    #[error("nobody has joined this FakeRoom yet -- call join() before speak()")]
    NobodyJoined,
    #[error("advance() takes a finite, non-negative number")]
    Advance,
    #[error("an OfflineRoom needs a name")]
    NoName,
    #[error(transparent)]
    Reason(#[from] UnknownDisconnectReason),
}

/// A room, a clock, and a microphone -- no network. The session takes the room's `VirtualClock`
/// as its `Timebase`, so driver timers and the playout model cannot drift. A test drives it from
/// the loop thread while the driver reads it from a tokio worker.
#[lk_agents_abi::class(name = "FakeRoom")]
pub struct ScriptedRoom {
    rtc: Arc<FakeRtc>,
    room: Arc<FakeRoom>,
    clock: Arc<VirtualClock>,
    /// The microphone of the last participant to join, which [`speak`](Self::speak) pushes into.
    mic: Mutex<Option<Arc<FakeTrackPublication>>>,
    /// The microphone's audio: distinct speech per frame.
    synth: FrameSynth,
}

#[lk_agents_abi::host(object(name = "FakeRoom", as_dyn(RtcRoom = room, SessionRoom = session_room)))]
impl ScriptedRoom {
    /// `sample_rate` is what the room carries; 48 kHz is what a real WebRTC track publishes.
    #[host(new(defaults(name = "fake".to_owned(), sample_rate = 48_000)))]
    pub fn new(name: String, sample_rate: u32) -> Result<Self, RoomRefused> {
        if sample_rate == 0 || !sample_rate.is_multiple_of(1_000) {
            return Err(RoomRefused::SampleRate);
        }
        let (rtc, clock) = FakeRtc::with_virtual_clock();
        let room = rtc.room(&name);
        Ok(ScriptedRoom { rtc, room, clock, mic: Mutex::new(None), synth: FrameSynth::new(sample_rate, FRAME_MS) })
    }

    /// The room, as the seam sees it.
    pub fn room(&self) -> Arc<dyn RtcRoom> {
        self.room.clone()
    }

    /// The session's clock, shared rather than copied, so timers and playout agree.
    pub fn timebase(&self) -> Timebase {
        Timebase::from_virtual(Arc::clone(&self.clock))
    }

    /// The room and its clock, as a session is built on them.
    pub fn session_room(&self) -> Arc<dyn crate::agent_session::SessionRoom> {
        Arc::new(crate::agent_session::FixedRoom { rtc: self.room(), time: self.timebase() })
    }

    /// The room's virtual clock as the handle a provider's `_bind_session_clock` takes.
    #[host(getter)]
    pub fn session_clock(&self) -> Timebase {
        self.timebase()
    }

    #[host(getter)]
    pub fn name(&self) -> String {
        self.room.name()
    }

    /// Add a remote participant and publish their microphone.
    #[host(method(defaults(identity = "user".to_owned())))]
    pub fn join(&self, identity: String, track_sid: Option<String>) {
        let sid = track_sid.unwrap_or_else(|| format!("TR_{identity}"));
        let participant = self.room.add_participant(identity.as_str(), ParticipantKind::Standard);
        *self.mic.lock().unwrap_or_else(|p| p.into_inner()) = Some(participant.publish_microphone(sid));
    }

    /// Push `frames` of 20 ms of speech from the joined participant. Returns how many a reader
    /// got; zero means nobody was subscribed and the audio is gone.
    #[host(method(defaults(frames = 1)))]
    pub fn speak(&self, frames: usize) -> Result<usize, RoomRefused> {
        let mic = self.mic.lock().unwrap_or_else(|p| p.into_inner()).clone();
        let mic = mic.ok_or(RoomRefused::NobodyJoined)?;
        Ok(mic.push_frames((0..frames).map(|_| self.synth.frame(false))))
    }

    /// Move the clock, settle the playout model, and wake the driver's timers. A deadline inside a
    /// jump fires at its end, so step when order matters.
    #[host(method)]
    pub fn advance(&self, seconds: f64) -> Result<(), RoomRefused> {
        if !(seconds.is_finite() && seconds >= 0.0) {
            return Err(RoomRefused::Advance);
        }
        self.clock.advance(Duration::from_secs_f64(seconds));
        self.rtc.settle();
        Ok(())
    }

    /// Bytes that actually reached the speaker.
    #[host(method)]
    pub fn played_bytes(&self) -> usize {
        self.room.local_participant().played_bytes()
    }

    /// Bytes cut by an interruption. Non-zero is the barge-in signal.
    #[host(method)]
    pub fn discarded_bytes(&self) -> usize {
        self.room.local_participant().discarded_bytes()
    }

    /// Every text stream the agent opened, as `(topic, text)`. The transcript path's output.
    #[host(method)]
    pub fn text_streams(&self) -> Vec<(String, String)> {
        self.room.local_participant().text_streams().into_iter().map(|s| (s.opts.topic.clone(), s.text())).collect()
    }

    /// `identity` sends `text` on `lk.chat`, as a stream with id `stream_id`. `False` if nobody
    /// handles the topic.
    #[host(method)]
    pub fn send_chat(&self, identity: String, text: String, stream_id: String) -> bool {
        let info = StreamInfo {
            id: stream_id,
            topic: lk_agents_abi::TOPIC_CHAT.into(),
            mime_type: "text/plain".into(),
            ..StreamInfo::default()
        };
        self.room.send_text_stream(identity, info, [text]).is_ok()
    }

    /// One participant leaves. `reason` is a `DisconnectReason` name, lower-cased with underscores.
    /// Unlike [`close`](Self::close), this is the event `close_on_disconnect` is about. Returns
    /// `False` if that identity was not in the room.
    #[host(method(defaults(reason = "client_initiated".to_owned())))]
    pub fn leave(&self, identity: String, reason: String) -> Result<bool, RoomRefused> {
        let reason: DisconnectReason = reason.parse()?;
        Ok(self.room.remove_participant(&identity.into(), reason))
    }

    /// Disconnect the room. `reason` is a `DisconnectReason` name, lower-cased with underscores.
    #[host(method(defaults(reason = "client_initiated".to_owned())))]
    pub fn close(&self, reason: String) -> Result<(), RoomRefused> {
        self.room.close(reason.parse()?);
        Ok(())
    }
}

/// A session with no RTC room, on the real clock. Built on [`FakeRtc::offline`] but on the wall
/// clock ([`Timebase::real`]), since a console session is a real conversation -- unless it is built
/// on an asyncio loop on virtual time, whose clock it then takes (the ambient clock).
#[lk_agents_abi::class]
pub struct OfflineRoom {
    /// Held so the room is not dropped: `FakeRtc` owns the room registry.
    _rtc: Arc<FakeRtc>,
    room: Arc<FakeRoom>,
    time: Timebase,
}

#[lk_agents_abi::host(object(as_dyn(RtcRoom = room, SessionRoom = session_room)))]
impl OfflineRoom {
    #[host(new(defaults(name = "console".to_owned())))]
    pub fn new(name: String) -> Result<Self, RoomRefused> {
        if name.is_empty() {
            return Err(RoomRefused::NoName);
        }
        // One clock, shared: the room gets the timebase's clock (`FakeRtc::offline` says why).
        #[cfg(feature = "python")]
        let time = Timebase::of_clock(<Arc<dyn lk_agents_abi::Clock> as lk_agents_abi::python::Ambient>::ambient());
        #[cfg(not(feature = "python"))]
        let time = Timebase::real();
        let (rtc, room) = FakeRtc::offline(&name, time.clock());
        Ok(OfflineRoom { _rtc: rtc, room, time })
    }

    /// The room, as the seam sees it.
    pub fn room(&self) -> Arc<dyn RtcRoom> {
        self.room.clone()
    }

    /// The session's clock: the **wall** clock, unlike `FakeRoom`'s.
    pub fn timebase(&self) -> Timebase {
        self.time.clone()
    }

    /// The room and its clock, as a session is built on them.
    pub fn session_room(&self) -> Arc<dyn crate::agent_session::SessionRoom> {
        Arc::new(crate::agent_session::FixedRoom { rtc: self.room(), time: self.timebase() })
    }

    #[host(getter)]
    pub fn name(&self) -> String {
        self.room.name()
    }
}
