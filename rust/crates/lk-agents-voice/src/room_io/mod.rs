pub mod coordinator;
pub mod detached;
pub mod ingress;
pub mod input_state;
pub mod options;
pub mod output;
pub mod playout;
pub mod pre_connect_state;
pub mod publication;
pub mod transcription;
pub mod video_sampler;
pub mod wait;

use crate::agent_session::effects::{Out, StreamKind};
use crate::agent_session::{Close, Driver, HostEvent};
use crate::trace::{RoomLinkStep, RoomStep, Step};
use lk_agents_abi::Ts;
use lk_agents_abi::voice::events::CloseReason;
pub use lk_agents_abi::voice::room_io::{
    AudioInputConfig, AudioOutputConfig, DeleteRoomRequested, ParticipantLinked,
    ParticipantUnlinked, RoomIoConfig, TextInput, TextOutputConfig,
};
use lk_agents_abi::voice::rtc::{
    DisconnectReason, ParticipantIdentity, ParticipantKind, TrackSid, TrackSource,
};
use std::collections::BTreeMap;
use std::time::Duration;

/// `job.DEFAULT_PARTICIPANT_KINDS`: no `Agent` or `Egress`, so agents never auto-link to each other.
pub const DEFAULT_PARTICIPANT_KINDS: &[ParticipantKind] =
    &[ParticipantKind::Connector, ParticipantKind::Sip, ParticipantKind::Standard];

/// The track sources the audio input leg will read. Microphone only, as in the reference.
pub const AUDIO_INPUT_SOURCES: &[TrackSource] = &[TrackSource::Microphone];

/// `_pre_connect_audio.PRE_CONNECT_AUDIO_BUFFER_STREAM`.
pub const PRE_CONNECT_AUDIO_BUFFER_STREAM: &str = "lk.agent.pre-connect-audio-buffer";

/// `PreConnectAudioHandler._max_delta_s`: a buffer older than this is stale and dropped.
pub const PRE_CONNECT_MAX_DELTA: Duration = Duration::from_secs(1);

/// The kinds a [`RoomIoConfig`] auto-links: `participant_kinds`, or the defaults when empty.
pub fn accepted_kinds(config: &RoomIoConfig) -> &[ParticipantKind] {
    if config.participant_kinds.is_empty() {
        DEFAULT_PARTICIPANT_KINDS
    } else {
        &config.participant_kinds
    }
}

/// Whether the agent transcript follows audio playout: `sync_transcription` unset (`NOT_GIVEN`)
/// means yes when there is an audio output.
pub fn wants_transcript_sync(config: &RoomIoConfig, has_audio_output: bool) -> bool {
    let Some(text) = &config.text_output else { return false };
    text.sync_transcription != Some(false) && has_audio_output
}

/// `RecorderIO._write_task`'s `WRITE_INTERVAL`, on the session clock.
pub const RECORDER_FLUSH: Duration = Duration::from_millis(2500);

pub struct StreamOpened {
    pub open: u64,
    pub leg: StreamKind,
    pub outcome: std::result::Result<Opened, lk_agents_abi::errors::ProviderError>,
}

pub enum Opened {
    /// Runtime owns both sides until the core accepts or discards the prepared resource.
    Stt(lk_agents_abi::SttSink, lk_agents_abi::SttSource),
    Vad(lk_agents_abi::VadSink, lk_agents_abi::VadSource),
}

/// The legs an agent has after `reopen_legs`: given, or the table's row where none was given.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct InputProviders {
    pub(crate) stt: bool,
    pub(crate) vad: bool,
}

pub use lk_agents_abi::voice::room_owner::RoomIoEvent;

/// `RoomIO._on_participant_connected`. Not when someone is already linked, when an explicit target
/// is configured and this is not it, or when the candidate is an avatar publishing for the agent.
/// The publish-on-behalf filter applies only without an explicit target (the reference's `elif`).
pub fn decide_link(
    already_linked: bool,
    configured_target: Option<&ParticipantIdentity>,
    candidate: &ParticipantIdentity,
    candidate_kind: ParticipantKind,
    candidate_publishes_on_behalf_of: Option<&str>,
    agent_identity: &ParticipantIdentity,
    accepted_kinds: &[ParticipantKind],
) -> bool {
    let target_ok = match configured_target {
        Some(target) => target == candidate,
        None => candidate_publishes_on_behalf_of != Some(agent_identity.as_str()),
    };
    !already_linked && target_ok && accepted_kinds.contains(&candidate_kind)
}

pub fn publishes_on_behalf_of(attributes: &BTreeMap<String, String>) -> Option<&str> {
    attributes.get(lk_agents_abi::ATTRIBUTE_PUBLISH_ON_BEHALF).map(String::as_str)
}

/// A SIP participant no dispatch rule admitted (`sip.ruleID` unset): a call the agent dialed.
pub fn outbound_sip(
    kind: lk_agents_abi::voice::rtc::ParticipantKind,
    attributes: &BTreeMap<String, String>,
) -> bool {
    kind == lk_agents_abi::voice::rtc::ParticipantKind::Sip && !attributes.contains_key("sip.ruleID")
}

/// `_ParticipantInputStream._on_track_available`: not the track already read (reopening would
/// restart the STT mid-utterance), nor a screenshare on the microphone leg.
pub fn decide_track(
    linked: Option<&ParticipantIdentity>,
    currently_reading: Option<&TrackSid>,
    owner: &ParticipantIdentity,
    source: TrackSource,
    sid: &TrackSid,
    accepted_sources: &[TrackSource],
) -> bool {
    linked == Some(owner) && accepted_sources.contains(&source) && currently_reading != Some(sid)
}

/// `_ParticipantLegacyTranscriptionOutput._is_local_proxy_participant`: publishing on behalf of the
/// agent is a reason to accept (the caption belongs on the avatar's track).
pub fn is_local_proxy_participant(
    subject: Option<&ParticipantIdentity>,
    candidate: &ParticipantIdentity,
    candidate_publishes_on_behalf_of: Option<&str>,
) -> bool {
    let Some(subject) = subject else { return false };
    subject == candidate || candidate_publishes_on_behalf_of == Some(subject.as_str())
}

/// `RoomOptions`' two session-level switches and the room they apply to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoomLinkConfig {
    pub close_on_disconnect: bool,
    pub delete_room_on_close: bool,
    pub room: String,
}

pub(crate) struct RoomLink {
    config: RoomLinkConfig,
    /// A later disconnect closes nothing, a second close deletes nothing.
    deletion_requested: bool,
}

impl RoomLink {
    pub(crate) fn new(config: Option<RoomLinkConfig>) -> Self {
        let config = config.unwrap_or(RoomLinkConfig {
            close_on_disconnect: true,
            delete_room_on_close: false,
            room: String::new(),
        });
        RoomLink { config, deletion_requested: false }
    }

    /// Does this disconnect end the session? A deletion in flight is the end of the call already
    /// (`not self._delete_room_task`).
    fn disconnect_closes(&self, reason: DisconnectReason) -> bool {
        !self.deletion_requested
            && self.config.close_on_disconnect
            && reason.closes_session_by_default()
    }

    /// `delete_room_on_close`, once: the room to delete.
    fn request_deletion(&mut self) -> Option<String> {
        if self.deletion_requested || !self.config.delete_room_on_close {
            return None;
        }
        self.deletion_requested = true;
        Some(self.config.room.clone())
    }
}

impl Driver {
    pub(crate) fn on_room_io(&mut self, now: Ts, ev: RoomIoEvent, out: &mut Out) {
        let closing = self.session.closing_past_edge();
        match &ev {
            RoomIoEvent::ParticipantUnlinked(unlinked) => {
                out.machine().append(self.session.states.on_link(now, false, closing));
                let reason = unlinked.reason;
                // A deletion in flight is the end of the call already (`not self._delete_room_task`).
                let closes = self.session.room.disconnect_closes(reason);
                out.trace(RoomStep::Link(RoomLinkStep::Unlinked { reason, closes }));
                if closes {
                    let reason = CloseReason::ParticipantDisconnected;
                    self.begin_closing(now, Close { reason, drain: false, error: None }, out);
                }
            }
            RoomIoEvent::ParticipantLinked { outbound_sip, .. } => {
                out.trace(RoomStep::Link(RoomLinkStep::Linked));
                out.machine().append(self.session.states.on_link(now, true, closing));
                // `_on_room_io_participant_linked`: an outbound SIP call has no warmup, unless the
                // application set `aec_warmup_duration` itself.
                if *outbound_sip && !self.session.config.aec_warmup_given() {
                    let patch = crate::agent_activity::options::aec_warmup_layer(None);
                    let diff = self.session.config.configure(&patch);
                    let layer = crate::agent_activity::options::ConfigLayer::Session;
                    self.apply_config(now, layer, diff, out);
                }
            }
            RoomIoEvent::KeyPressed(digit) => {
                let key = crate::workflows::dtmf::DtmfEvent::parse(digit);
                out.trace(RoomStep::Link(RoomLinkStep::KeyPressed { key }));
                out.machine().append(self.session.states.reset_away_timer(now, closing));
                let digit = crate::workflows::dtmf::DtmfTaskEvent::Digit(digit.clone());
                return self.dtmf_step(digit, out);
            }
            RoomIoEvent::TextInput(_) | RoomIoEvent::DeleteRoomRequested(_) => {}
        }
        out.machine().push(crate::agent_session::effects::Effect::Host(HostEvent::Room(ev)));
    }

    pub(crate) fn on_session_closed(&mut self) {
        if let Some(room) = self.session.room.request_deletion() {
            self.journal(Step::Room(RoomStep::Link(RoomLinkStep::DeleteRoomRequested)));
            self.outbox.push(crate::agent_session::effects::Effect::Host(HostEvent::Room(
                RoomIoEvent::DeleteRoomRequested(DeleteRoomRequested { room }),
            )));
        }
    }
}
