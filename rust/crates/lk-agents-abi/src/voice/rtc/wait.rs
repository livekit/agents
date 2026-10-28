//! Host wait entry points forward through an explicitly supplied room performer.

use super::{ParticipantIdentity, RtcError, TrackSid};
use crate::voice::room_owner::RoomSession;

#[crate::host(export)]
pub async fn wait_for_participant(
    room: RoomSession,
    identity: Option<String>,
    kinds: Option<Vec<i32>>,
    include_local: bool,
) -> Result<Option<ParticipantIdentity>, RtcError> {
    room.0
        .rtc()
        .ok_or(RtcError::NotConnected)?
        .wait_for_participant(identity, kinds, include_local)
        .await
}

#[crate::host(export)]
pub async fn wait_for_agent(
    room: RoomSession,
    name: Option<String>,
) -> Result<ParticipantIdentity, RtcError> {
    room.0.rtc().ok_or(RtcError::NotConnected)?.wait_for_agent(name).await
}

#[crate::host(export)]
pub async fn wait_for_participant_attribute(
    room: RoomSession,
    identity: String,
    attribute: String,
    value: String,
) -> Result<(), RtcError> {
    room.0
        .rtc()
        .ok_or(RtcError::NotConnected)?
        .wait_for_participant_attribute(identity, attribute, value)
        .await
}

#[crate::host(export)]
pub async fn wait_for_track_publication(
    room: RoomSession,
    identity: Option<String>,
    kinds: Option<Vec<i32>>,
    include_local: bool,
    subscribed: bool,
) -> Result<(Option<ParticipantIdentity>, TrackSid), RtcError> {
    room.0
        .rtc()
        .ok_or(RtcError::NotConnected)?
        .wait_for_track_publication(identity, kinds, include_local, subscribed)
        .await
}
