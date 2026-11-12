# GENERATED FILE -- DO NOT EDIT.
#
# Written by `cargo run -p lk-agents-codegen -- --write` (in `rust/`) from the `events(..)`
# of `::lk_agents_abi::voice::rtc::RtcRoom` in rust/crates/lk-agents-abi/src/voice/rtc/capabilities.rs.

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from livekit import rtc as _rtc

from .. import _core

if TYPE_CHECKING:
    from livekit.agents.voice.room_io import Room


def subscribe(rtc_room: Room) -> None:
    """Deliver each SDK event `rtc_room.sdk` emits to its native entry."""
    from livekit.agents.voice import room_io as _room_io

    on = rtc_room.sdk.on
    bus = rtc_room.bus

    def track_subscribed(_track: Any, publication: Any, participant: Any) -> None:
        _core._core_room_track_subscribed(bus, _room_io.RemoteParticipant(rtc_room, participant), _room_io.TrackPublication(rtc_room, publication))

    on("track_subscribed", track_subscribed)

    def track_unsubscribed(_track: Any, publication: Any, participant: Any) -> None:
        _core._core_room_track_unsubscribed(bus, _room_io.RemoteParticipant(rtc_room, participant), _room_io.TrackPublication(rtc_room, publication))

    on("track_unsubscribed", track_unsubscribed)

    def track_published(publication: Any, participant: Any) -> None:
        _core._core_room_track_published(bus, _room_io.RemoteParticipant(rtc_room, participant), _room_io.TrackPublication(rtc_room, publication))

    on("track_published", track_published)

    def track_unpublished(publication: Any, participant: Any) -> None:
        _core._core_room_track_unpublished(bus, _room_io.RemoteParticipant(rtc_room, participant), _room_io.TrackPublication(rtc_room, publication))

    on("track_unpublished", track_unpublished)

    def local_track_published(publication: Any, _track: Any) -> None:
        _core._core_room_local_track_published(bus, _room_io.TrackPublication(rtc_room, publication))

    on("local_track_published", local_track_published)

    def participant_connected(participant: Any) -> None:
        _core._core_room_participant_connected(bus, _room_io.RemoteParticipant(rtc_room, participant))

    on("participant_connected", participant_connected)

    def participant_active(participant: Any) -> None:
        _core._core_room_participant_active(bus, _room_io.RemoteParticipant(rtc_room, participant))

    on("participant_active", participant_active)

    def participant_attributes_changed(_changed: Any, participant: Any) -> None:
        if not isinstance(participant, _rtc.RemoteParticipant):
            return
        _core._core_room_participant_attributes_changed(bus, _room_io.RemoteParticipant(rtc_room, participant))

    on("participant_attributes_changed", participant_attributes_changed)

    def participant_disconnected(participant: Any) -> None:
        _core._core_room_participant_disconnected(bus, _room_io.RemoteParticipant(rtc_room, participant), participant.disconnect_reason)

    on("participant_disconnected", participant_disconnected)

    def connection_state_changed(state: Any) -> None:
        _core._core_room_connection_state_changed(bus, state)

    on("connection_state_changed", connection_state_changed)

    def disconnected(reason: Any) -> None:
        _core._core_room_disconnected(bus, reason)

    on("disconnected", disconnected)

    def sip_dtmf_received(event: Any) -> None:
        participant_owner = event.participant
        _core._core_room_sip_dtmf_received(bus, event.digit, None if participant_owner is None else participant_owner.identity)

    on("sip_dtmf_received", sip_dtmf_received)
