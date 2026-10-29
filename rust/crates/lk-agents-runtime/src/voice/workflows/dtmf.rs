//! Runtime publication and collection of room keypad tones.

pub use lk_agents_voice::workflows::dtmf::DtmfResult;
use lk_agents_voice::workflows::dtmf::{
    DtmfEvent, DtmfReception, ReceiveDtmfError, dtmf_event_to_code,
};
use std::time::Duration;

/// The pause between two published keypad tones.
pub const DTMF_PUBLISH_DELAY: Duration = Duration::from_millis(300);

/// `send_dtmf_events(events)`: each key published as a keypad tone in `room`, 0.3 s apart; the
/// answer to the model names the keys sent, or the first that failed and why.
pub async fn send_dtmf_events(
    room: lk_agents_abi::voice::room_owner::RoomSession,
    events: Vec<String>,
    clock: std::sync::Arc<dyn lk_agents_abi::Clock>,
) -> String {
    let Some(room) = room.0.rtc() else {
        return "Failed to send DTMF events: room is not connected".into();
    };
    for event in &events {
        let sent = match DtmfEvent::parse(event) {
            Some(key) => room
                .local()
                .publish_dtmf(u32::from(dtmf_event_to_code(key)), event.clone())
                .await
                .map_err(|err| match err {
                    lk_agents_abi::voice::rtc::RtcError::Host(host) => host.message,
                    other => other.to_string(),
                }),
            None => Err(format!("unknown DTMF symbol {event:?}")),
        };
        if let Err(error) = sent {
            return format!("Failed to send DTMF event: {event}. Error: {error}");
        }
        clock.wait_until(clock.now().saturating_add(DTMF_PUBLISH_DELAY)).await;
    }
    format!("Successfully sent DTMF events: {}", events.join(", "))
}

/// `receive_dtmf(room, num_digits, timeout, stop_event)`: the keys a SIP caller presses in
/// `room`, until `num_digits` of them, the stop key, or `timeout` seconds without a key once the
/// first is in. Fewer than `num_digits` is [`ReceiveDtmfError::Incomplete`].
pub async fn receive_dtmf(
    room: lk_agents_abi::voice::room_owner::RoomSession,
    num_digits: usize,
    timeout: f64,
    stop_event: String,
    clock: std::sync::Arc<dyn lk_agents_abi::Clock>,
) -> Result<DtmfResult, ReceiveDtmfError> {
    let mut reception = DtmfReception::new(num_digits, timeout, &stop_event)?;
    let room =
        room.0.rtc().ok_or_else(|| ReceiveDtmfError::Room("room is not connected".into()))?;
    let mut events = room.events();
    loop {
        let next = match reception.deadline() {
            None => events.next().await,
            Some(at) => tokio::select! { biased;
                next = events.next() => next,
                _ = clock.wait_until(at) => break,
            },
        };
        let Some(event) = next.map_err(|error| ReceiveDtmfError::Room(error.to_string()))? else {
            break;
        };
        let lk_agents_abi::voice::rtc::RoomEvent::SipDtmf { digit, .. } = event else { continue };
        if reception.on_digit(&digit, clock.now())? {
            break;
        }
    }
    reception.finish()
}

/// The room capability supplies the same clock as its I/O.
pub struct DtmfIo;

impl lk_agents_abi::voice::workflows::dtmf::DtmfIo for DtmfIo {
    type Result = DtmfResult;
    async fn send_dtmf_events(
        room: lk_agents_abi::voice::room_owner::RoomSession,
        events: Vec<String>,
    ) -> String {
        let clock = room.0.clock();
        send_dtmf_events(room, events, clock).await
    }
    async fn receive_dtmf(
        room: lk_agents_abi::voice::room_owner::RoomSession,
        num_digits: usize,
        timeout: f64,
        stop_event: String,
    ) -> Result<DtmfResult, ReceiveDtmfError> {
        let clock = room.0.clock();
        receive_dtmf(room, num_digits, timeout, stop_event, clock).await
    }
}
