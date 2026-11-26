use crate::telemetry;
use lk_agents_abi::errors::HostErrorKind;
use lk_agents_abi::voice::rtc::RtcError;
use lk_agents_voice::workflows::machine::dtmf::{
    DtmfEvent, DtmfReception, ReceiveDtmfError, SendDtmfError, dtmf_event_to_code, dtmf_sent_answer,
};
use std::sync::Arc;
use std::time::Duration;

/// The pause between two published keypad tones.
pub const DTMF_PUBLISH_DELAY: Duration = Duration::from_millis(300);

/// `send_dtmf_events(events)`: each key published as a keypad tone in `room`, 0.3 s apart; the
/// answer to the model is [`dtmf_sent_answer`]'s. A room that is not connected, or a key it
/// refused, is [`SendDtmfError::Failed`], without the room's own error. A publish the host
/// cancelled is [`SendDtmfError::Cancelled`], which the host raises as a cancellation.
pub async fn send_dtmf_events(
    room: Arc<dyn lk_agents_abi::voice::room_owner::SessionRoom>,
    events: Vec<DtmfEvent>,
    clock: std::sync::Arc<dyn lk_agents_abi::Clock>,
) -> Result<String, SendDtmfError> {
    let room = room.rtc().ok_or(SendDtmfError::Failed)?;
    for &event in &events {
        let code = u32::from(dtmf_event_to_code(event));
        room.local()
            .publish_dtmf(&telemetry::Cx::NONE, code, event.as_str().to_owned())
            .await
            .map_err(|error| match error {
                RtcError::Host(error) if error.kind == HostErrorKind::Cancelled => {
                    SendDtmfError::Cancelled
                }
                _ => SendDtmfError::Failed,
            })?;
        clock.wait_until(clock.now().saturating_add(DTMF_PUBLISH_DELAY)).await;
    }
    Ok(dtmf_sent_answer(&events))
}

/// `receive_dtmf(room, num_digits, timeout, stop_event)`: the keys a SIP caller presses in
/// `room`, until `num_digits` of them, the stop key, or `timeout` seconds without a key once the
/// first is in. Fewer than `num_digits` is [`ReceiveDtmfError::Incomplete`].
pub async fn receive_dtmf(
    room: Arc<dyn lk_agents_abi::voice::room_owner::SessionRoom>,
    num_digits: usize,
    timeout: f64,
    stop_event: DtmfEvent,
    clock: std::sync::Arc<dyn lk_agents_abi::Clock>,
) -> Result<String, ReceiveDtmfError> {
    let mut reception = DtmfReception::new(num_digits, timeout, stop_event)?;
    let room = room.rtc().ok_or_else(|| ReceiveDtmfError::Room("room is not connected".into()))?;
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
    async fn send_dtmf_events(
        room: Arc<dyn lk_agents_abi::voice::room_owner::SessionRoom>,
        events: Vec<DtmfEvent>,
    ) -> Result<String, SendDtmfError> {
        let clock = room.clock();
        send_dtmf_events(room, events, clock).await
    }
    async fn receive_dtmf(
        room: Arc<dyn lk_agents_abi::voice::room_owner::SessionRoom>,
        num_digits: usize,
        timeout: f64,
        stop_event: DtmfEvent,
    ) -> Result<String, ReceiveDtmfError> {
        let clock = room.clock();
        receive_dtmf(room, num_digits, timeout, stop_event, clock).await
    }
}
