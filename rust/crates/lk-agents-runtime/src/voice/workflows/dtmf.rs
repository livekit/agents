//! VoiceRuntime publication and collection of room keypad tones.

use crate::telemetry;
use lk_agents_voice::workflows::dtmf::{
    DtmfReception, DtmfSendError, ReceiveDtmfError, dtmf_code, dtmf_send_answer,
};
use std::sync::Arc;
use std::time::Duration;

/// The pause between two published keypad tones.
pub const DTMF_PUBLISH_DELAY: Duration = Duration::from_millis(300);

/// `send_dtmf_events(events)`: each key published as a keypad tone in `room`, 0.3 s apart; the
/// answer to the model is [`dtmf_send_answer`]'s.
pub async fn send_dtmf_events(
    room: Arc<dyn lk_agents_abi::voice::room_owner::SessionRoom>,
    events: Vec<String>,
    clock: std::sync::Arc<dyn lk_agents_abi::Clock>,
) -> String {
    let outcome = async {
        let room = room.rtc().ok_or(DtmfSendError::NotConnected)?;
        for event in &events {
            let code = dtmf_code(event)
                .map_err(|error| DtmfSendError::Key { event: event.clone(), error })?;
            room.local()
                .publish_dtmf(&telemetry::Cx::NONE, u32::from(code), event.clone())
                .await
                .map_err(|error| DtmfSendError::refused(event, error))?;
            clock.wait_until(clock.now().saturating_add(DTMF_PUBLISH_DELAY)).await;
        }
        Ok(())
    }
    .await;
    dtmf_send_answer(&events, outcome)
}

/// `receive_dtmf(room, num_digits, timeout, stop_event)`: the keys a SIP caller presses in
/// `room`, until `num_digits` of them, the stop key, or `timeout` seconds without a key once the
/// first is in. Fewer than `num_digits` is [`ReceiveDtmfError::Incomplete`].
pub async fn receive_dtmf(
    room: Arc<dyn lk_agents_abi::voice::room_owner::SessionRoom>,
    num_digits: usize,
    timeout: f64,
    stop_event: String,
    clock: std::sync::Arc<dyn lk_agents_abi::Clock>,
) -> Result<String, ReceiveDtmfError> {
    let mut reception = DtmfReception::new(num_digits, timeout, &stop_event)?;
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
        events: Vec<String>,
    ) -> String {
        let clock = room.clock();
        send_dtmf_events(room, events, clock).await
    }
    async fn receive_dtmf(
        room: Arc<dyn lk_agents_abi::voice::room_owner::SessionRoom>,
        num_digits: usize,
        timeout: f64,
        stop_event: String,
    ) -> Result<String, ReceiveDtmfError> {
        let clock = room.clock();
        receive_dtmf(room, num_digits, timeout, stop_event, clock).await
    }
}
