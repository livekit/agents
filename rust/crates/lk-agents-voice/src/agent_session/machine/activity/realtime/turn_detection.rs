//! Reconcile client and realtime-server turn ownership.

use crate::agent_session::options::TurnDetection;
use crate::audio_recognition::TurnDetectionMode;
use lk_agents_abi::llm::realtime::RealtimeCapabilities;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionSide {
    pub stt: bool,
    pub vad: bool,
    pub detection: Option<TurnDetection>,
    pub streaming_detector: bool,
    pub interruption_given: bool,
    pub turn_detection_mode: TurnDetectionMode,
}

pub fn server_turn_detection_enabled(caps: &RealtimeCapabilities, side: SessionSide) -> bool {
    if !caps.turn_detection || !caps.can_disable_turn_detection {
        return caps.turn_detection;
    }
    match side.detection {
        Some(TurnDetection::RealtimeLlm) => return true,
        Some(TurnDetection::Manual) => return false,
        Some(TurnDetection::Vad | TurnDetection::Stt) | None => {}
    }
    let client_side = side.detection == Some(TurnDetection::Vad)
        || side.streaming_detector
        || side.interruption_given;
    !(side.vad && client_side)
}

pub fn turn_detection_disabled(caps: &RealtimeCapabilities, side: SessionSide) -> bool {
    !server_turn_detection_enabled(caps, side) && caps.can_disable_turn_detection
}

pub fn realtime_turn_detection_mode(
    caps: &RealtimeCapabilities,
    side: SessionSide,
    disabled: bool,
) -> TurnDetectionMode {
    if caps.turn_detection && caps.auto_turn_reply_generation && !disabled {
        return TurnDetectionMode::RealtimeLlm;
    }
    match side.turn_detection_mode {
        TurnDetectionMode::Manual => TurnDetectionMode::Manual,
        TurnDetectionMode::Vad | TurnDetectionMode::Stt | TurnDetectionMode::RealtimeLlm => {
            if side.vad || side.stt { TurnDetectionMode::Vad } else { TurnDetectionMode::Manual }
        }
    }
}
