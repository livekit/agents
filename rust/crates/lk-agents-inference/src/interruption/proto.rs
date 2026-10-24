//! The `/bargein` wire protocol: seven JSON messages plus one binary frame format. Pure.

use serde::{Deserialize, Serialize};

/// `InterruptionWSSessionCreateSettings` (`inference/interruption.py:696`), in the pydantic
/// model's field order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionCreateSettings {
    pub sample_rate: u32,
    pub num_channels: u16,
    /// Omitted when unset (`exclude_none=True`): `null` would mean "no threshold".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub threshold: Option<f64>,
    pub min_frames: u32,
    /// `Literal["s16le"]`, a field so the JSON keeps it in position.
    pub encoding: &'static str,
}

/// Little-endian signed 16-bit, what [`frame_request`] writes.
pub const ENCODING_S16LE: &str = "s16le";

/// What the client sends. Two messages; the audio goes over the binary channel.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum ClientMessage {
    #[serde(rename = "session.create")]
    SessionCreate { settings: SessionCreateSettings },
    #[serde(rename = "session.close")]
    SessionClose,
}

/// What the server sends. Echoed client messages are ignored variants; an unknown `type` is a
/// deserialisation error that takes the connection down, as in the reference.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(tag = "type")]
pub enum ServerMessage {
    #[serde(rename = "session.created")]
    SessionCreated {
        /// The server's recommended threshold; absent is fatal without a user threshold (#6034).
        #[serde(default)]
        default_threshold: Option<f64>,
    },
    #[serde(rename = "session.closed")]
    SessionClosed,
    #[serde(rename = "bargein_detected")]
    Detected(Verdict),
    /// "I looked and it is not a interruption (yet)."
    #[serde(rename = "inference_done")]
    InferenceDone(Verdict),
    #[serde(rename = "error")]
    Error { message: String, code: i64, session_id: String },
    /// The client messages, echoed; ignored with a warning.
    #[serde(rename = "session.create")]
    EchoedSessionCreate { settings: serde_json::Value },
    #[serde(rename = "session.close")]
    EchoedSessionClose,
}

#[derive(Debug, Clone, PartialEq, Default, Deserialize)]
pub struct Verdict {
    /// Echo of the 8-byte header the client stamped on the audio; the cache key.
    pub created_at: u64,
    #[serde(default)]
    pub prediction_duration: f64,
    #[serde(default)]
    pub probabilities: Vec<f32>,
}

/// `struct.pack("<Q", created_at) + audio_data.tobytes()`: the 8-byte LE header is the
/// correlation id the server echoes.
pub fn frame_request(created_at: u64, samples: &[i16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + samples.len() * 2);
    out.extend_from_slice(&created_at.to_le_bytes());
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: emitting `"threshold": null` when the user set none, or dropping a given one.
    #[test]
    fn session_create_carries_the_threshold_only_when_given() {
        let create = |threshold, min_frames| {
            serde_json::to_string(&ClientMessage::SessionCreate {
                settings: SessionCreateSettings {
                    sample_rate: 16_000,
                    num_channels: 1,
                    threshold,
                    min_frames,
                    encoding: ENCODING_S16LE,
                },
            })
            .unwrap()
        };
        assert_eq!(
            create(None, 2),
            r#"{"type":"session.create","settings":{"sample_rate":16000,"num_channels":1,"min_frames":2,"encoding":"s16le"}}"#
        );
        assert_eq!(
            create(Some(0.7), 4),
            r#"{"type":"session.create","settings":{"sample_rate":16000,"num_channels":1,"threshold":0.7,"min_frames":4,"encoding":"s16le"}}"#
        );
        assert_eq!(
            serde_json::to_string(&ClientMessage::SessionClose).unwrap(),
            r#"{"type":"session.close"}"#
        );
    }

    // Catches: making `prediction_duration`/`probabilities` required.
    #[test]
    fn a_verdict_without_optional_fields_parses_with_zeroed_defaults() {
        let msg: ServerMessage =
            serde_json::from_str(r#"{"type":"bargein_detected","created_at":42}"#).unwrap();
        assert_eq!(
            msg,
            ServerMessage::Detected(Verdict {
                created_at: 42,
                prediction_duration: 0.0,
                probabilities: vec![]
            })
        );
    }

    // Catches: reading `default_threshold` as required, or as 0.0 when absent.
    #[test]
    fn an_absent_default_threshold_stays_absent() {
        let msg: ServerMessage = serde_json::from_str(r#"{"type":"session.created"}"#).unwrap();
        assert_eq!(msg, ServerMessage::SessionCreated { default_threshold: None });
        let msg: ServerMessage =
            serde_json::from_str(r#"{"type":"session.created","default_threshold":0.0}"#).unwrap();
        assert_eq!(msg, ServerMessage::SessionCreated { default_threshold: Some(0.0) });
    }

    // Catches: swallowing an unknown message type.
    #[test]
    fn an_unknown_message_type_fails_to_parse_but_an_echo_does_not() {
        assert!(serde_json::from_str::<ServerMessage>(r#"{"type":"nope"}"#).is_err());
        assert!(serde_json::from_str::<ServerMessage>(r#"{"type":"session.close"}"#).is_ok());
    }

    // Catches: a big-endian header, or none.
    #[test]
    fn the_binary_frame_is_a_little_endian_u64_header_then_pcm() {
        assert_eq!(
            frame_request(0x0102_0304_0506_0708, &[1, -1]),
            vec![0x08, 0x07, 0x06, 0x05, 0x04, 0x03, 0x02, 0x01, 0x01, 0x00, 0xff, 0xff]
        );
    }
}
