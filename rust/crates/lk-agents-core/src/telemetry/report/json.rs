//! The report as JSON: the options it records and the console's `session_report.json`.

use std::io;

use lk_agents_abi::telemetry::report::{RecordingOptions, SessionReport};
use lk_agents_abi::voice::session_options::ResolvedSessionOptions;
use lk_agents_abi::{ToDictError, ToDictHost, ToDictOptions};
use serde::Serialize;
use serde_json::{Map, Value};

/// The options a report records: the resolved session options, with the recording options under
/// `recording_options`.
pub fn options(resolved: &ResolvedSessionOptions, recording: &RecordingOptions) -> Value {
    let mut options = match serde_json::to_value(resolved) {
        Ok(Value::Object(options)) => options,
        _ => Map::new(),
    };
    let recording = serde_json::to_value(recording).unwrap_or_default();
    options.insert("recording_options".to_owned(), recording);
    Value::Object(options)
}

/// The console's `session_report.json`.
pub fn console_json(report: &SessionReport) -> Result<String, ToDictError> {
    let dict = ToDictOptions { exclude_timestamp: false, ..ToDictOptions::default() };
    let chat_history =
        crate::llm::chat_dict::to_dict(&report.chat_history, &dict, &ToDictHost::default())?;
    let mut usage = serde_json::to_value(&report.usage).unwrap_or_default();
    strip_nulls(&mut usage);
    let data = serde_json::json!({
        "job_id": report.job_id,
        "room_id": report.room_id,
        "room": report.room,
        "events": report.events,
        "options": report.options,
        "timestamp": report.timestamp,
        "audio_recording_path": report.audio_recording_path,
        "audio_recording_started_at": report.audio_recording_started_at,
        "chat_history": chat_history,
        "usage": usage,
        "sdk_version": report.sdk_version,
    });
    let mut out = Vec::new();
    // Writing a `Value` to memory cannot fail.
    let _ = data.serialize(&mut serde_json::Serializer::with_formatter(&mut out, PythonJson));
    Ok(String::from_utf8(out).unwrap_or_default())
}

/// A model dump without its absent fields (`exclude_none=True`).
fn strip_nulls(value: &mut Value) {
    match value {
        Value::Object(fields) => {
            fields.retain(|_, v| !v.is_null());
            fields.values_mut().for_each(strip_nulls);
        }
        Value::Array(items) => items.iter_mut().for_each(strip_nulls),
        _ => {}
    }
}

/// Python's `json.dumps` defaults: `", "` and `": "` separators, non-ASCII escaped, floats as
/// `repr` writes them.
struct PythonJson;

impl serde_json::ser::Formatter for PythonJson {
    fn begin_array_value<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }

    fn begin_object_key<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        first: bool,
    ) -> io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }

    fn begin_object_value<W: ?Sized + io::Write>(&mut self, w: &mut W) -> io::Result<()> {
        w.write_all(b": ")
    }

    fn write_f64<W: ?Sized + io::Write>(&mut self, w: &mut W, value: f64) -> io::Result<()> {
        w.write_all(python_float(value).as_bytes())
    }

    fn write_string_fragment<W: ?Sized + io::Write>(
        &mut self,
        w: &mut W,
        fragment: &str,
    ) -> io::Result<()> {
        for c in fragment.chars() {
            if (' '..='~').contains(&c) {
                w.write_all(&[c as u8])?;
                continue;
            }
            for unit in c.encode_utf16(&mut [0; 2]) {
                write!(w, "\\u{unit:04x}")?;
            }
        }
        Ok(())
    }
}

/// Python's `repr(float)`: the shortest round-trip digits, positional when the decimal point
/// falls within `(-4, 16]` digits of the first one, else `<d>[.<ddd>]e<sign><two or more digits>`.
fn python_float(value: f64) -> String {
    let sci = format!("{value:e}");
    let (mantissa, exponent) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let (sign, mantissa) = match mantissa.strip_prefix('-') {
        Some(m) => ("-", m),
        None => ("", mantissa),
    };
    let digits = mantissa.replace('.', "");
    let point = exponent + 1;
    let body = if -4 < point && point <= 16 {
        if point <= 0 {
            format!("0.{}{digits}", "0".repeat(point.unsigned_abs() as usize))
        } else if point as usize >= digits.len() {
            format!("{digits}{}.0", "0".repeat(point as usize - digits.len()))
        } else {
            let (int, frac) = digits.split_at(point as usize);
            format!("{int}.{frac}")
        }
    } else {
        let (first, rest) = digits.split_at(1);
        let rest = if rest.is_empty() { String::new() } else { format!(".{rest}") };
        let exp_sign = if exponent < 0 { '-' } else { '+' };
        format!("{first}{rest}e{exp_sign}{:02}", exponent.unsigned_abs())
    };
    format!("{sign}{body}")
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: a float written otherwise than Python's `repr` (the console file is diffed).
    #[test]
    fn floats_are_written_as_python_repr_writes_them() {
        let cases = [
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (0.4, "0.4"),
            (2.5, "2.5"),
            (1_730_000_000.123_456, "1730000000.123456"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.5e17, "1.5e+17"),
            (0.0001, "0.0001"),
            (0.00001, "1e-05"),
            (-1.25e-7, "-1.25e-07"),
            (1e100, "1e+100"),
            (0.1 + 0.2, "0.30000000000000004"),
        ];
        for (value, repr) in cases {
            assert_eq!(python_float(value), repr, "{value}");
        }
    }

    // Catches: separators or string escaping that differ from `json.dumps(data)`.
    #[test]
    fn json_is_written_as_python_json_dumps_writes_it() {
        let value = serde_json::json!({"a": [1, 2.0, null, true], "s": "h\u{e9}llo \u{2603} \u{1f600} \"q\"\n\u{7f}", "o": {}});
        let mut out = Vec::new();
        value.serialize(&mut serde_json::Serializer::with_formatter(&mut out, PythonJson)).unwrap();
        assert_eq!(
            String::from_utf8(out).unwrap(),
            r#"{"a": [1, 2.0, null, true], "s": "h\u00e9llo \u2603 \ud83d\ude00 \"q\"\n\u007f", "o": {}}"#
        );
    }

    // Catches: a report option other than the resolved options' own keys, with the recording
    // options last; a group's unset optional key written as null instead of left out.
    #[test]
    fn options_are_the_resolved_options_then_the_recording_options() {
        use lk_agents_abi::voice::session_options as o;
        let resolved = o::ResolvedSessionOptions {
            endpointing: o::ResolvedEndpointing {
                mode: o::EndpointingMode::Fixed,
                min_delay: 0.5,
                max_delay: 3.0,
                alpha: 0.9,
            },
            interruption: o::ResolvedInterruption {
                enabled: true,
                mode: None,
                discard_audio_if_uninterruptible: true,
                min_duration: 0.5,
                min_words: 0,
                resume_false_interruption: true,
                false_interruption_timeout: Some(2.0),
                backchannel_boundary: Some((1.0, 1.0)),
            },
            preemptive_generation: o::ResolvedPreemptiveGeneration {
                enabled: true,
                preemptive_tts: false,
                max_speech_duration: 10.0,
                max_retries: 3,
            },
            user_turn_limit: o::ResolvedUserTurnLimit { max_words: None, max_duration: None },
            turn_detection_mode: None,
            endpointing_overrides: o::EndpointingOverrides::default(),
            max_tool_steps: 3,
            min_consecutive_speech_delay: 0.0,
            use_tts_aligned_transcript: None,
            tts_text_transforms: None,
            ivr_detection: false,
            user_away_timeout: Some(15.0),
            transcription_timeout: None,
            aec_warmup_duration: Some(3.0),
            session_close_transcript_timeout: 2.0,
        };
        let recording = RecordingOptions { audio: false, ..RecordingOptions::default() };
        let options = options(&resolved, &recording);
        let keys: Vec<&str> = options.as_object().unwrap().keys().map(String::as_str).collect();
        assert_eq!(keys.first(), Some(&"endpointing"));
        assert_eq!(keys.last(), Some(&"recording_options"));
        assert_eq!(
            options["recording_options"],
            serde_json::json!({"audio": false, "traces": true, "logs": true, "transcript": true, "redaction": false})
        );
        assert!(options["interruption"].get("mode").is_none());
        assert_eq!(options["endpointing_overrides"], serde_json::json!({}));
        assert!(options["user_turn_limit"]["max_words"].is_null());
        assert_eq!(options["interruption"]["backchannel_boundary"], serde_json::json!([1.0, 1.0]));
    }
}
