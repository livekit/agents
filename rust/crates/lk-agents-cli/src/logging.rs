pub use lk_agents_abi::cli::{LogRecord, LogStyle};
use serde_json::Value;
use std::io;

pub fn format_log_record(record: LogRecord, style: LogStyle) -> String {
    match style {
        LogStyle::Json => json_line(record),
        LogStyle::Dev => dev_line(record),
        LogStyle::Console => format!("\r{}", dev_line(record)),
    }
}

fn json_line(record: LogRecord) -> String {
    let mut fields = record.fields;
    fields.insert("message".into(), record.message.into());
    fields.insert("level".into(), record.level.into());
    fields.insert("name".into(), record.logger.into());
    fields.insert("pid".into(), record.process_id.map_or(Value::Null, Value::from));
    fields.insert("timestamp".into(), timestamp(record.timestamp).into());
    if let Some(exception) = record.exception {
        fields.insert("exception".into(), exception.into());
    }
    if let Some(stack) = record.stack {
        fields.insert("stack".into(), stack.into());
    }
    Value::Object(fields).to_string()
}

const RESET: &str = "\x1b[0m";
const GRAY: &str = "\x1b[90m";

/// `10:12:03 INFO  livekit.agents registered worker {"id": "AW_x"}`, then the tracebacks.
fn dev_line(record: LogRecord) -> String {
    let color = match record.level.as_str() {
        "TRACE" | "DEBUG" => "\x1b[36m",
        "INFO" => "\x1b[32m",
        "WARNING" => "\x1b[33m",
        "ERROR" => "\x1b[31m",
        "CRITICAL" => "\x1b[1;31m",
        _ => "",
    };
    let time = chrono::DateTime::from_timestamp_micros((record.timestamp * 1e6).round() as i64)
        .map(|at| at.with_timezone(&chrono::Local).format("%H:%M:%S").to_string())
        .unwrap_or_default();
    let mut line =
        format!("{time} {color}{:<5}{RESET} {} {}", record.level, record.logger, record.message);
    if !record.fields.is_empty() {
        line.push_str(&format!(" {GRAY}{}{RESET}", spaced_json(&Value::Object(record.fields))));
    }
    for trace in [record.exception, record.stack].into_iter().flatten() {
        line.push('\n');
        line.push_str(trace.trim_end());
    }
    line
}

/// JSON with `", "` and `": "` separators, as a person reads it.
fn spaced_json(value: &Value) -> String {
    struct Spaced;
    impl serde_json::ser::Formatter for Spaced {
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
    }
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, Spaced);
    serde::Serialize::serialize(value, &mut ser).expect("a JSON value serializes into memory");
    String::from_utf8(out).expect("serde_json writes UTF-8")
}

fn timestamp(seconds: f64) -> String {
    let micros = (seconds * 1e6).round() as i64;
    chrono::DateTime::from_timestamp_micros(micros)
        .map(|at| at.to_rfc3339_opts(chrono::SecondsFormat::Micros, true))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record() -> LogRecord {
        LogRecord {
            message: "héllo 🗣️".into(),
            level: "ERROR".into(),
            logger: "application".into(),
            exception: Some("failure\n".into()),
            fields: serde_json::Map::from_iter([("job_id".into(), "AJ_1".into())]),
            ..Default::default()
        }
    }

    #[test]
    fn json_logs_preserve_structured_fields_and_unicode() {
        let line = format_log_record(record(), LogStyle::Json);
        let fields: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(fields["message"], "héllo 🗣️");
        assert_eq!(fields["job_id"], "AJ_1");
        assert_eq!(fields["exception"], "failure\n");
        assert!(!line.contains('\u{1b}'));
    }

    // Catches: dev logs regressing to JSON, or losing the fields or the traceback.
    #[test]
    fn dev_logs_are_one_readable_colored_line_then_the_traceback() {
        let line = format_log_record(record(), LogStyle::Dev);
        let (head, trace) = line.split_once('\n').unwrap();
        assert!(head.contains("\x1b[31mERROR\x1b[0m application héllo 🗣️"), "{head}");
        assert!(head.ends_with(&format!("{GRAY}{{\"job_id\": \"AJ_1\"}}{RESET}")), "{head}");
        assert_eq!(trace, "failure");
        assert!(format_log_record(record(), LogStyle::Console).starts_with('\r'));
    }
}
