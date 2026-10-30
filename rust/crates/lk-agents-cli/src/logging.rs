//! Structured process logs; presentation belongs to the remote CLI.

pub use lk_agents_abi::cli::LogRecord;
use serde_json::Value;

pub fn format_log_record(record: LogRecord) -> String {
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

fn timestamp(seconds: f64) -> String {
    let micros = (seconds * 1e6).round() as i64;
    chrono::DateTime::from_timestamp_micros(micros)
        .map(|at| at.to_rfc3339_opts(chrono::SecondsFormat::Micros, true))
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logs_preserve_structured_fields_and_unicode() {
        let record = LogRecord {
            message: "héllo 🗣️".into(),
            level: "ERROR".into(),
            logger: "application".into(),
            exception: Some("failure".into()),
            fields: serde_json::Map::from_iter([("job_id".into(), "AJ_1".into())]),
            ..Default::default()
        };
        let line = format_log_record(record);
        let fields: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(fields["message"], "héllo 🗣️");
        assert_eq!(fields["job_id"], "AJ_1");
        assert_eq!(fields["exception"], "failure");
        assert!(!line.contains('\u{1b}'));
    }
}
