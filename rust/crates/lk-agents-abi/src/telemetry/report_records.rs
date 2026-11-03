//! The session report's records, exported to Cloud as OTLP logs.

use super::decl::{Attrs, JsonObject, KvList, SpanRef, record};

/// One chat history item, linked to the span of the turn it belongs to.
#[record(scope = "chat_history", body = "chat item")]
#[derive(Attrs)]
pub struct ChatItemRecord {
    /// One of `message`, `function_call`, `function_call_output`, `agent_handoff`,
    /// `agent_config_update`, in the `livekit.agent.ChatContext.ChatItem` protobuf JSON form.
    #[attr("chat.item")]
    pub item: KvList<JsonObject>,
    #[span_id]
    pub turn: Option<SpanRef>,
}

/// The session summary, timestamped at the session start.
#[record(scope = "chat_history", body = "session report")]
#[derive(Attrs)]
pub struct SessionReportRecord {
    /// The resolved session options, recursively serialized; `instructions` omitted, `keyterms`
    /// as `lk.pii.keyterms`.
    #[attr("session.options")]
    pub options: KvList<JsonObject>,
    /// When the report was made, in unix seconds.
    #[attr("session.report_timestamp")]
    pub report_timestamp: f64,
    /// The session's tags, sorted.
    #[attr("session.tags")]
    pub tags: Vec<String>,
    /// The agent's name.
    #[attr("agent_name")]
    pub agent_name: String,
    /// The framework's version.
    #[attr("sdk_version")]
    pub sdk_version: String,
    /// The `Usage` model: one entry per model and kind.
    #[attr("usage")]
    pub usage: Vec<KvList<JsonObject>>,
}

/// One judge's evaluation.
#[record(scope = "evaluations", body = "evaluation")]
#[derive(Attrs)]
pub struct EvaluationRecord {
    /// `name`, `tag` (`lk.judge.<name>:<verdict>`), `verdict`, `reasoning`, `instructions`.
    #[attr("evaluation")]
    pub evaluation: KvList<JsonObject>,
}

/// One tag the application added.
#[record(scope = "evaluations", body = "tag")]
#[derive(Attrs)]
pub struct TagRecord {
    /// `name`, `metadata`.
    #[attr("tag")]
    pub tag: KvList<JsonObject>,
}

/// The session's outcome.
#[record(scope = "evaluations", body = "outcome")]
#[derive(Attrs)]
pub struct OutcomeRecord {
    /// `outcome` (`success`, `fail`), `reason`.
    #[attr("outcome")]
    pub outcome: KvList<JsonObject>,
}
