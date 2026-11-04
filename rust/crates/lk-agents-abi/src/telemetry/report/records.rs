//! The session report's records, exported to Cloud as OTLP logs.

use crate::telemetry::decl::{Attrs, JsonObject, KvList, SpanRef, record};
use crate::usage::ModelUsage;
use livekit_protocol::agent::chat_context::ChatItem;

use super::{EvaluationRecord, ReportOutcome, TagMetadata};

/// One chat history item, linked to the span of the turn it belongs to.
#[record(scope = "chat_history", body = "chat item")]
#[derive(Attrs)]
pub struct ChatItemLog {
    /// The `livekit.agent.ChatContext.ChatItem` protobuf JSON form: camelCase keys, defaults
    /// omitted.
    #[attr("chat.item")]
    pub item: KvList<ChatItem>,
    #[span_id]
    pub turn: Option<SpanRef>,
}

/// `session.options`' keyterms, under a key Cloud's ingest redacts.
pub const KEYTERMS: &str = "lk.pii.keyterms";

/// The session summary, timestamped at the session start.
#[record(scope = "chat_history", body = "session report")]
#[derive(Attrs)]
pub struct SessionReportLog {
    /// The resolved session options, recursively serialized; `instructions` omitted, `keyterms`
    /// as `lk.pii.keyterms`.
    #[attr("session.options")]
    pub options: KvList<JsonObject>,
    /// When the report was made, in unix seconds.
    #[attr("session.report_timestamp")]
    pub report_timestamp: f64,
    /// The session's tags, sorted; absent when there are none.
    #[attr("session.tags")]
    pub tags: Option<Vec<String>>,
    /// The agent's name.
    #[attr("agent_name")]
    pub agent_name: String,
    /// The framework's version.
    #[attr("sdk_version")]
    pub sdk_version: String,
    /// The `Usage` model's entries: one per model and kind; absent when nothing was used.
    #[attr("usage")]
    pub usage: Option<Vec<KvList<ModelUsage>>>,
}

/// One judge's evaluation.
#[record(scope = "evaluations", body = "evaluation")]
#[derive(Attrs)]
pub struct EvaluationLog {
    /// The judgment; its `tag` is `lk.judge.<name>:<verdict>`.
    #[attr("evaluation")]
    pub evaluation: KvList<EvaluationRecord>,
}

/// One tag the application added.
#[record(scope = "evaluations", body = "tag")]
#[derive(Attrs)]
pub struct TagLog {
    /// The tag and its metadata.
    #[attr("tag")]
    pub tag: KvList<TagMetadata>,
}

/// The session's outcome.
#[record(scope = "evaluations", body = "outcome")]
#[derive(Attrs)]
pub struct OutcomeLog {
    /// The outcome and its reason.
    #[attr("outcome")]
    pub outcome: KvList<ReportOutcome>,
}
