#![allow(
    clippy::disallowed_macros,
    reason = "Clippy checks macro bans at crate scope; exact legacy macro sites are ratcheted."
)]

//! `lk-agents-a2a`: the A2A text door's synchronous core. The records the host and the wire
//! read ([`Message`], [`TurnContext`], [`HandlerOutput`], [`TurnResult`]), the door's options and
//! their per-channel resolution ([`policy`]), and the decisions of a turn ([`outcome`]): which task
//! a message answers, what each task row becomes, the checkpoint verdict. The runtime
//! (`lk_agents_runtime::a2a`) runs the policies against storage and drives the host's handler.
//!
//! [`TurnHost::handle`] answers a stream of [`TurnEvent`]s: each [`Checkpoint`] answered with a
//! [`Verdict`], then the [`HandlerOutput`]. A turn yields at a checkpoint once a newer message
//! waits under `interrupt` or a client canceled its task.

pub mod auth;
pub mod attachments;
pub mod caller_history;
pub mod client;
pub mod delegation;
mod host;
pub mod outcome;
pub mod policy;
pub mod rpc;
pub mod task;

use lk_agents_abi::errors::ProviderError;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use livekit_protocol::agent::ChatMessage;
use livekit_protocol::agent::ChatRole;
use livekit_protocol::agent::chat_context::{ChatItem, chat_item};
use livekit_protocol::agent::chat_message::{ChatContent, chat_content::Payload};
use lk_agents_abi::storage::StorageError;
use lk_agents_abi::a2a::metadata::KIND_KEY;
use lk_agents_abi::storage::RunUsage;
use lk_agents_abi::storage::thread::{ItemRow, TaskRow};
use lk_agents_abi::storage::thread::timestamp;

pub use outcome::*;
pub use policy::*;

pub use lk_agents_abi::a2a::options::{Policy, TurnOptions};
pub use lk_agents_abi::a2a::turn::{TurnEvent, TurnHost, Verdict};

/// One incoming A2A message, as the handler sees it in `ctx.messages`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]

pub struct Message {
    pub message_id: String,
    /// v1 parts, as protobuf JSON.
    pub parts: Vec<a2a::Part>,

    pub metadata: Map<String, Value>,
    pub channel: String,
    /// Unix seconds.
    pub received_at: f64,
    pub task_id: Option<String>,
    pub context_id: Option<String>,
    /// The A2A message as received, less a delegation's `chat_ctx` part.
    pub raw: a2a::Message,
}

impl Message {
    /// A message without a `messageId` is invalid. `received_at` is Unix seconds.
    pub fn from_a2a(
        message: &a2a::Message,
        received_at: f64,
        task_id: Option<&str>,
    ) -> Result<Message, StorageError> {
        if message.message_id.is_empty() {
            return Err(StorageError::Invalid("an A2A message needs a messageId".into()));
        }
        attachments::validate_received_parts(message)?;
        // a delegation's `chat_ctx` part is the caller's conversation, read by `caller_history_of`
        // before the message is parsed; it is not part of the turn's text
        let mut raw = message.clone();
        raw.parts.retain(|p| !is_chat_ctx_part(p));
        let nonempty = |s: &Option<String>| s.clone().filter(|s| !s.is_empty());
        Ok(Message {
            message_id: message.message_id.clone(),
            parts: raw.parts.clone(),
            metadata: message.metadata.clone().unwrap_or_default().into_iter().collect(),
            channel: channel_of(message),
            received_at,
            task_id: task_id.map(str::to_owned).or_else(|| nonempty(&message.task_id)),
            context_id: nonempty(&message.context_id),
            raw,
        })
    }

    /// The text parts, joined by newlines.
    pub fn text(&self) -> String {
        self.parts.iter().filter_map(a2a::Part::as_text).collect::<Vec<_>>().join("\n")
    }

    /// The first data part's object.
    pub fn data(&self) -> Option<Map<String, Value>> {
        self.parts.iter().find_map(|p| match &p.content {
            a2a::PartContent::Data(Value::Object(o)) => Some(o.clone()),
            _ => None,
        })
    }

    /// The parts as chat content: text as is, data as compact JSON, a file as its part's JSON.
    pub fn content(&self) -> Vec<String> {
        self.parts
            .iter()
            .map(|p| match &p.content {
                a2a::PartContent::Text(t) => t.clone(),
                a2a::PartContent::Data(d) => d.to_string(),
                _ => serde_json::to_string(p).unwrap_or_default(),
            })
            .collect()
    }

    /// The message as received; a user's when the wire named no role.
    pub fn to_a2a(&self) -> a2a::Message {
        let mut out = self.raw.clone();
        if out.role == a2a::Role::Unspecified {
            out.role = a2a::Role::User;
        }
        out
    }
}

/// The batch as one user chat message `id` with every part in order.
pub fn batch_to_chat_message(messages: &[Message], created_at: f64, id: String) -> ItemRow {
    let mut content: Vec<String> = messages.iter().flat_map(Message::content).collect();
    if content.is_empty() {
        content.push(String::new());
    }
    chat_message(ChatRole::User, content, created_at, id)
}

/// A text chat message `id`.
pub fn chat_message(role: ChatRole, content: Vec<String>, created_at: f64, id: String) -> ItemRow {
    ItemRow::new(chat_item::Item::Message(ChatMessage {
        id,
        role: role as i32,
        content: content
            .into_iter()
            .map(|t| ChatContent { payload: Some(Payload::Text(t)) })
            .collect(),
        created_at: Some(timestamp(created_at)),
        ..Default::default()
    }))
}

/// Whether a part is a delegation's `chat_ctx` data part: `livekit/kind = chat_ctx` on the part.
fn is_chat_ctx_part(part: &a2a::Part) -> bool {
    matches!(part.content, a2a::PartContent::Data(_))
        && part.metadata.as_ref().and_then(|m| m.get(KIND_KEY)).and_then(Value::as_str)
            == Some("chat_ctx")
}

/// The `chat_ctx` data part of a delegation, as the part's `data`: the caller's
/// conversation, read from the A2A message before `Message::from_a2a` strips the part.
pub fn caller_history_of(message: &a2a::Message) -> Option<Value> {
    message.parts.iter().filter(|p| is_chat_ctx_part(p)).find_map(|p| match &p.content {
        a2a::PartContent::Data(d) if d.is_object() => Some(d.clone()),
        _ => None,
    })
}

/// A failure the host's stream raised: a cancellation ends the run `canceled`, a storage error
/// the handler met fails it as that error.
pub fn host_error(e: ProviderError) -> StorageError {
    let ProviderError::Host { error, .. } = &e else {
        return StorageError::Unavailable(e.to_string());
    };
    let message = error.message.clone();
    match error.host_kind() {
        "CancelledError" => StorageError::Cancelled(message),
        "ThreadNotFound" => StorageError::ThreadNotFound(message),
        "StorageUnavailable" => StorageError::Unavailable(message),
        kind => StorageError::Server { code: kind.to_owned(), message },
    }
}

/// What the handler is opened on: the message, the batch, the history, the state, the task.
#[derive(Debug, Clone, Serialize, Deserialize)]

pub struct TurnContext {
    /// The database id of the session this turn runs in.
    pub thread_id: String,
    /// The lane: the registered agent this turn belongs to.
    pub lane: String,
    /// The batch as one user chat message (`batch_to_chat_message`).
    pub message: ChatItem,
    /// The individual messages with their channel and time; `batch.last()` is the one answered.
    pub batch: Vec<Message>,
    /// The lane's history as loaded, plus the pending input-required answer when there was one.
    pub history: Vec<ItemRow>,
    /// The state as loaded; `None` for a fresh lane.
    pub state: Option<Value>,
    pub context_scope_id: String,
    pub context_scopes: Vec<lk_agents_abi::storage::context::ContextScope>,
    /// Context field writes require the lane's exclusive lease.
    #[serde(default)]
    pub read_only: bool,
    /// The task that carries the reply.
    pub task_id: String,
    /// The lane's task rows, newest first, up to 50.
    pub tasks: Vec<TaskRow>,
    /// A delegating caller's conversation when the message carried a `chat_ctx` data part, as the
    /// part's `data`.
    pub caller_history: Option<Value>,
    /// The tool the message is handed to before the model runs: an `InputRequired(tool=)` of
    /// the last turn named it, and this message is its answer.
    pub answer_tool: Option<String>,
}

/// A conversation tool's pending call, the one an `InputRequired` came from.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]

pub struct PendingCall {
    pub name: String,
    pub call_id: String,
    /// The tool that takes the answer, when the `InputRequired` named one.
    #[serde(default)]
    pub answer_tool: Option<String>,
}

/// A journal event the handler recorded (docs/internals/storage.md#journal): `tool_execution_updated` for a
/// call's start, its progress and its end.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]

pub struct HostEvent {
    #[serde(rename = "type")]
    pub r#type: String,
    pub payload: Value,
}

/// What the turn reads from the handler's results: the `RunResult` fields and the context's
/// metadata, as JSON. Every field has a default: a handler reports only what it has.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]

pub struct HandlerOutput {
    /// The answer: text, or a typed value as JSON.
    #[serde(default)]
    pub output: Option<Value>,
    /// The chat items the runs produced, in order, each with its `id` and `type`; empty when no
    /// run was made, in which case the turn stores the message and a text answer itself.
    #[serde(default)]
    pub new_items: Vec<ChatItem>,
    /// The state after the handler; written at commit when it differs from what was loaded.
    #[serde(default)]
    pub state: Option<Value>,
    #[serde(default)]
    pub scope_writes: Vec<lk_agents_abi::storage::context::ContextScopeWrite>,
    #[serde(default)]
    pub usage: RunUsage,
    /// A conversation tool's `InputRequired(question)`.
    #[serde(default)]
    pub input_required: Option<String>,
    /// The call that asked, when `input_required` is set.
    #[serde(default)]
    pub pending_call: Option<PendingCall>,
    /// The handler yielded at a checkpoint.
    #[serde(default)]
    pub interrupted: bool,
    /// What the handler put on the task's terminal row, under short keys.
    #[serde(default)]
    pub metadata: Map<String, Value>,
    /// The handler raised: `"<TypeName>: <text>"`; the task fails with it.
    #[serde(default)]
    pub error: Option<String>,
    /// Journal events the handler recorded, in order, written with the commit.
    #[serde(default)]
    pub events: Vec<HostEvent>,
}

/// What `Turn::run` returns: the message's task after the turn, and the reply.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]

pub struct TurnResult {
    pub task: TaskRow,
    /// The answer, as the artifact holds it: text, or the data of a typed value.
    pub output: Option<Value>,
    /// The database id of the session the turn ran in.
    pub thread_id: String,
    /// The task that carries the reply: this one, or the later one it was merged into.
    pub answered_by: String,
    pub run_id: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::a2a::options::LEASE_WAIT_TIMEOUT;
    use lk_agents_abi::a2a::metadata::{MERGED_INTO_KEY, REASON_KEY};
use lk_agents_abi::storage::TaskState;
    use serde_json::json;
    use std::time::Duration;

    #[test]
    fn options_default_to_the_documented_numbers() {
        let o = TurnOptions::default();
        assert_eq!(o.wait_timeout(), LEASE_WAIT_TIMEOUT);
        assert_eq!(o.policy, PolicySpec::Default);
        assert_eq!(
            serde_json::to_string(&interrupted_metadata()).unwrap(),
            r#"{"https://livekit.io/a2a/ext/agents/v1/interrupted":true,"https://livekit.io/a2a/ext/agents/v1/interruptionReason":"new-message"}"#
        );
    }

    // Catches: the options' wire shape drifting from what the host's `TurnOptions` model writes
    // (a policy as a string or a map, seconds as numbers, `null` for a default), or an unknown
    // policy accepted or reported without its name.
    #[test]
    fn options_cross_as_the_models_json() {
        let o: TurnOptions = serde_json::from_str("{}").unwrap();
        assert_eq!(o, TurnOptions::default());
        let json = r#"{"policy": {"sms": "batch"}, "debounce": 2.5, "wait_timeout": 1}"#;
        let o: TurnOptions = serde_json::from_str(json).unwrap();
        assert_eq!(
            o.policy,
            PolicySpec::ByChannel([("sms".to_owned(), Policy::Batch)].into_iter().collect())
        );
        assert_eq!(o.debounce, DebounceSpec::All(Duration::from_millis(2500)));
        assert_eq!(o.wait_timeout(), Duration::from_secs(1));
        let back: TurnOptions = serde_json::from_value(serde_json::to_value(&o).unwrap()).unwrap();
        assert_eq!(back, o);
        assert_eq!(
            serde_json::to_value(TurnOptions::default()).unwrap(),
            serde_json::json!({"policy": null, "debounce": null, "wait_timeout": null})
        );
        let e = serde_json::from_str::<TurnOptions>(r#"{"policy": "later"}"#).unwrap_err();
        assert!(e.to_string().contains("later"), "{e}");
        assert!(serde_json::from_str::<TurnOptions>(r#"{"debounce": -1}"#).is_err());
    }

    #[test]
    fn policy_and_debounce_resolution() {
        let none = PolicySpec::Default;
        assert_eq!(resolve_policy(&none, "sms"), Policy::Batch);
        assert_eq!(resolve_policy(&none, "a2a"), Policy::Interrupt);
        assert_eq!(resolve_policy(&none, "delegation"), Policy::Queue);
        assert_eq!(resolve_policy(&none, "carrier-pigeon"), Policy::Interrupt);
        let map = PolicySpec::ByChannel([("a2a".to_owned(), Policy::Queue)].into_iter().collect());
        assert_eq!(resolve_policy(&map, "a2a"), Policy::Queue);
        assert_eq!(resolve_policy(&map, "sms"), Policy::Batch);
        assert_eq!(resolve_policy(&PolicySpec::All(Policy::Reject), "sms"), Policy::Reject);
        assert_eq!(Policy::parse("later"), None);
        assert_eq!(Policy::parse("batch"), Some(Policy::Batch));
        let d = DebounceSpec::Default;
        assert_eq!(resolve_debounce(&d, "sms", Policy::Batch), Duration::from_secs(2));
        assert_eq!(resolve_debounce(&d, "a2a", Policy::Interrupt), Duration::ZERO);
        assert_eq!(resolve_debounce(&d, "pigeon", Policy::Batch), Duration::from_secs(2));
        assert_eq!(resolve_debounce(&d, "pigeon", Policy::Queue), Duration::ZERO);
        let one = DebounceSpec::All(Duration::from_secs(1));
        assert_eq!(resolve_debounce(&one, "a2a", Policy::Interrupt), Duration::from_secs(1));
        let per = DebounceSpec::ByChannel(
            [("sms".to_owned(), Duration::from_secs(5))].into_iter().collect(),
        );
        assert_eq!(resolve_debounce(&per, "sms", Policy::Batch), Duration::from_secs(5));
        assert_eq!(resolve_debounce(&per, "email", Policy::Batch), Duration::from_secs(2));
    }

    fn v1(message: Value) -> a2a::Message {
        serde_json::from_value(message).unwrap()
    }

    #[test]
    fn a_message_reads_its_channel_text_data_and_round_trips() {
        let raw = v1(json!({
            "messageId": "m1",
            "role": "ROLE_USER",
            "parts": [
                {"text": "hi"},
                {"data": {"b": 1, "a": 2}},
                {"url": "https://x/y.png", "mediaType": "image/png"},
                {"text": "there"},
            ],
            "metadata": {"https://livekit.io/a2a/ext/agents/v1/channel": "sms"},
            "taskId": "AST_1",
            "contextId": "AT_1",
        }));
        let m = Message::from_a2a(&raw, 5.0, None).unwrap();
        assert_eq!((m.message_id.as_str(), m.channel.as_str()), ("m1", "sms"));
        assert_eq!((m.task_id.as_deref(), m.context_id.as_deref()), (Some("AST_1"), Some("AT_1")));
        assert_eq!(m.text(), "hi\nthere");
        assert_eq!(m.data(), json!({"b": 1, "a": 2}).as_object().cloned());
        assert_eq!(
            m.content(),
            vec![
                "hi",
                r#"{"b":1,"a":2}"#,
                r#"{"url":"https://x/y.png","mediaType":"image/png"}"#,
                "there"
            ]
        );
        assert_eq!(m.to_a2a(), raw, "the message comes back as it was");
        assert_eq!(
            Message::from_a2a(&raw, 0.0, Some("AST_9")).unwrap().task_id.as_deref(),
            Some("AST_9")
        );

        let bare = v1(json!({"messageId": "m2", "role": "ROLE_UNSPECIFIED", "parts": []}));
        assert_eq!(Message::from_a2a(&bare, 0.0, None).unwrap().to_a2a().role, a2a::Role::User);
        assert!(
            Message::from_a2a(
                &v1(json!({"messageId": "", "role": "ROLE_USER", "parts": []})),
                0.0,
                None
            )
            .is_err()
        );
        let delegated = v1(json!({"messageId": "x", "role": "ROLE_USER", "parts": [],
            "metadata": {"https://livekit.io/a2a/ext/agents/v1/kind": "delegation"}}));
        assert_eq!(Message::from_a2a(&delegated, 0.0, None).unwrap().channel, "delegation");
        let with_history = v1(json!({"messageId": "y", "role": "ROLE_USER", "parts": [
            {"text": "fare?"},
            {"data": {"items": []}, "metadata": {"https://livekit.io/a2a/ext/agents/v1/kind": "chat_ctx"}},
        ]}));
        assert_eq!(caller_history_of(&with_history), Some(json!({"items": []})));
        let stripped = Message::from_a2a(&with_history, 1.0, None).unwrap();
        assert_eq!(stripped.parts, vec![a2a::Part::text("fare?")]);
        assert_eq!(stripped.to_a2a().parts, vec![a2a::Part::text("fare?")]);
    }

    // Catches: a sender's `livekit/<name>` key read as LiveKit's own (a2a.md#metadata-keys), e.g.
    // a direct message choosing the delegation policy or a chat_ctx part dropped as history.
    #[test]
    fn short_livekit_keys_on_the_wire_are_the_senders_own() {
        let raw = v1(json!({"messageId": "m", "role": "ROLE_USER",
            "metadata": {"livekit/channel": "sms", "livekit/kind": "delegation"},
            "parts": [{"data": {"items": []}, "metadata": {"livekit/kind": "chat_ctx"}}]}));
        let m = Message::from_a2a(&raw, 0.0, None).unwrap();
        assert_eq!(m.channel, A2A_CHANNEL);
        assert_eq!(caller_history_of(&raw), None);
        assert_eq!(m.parts, raw.parts, "the data part is the turn's, not a caller history");
    }

    #[test]
    fn the_batch_is_one_user_message_with_every_part_in_order() {
        let text = |id: &str, t: &str| {
            v1(json!({"messageId": id, "role": "ROLE_USER", "parts": [{"text": t}]}))
        };
        let a = Message::from_a2a(&text("m1", "where"), 1.0, None).unwrap();
        let b = Message::from_a2a(&text("m2", "are you"), 2.0, None).unwrap();
        let item = batch_to_chat_message(&[a, b], 2.0, "item_1".into());
        assert_eq!(item.id, "item_1");
        let Some(chat_item::Item::Message(m)) = item.kind() else { panic!("{item:?}") };
        assert_eq!(m.role(), ChatRole::User);
        let texts = |m: &ChatMessage| -> Vec<String> {
            m.content.iter().filter_map(|c| c.payload.clone().map(|Payload::Text(t)| t)).collect()
        };
        assert_eq!(texts(m), vec!["where", "are you"]);
        assert_eq!(item.created_at(), 2.0);
        let empty = batch_to_chat_message(&[], 0.0, "item_2".into());
        let Some(chat_item::Item::Message(m)) = empty.kind() else { panic!() };
        assert_eq!(texts(m), vec![""]);
    }

    fn item(id: &str, created_at: f64) -> ItemRow {
        chat_message(ChatRole::User, vec![], created_at, id.into())
    }

    #[test]
    fn insertion_follows_created_at_after_equal_times() {
        let mut h = vec![item("a", 1.0), item("b", 2.0), item("c", 2.0)];
        insert_by_created_at(&mut h, item("d", 2.0));
        insert_by_created_at(&mut h, item("e", 0.5));
        insert_by_created_at(&mut h, item("f", 9.0));
        assert_eq!(
            h.iter().map(|i| i.id.as_str()).collect::<Vec<_>>(),
            vec!["e", "a", "b", "c", "d", "f"]
        );
    }

    #[test]
    fn the_helpers_shape_as_the_python() {
        let f = failed_message("AST_1", "worker-lost", "msg_X".into());
        assert_eq!(
            serde_json::to_value(&f).unwrap(),
            json!({"messageId": "msg_X", "taskId": "AST_1", "role": "ROLE_AGENT",
                "parts": [{"text": "the task failed: worker-lost"}],
                "metadata": {"https://livekit.io/a2a/ext/agents/v1/reason": "worker-lost"}})
        );
        let failed =
            unpersisted_failed("AST_2", "front-desk", None, "lease-timeout", 1, "msg_Y".into());
        assert_eq!(failed.data.metadata.get(REASON_KEY), Some(&json!("lease-timeout")));
        assert_eq!(failed.status, TaskState::Failed);
        assert!(failed.data.history.is_empty());
        assert_eq!(artifact_text(&failed), None);
        let mut done = failed.clone();
        done.data.artifacts = Some(vec![artifact(&json!({"n": 1}), "art_1".into())]);
        assert_eq!(artifact_text(&done), Some(json!({"n": 1})));
        done.data.history = vec![v1(json!({"messageId": "m", "role": "ROLE_USER", "parts": [],
            "metadata": {"https://livekit.io/a2a/ext/agents/v1/channel": "sms"}}))];
        assert_eq!(channel_of_record(&done).as_deref(), Some("sms"));
        let r = result_for(done.clone(), "AT_1", Some("r1"));
        assert_eq!((r.answered_by.as_str(), r.run_id.as_deref()), ("AST_2", Some("r1")));
        done.data.metadata.insert(MERGED_INTO_KEY.into(), json!("AST_9"));
        assert_eq!(result_for(done, "AT_1", None).answered_by, "AST_9");
        assert_eq!(
            python_reason(&StorageError::Unavailable("lost".into())),
            "StorageUnavailable: lost"
        );
        assert_eq!(python_reason(&StorageError::Cancelled(String::new())), "CancelledError");
    }

    // Catches: a cancelled handler written as a storage failure, or a storage error the handler
    // met losing its kind.
    #[test]
    fn host_failures_map_to_storage_errors() {
        let host = |kind: &str| ProviderError::Host {
            error: lk_agents_abi::errors::HostError {
                kind: kind.into(),
                message: "m".into(),
                host_ref: None,
            },
            api_kind: None,
            retryable: false,
            status_code: None,
        };
        assert!(
            matches!(host_error(host("CancelledError")), StorageError::Cancelled(m) if m == "m")
        );
        assert!(matches!(host_error(host("StorageUnavailable")), StorageError::Unavailable(_)));
        assert!(matches!(host_error(host("ThreadNotFound")), StorageError::ThreadNotFound(_)));
        assert!(matches!(
            host_error(host("VersionConflict")),
            StorageError::Server { code, .. } if code == "VersionConflict"
        ));
    }
}
