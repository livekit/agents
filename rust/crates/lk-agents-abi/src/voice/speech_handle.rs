//! `_core.SpeechHandle`: one speech of a voice session as the host's `SpeechHandle` wraps it. The
//! speech's state machine is `lk-agents-voice`'s.

use std::any::Any;

use crate::errors::ProviderError;
use crate::{ChatItem, HostRef};

/// A speech handed to the host: it crosses as its `_core.SpeechHandle`.
#[crate::host(native(refused = "expected a _core.SpeechHandle"))]
pub trait Speech: Any + Send + Sync + std::fmt::Debug {}

/// A refusal on a speech handle; the message is the `RuntimeError` it raises.
#[crate::host(error(kind = "RuntimeError"))]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{0}")]
pub struct SpeechRefused(pub String);

#[crate::host(class(name = "SpeechHandle", native = Speech))]
pub trait SpeechHandle: Send + Sync + Sized + 'static {
    /// A handle constructed directly (`SpeechHandle.create()`): no loop will create a speech
    /// behind it.
    #[host(new)]
    fn new(
        #[host(kw, default = "")] speech_id: String,
        #[host(default = true)] allow_interruptions: bool,
    ) -> Self;

    /// The host's object for this speech, as it set it; `None` until then.
    #[host(getter, name = "host")]
    fn host_object(&self) -> Option<HostRef>;

    /// Set by the host when it creates its handle over this one: from then on every event, tool
    /// and hook naming the speech hands back this object.
    #[host(setter, name = "host")]
    fn set_host_object(&self, host: HostRef);

    /// `speech_handle.id`. Empty until the loop has minted `speech_{n}`; a detached handle answers
    /// the id it was constructed with.
    #[host(getter)]
    fn id(&self) -> String;

    #[host(getter)]
    fn num_steps(&self) -> u32;

    /// What this speech committed to the conversation; a copy.
    #[host(getter)]
    fn chat_items(&self) -> Vec<ChatItem>;

    #[host(getter)]
    fn scheduled(&self) -> bool;

    /// `True` from the tick `interrupt()` was called until the speech is done, and once the loop
    /// has interrupted the speech for any reason (an interruption, a supersede, the close). A done
    /// speech reads the loop's verdict.
    #[host(getter)]
    fn interrupted(&self) -> bool;

    #[host(call)]
    fn done(&self) -> bool;

    /// The core's last accepted `allow_interruptions`.
    #[host(getter)]
    fn allow_interruptions(&self) -> bool;

    /// Enqueues the change; the accepted view changes when the loop applies it.
    #[host(setter, name = "allow_interruptions")]
    fn set_allow_interruptions(&self, value: bool) -> Result<(), SpeechRefused>;

    /// The exception the speech failed with: the host's own object when a provider raised one.
    #[host(call)]
    fn exception(&self) -> Result<Option<ProviderError>, SpeechRefused>;

    /// `interrupt(force=)`, in the reference's order. If the speech does not exist yet, the
    /// command is sent when the loop creates it.
    #[host(call)]
    fn interrupt(&self, #[host(kw, default = false)] force: bool) -> Result<(), SpeechRefused>;

    /// `EndCallTool`: close the session once this speech is done, or once a realtime model's own
    /// reply to the tool result is. Sent to this speech's own session.
    #[host(call)]
    fn end_call(&self) -> Result<(), SpeechRefused>;

    /// `_hold_interruptions`: disallow interruptions until every hold taken is released. Raises
    /// when the speech is already interrupted, without counting the attempt.
    #[host(call)]
    fn hold_interruptions(&self) -> Result<(), SpeechRefused>;

    /// `_release_interruptions`. Idempotent past zero, like the reference's.
    #[host(call)]
    fn release_interruptions(&self) -> Result<(), SpeechRefused>;

    /// `wait_for_playout()`, and therefore `__await__`, from `origin` (`"tool:<call_id>"` inside a
    /// tool). Raises from inside a tool this speech waits on; otherwise never raises: a speech
    /// the session ended before creating, or a handle constructed directly, has nothing to play.
    #[host(await)]
    async fn wait_for_playout(
        &self,
        #[host(kw, default = None)] origin: Option<String>,
    ) -> Result<(), SpeechRefused>;

    /// `_wait_for_generation(step_idx)`: one step of the turn, not the whole thing; `None` is the
    /// latest. No generation, no such step and a never-created speech are distinct errors.
    #[host(await)]
    async fn wait_for_generation(&self, step: Option<usize>) -> Result<(), SpeechRefused>;

    /// Resolves when the speech is interrupted, and only then: a speech that finishes normally
    /// never resolves this, so `wait_if_not_interrupted` can race it.
    #[host(await)]
    async fn wait_for_interrupted(&self) -> Result<(), SpeechRefused>;

    /// Resolves once the loop has created the speech and put it in the queue.
    #[host(await)]
    async fn wait_until_created(&self) -> Result<(), SpeechRefused>;
}
