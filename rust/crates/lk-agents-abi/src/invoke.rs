//! HostInvoker's contract (docs/internals/rust-core.md "HostInvoker"): the scopes a host call can
//! require of what it runs in ([`HostContext`](crate::agent::HostContext)), and the port a
//! language binding implements to enter its host.
//! `lk-agents-runtime`'s `host` module owns the dispatch; the binding only enters.

use crate::BoxFuture;
use crate::utils::host_ref::HostRef;

/// A framework scope a host call can require of the invocation it runs in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scope {
    /// The job the work runs for.
    Job,
    /// The agent invocation (its session's active agent).
    Agent,
    /// The speech the work belongs to.
    Speech,
    /// A tool call's identity: preparation, read-back and body.
    ToolCall,
    /// The approval a body runs on.
    Approval,
}

impl std::fmt::Display for Scope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Scope::Job => "job",
            Scope::Agent => "agent",
            Scope::Speech => "speech",
            Scope::ToolCall => "tool call",
            Scope::Approval => "approval",
        })
    }
}

/// What a framework call runs in: the host context captured at its entry (a `_core` method's
/// binding captures it before scheduling native work) and the framework scopes it holds. Calls it
/// leads to carry it explicitly; Tokio tasks and threads never inherit it.
#[derive(Debug, Clone, Default)]
pub struct CallContext {
    pub host_context: Option<HostRef>,
    pub job: Option<String>,
    pub session: Option<String>,
    pub speech: Option<String>,
}

impl crate::agent::HostContext for CallContext {
    fn host_context(&self) -> Option<&HostRef> {
        self.host_context.as_ref()
    }

    fn holds(&self, scope: Scope) -> bool {
        match scope {
            Scope::Job => self.job.is_some(),
            Scope::Agent => self.session.is_some(),
            Scope::Speech => self.speech.is_some(),
            Scope::ToolCall | Scope::Approval => false,
        }
    }
}

/// A host call refused before its host is entered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum Refusal {
    #[error("the call captured no host context")]
    NoContext,
    #[error("the call requires a {0} scope")]
    MissingScope(Scope),
}

/// The language binding's entry into its host (Python: the GIL, a copy of the captured
/// `contextvars.Context` and the coroutine scheduled on its loop).
pub trait HostEntry: Send + Sync + 'static {
    /// Builds the host callable and its arguments, inside the entered context.
    type Call: Send + 'static;
    type Value: Send + 'static;
    type Error: Send + 'static;

    /// Runs `call` in a copy of `context`: resolves to its value, or to what the awaitable it
    /// returned resolved to. Dropping the future cancels the host side, which still settles.
    fn enter(
        &self,
        context: &HostRef,
        call: Self::Call,
    ) -> BoxFuture<'static, Result<Self::Value, Self::Error>>;

    /// Runs `call` in place, on the calling thread, never awaiting (a read the host answers at
    /// once).
    fn enter_now(&self, call: Self::Call) -> Result<Self::Value, Self::Error>;

    /// The failure a refused call reports, labelled with the call.
    fn refused(&self, label: &'static str, refusal: Refusal) -> Self::Error;
}

/// One proxy's ordered notifications: each runs in a copy of its own context, in call order.
pub trait HostNotices: Send + Sync {
    type Call: Send + 'static;

    /// Queues `call`; `lossy` lets it be dropped (and logged) while the host is far behind.
    fn notify(&self, context: HostRef, label: &'static str, lossy: bool, call: Self::Call);
}
