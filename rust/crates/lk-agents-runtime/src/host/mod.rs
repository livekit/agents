use crate::telemetry;
use futures_util::StreamExt as _;
use futures_util::stream::FuturesUnordered;
use lk_agents_abi::agent::{HostContext, Invocation};
use lk_agents_abi::invoke::{HostEntry, HostNotices, Refusal, Scope};
use lk_agents_abi::llm::tool_executor::Approval;
use lk_agents_abi::utils::host_ref::HostRef;
use lk_agents_abi::{BoxFuture, FunctionCall};
use lk_agents_abi::{SpawnerExt, TaskOutcome};
use lk_agents_net::task::TokioSpawner;
use std::collections::BTreeMap;

// Never call the host while holding a lock: an entry may wait for the GIL.

/// Runs `call` through `entry` in a copy of the host context `scope` captured, with `cx`'s span
/// current; dropping the future cancels the host side, which still settles. A refused call
/// resolves to the entry's error for it, the host never entered.
pub fn call<E: HostEntry>(
    entry: &E,
    label: &'static str,
    scope: &(impl HostContext + ?Sized),
    requires: &[Scope],
    cx: &telemetry::Cx,
    call: E::Call,
) -> BoxFuture<'static, Result<E::Value, E::Error>> {
    match admit(scope, requires) {
        Ok(context) => entry.enter(context, cx.span(), call),
        Err(refusal) => {
            let error = entry.refused(label, refusal);
            Box::pin(async move { Err(error) })
        }
    }
}

/// An owned operation's cancellation is delivered without dropping its settlement future.
pub fn call_cancellable<E: HostEntry>(
    entry: &E,
    label: &'static str,
    scope: &(impl HostContext + ?Sized),
    requires: &[Scope],
    cx: &telemetry::Cx,
    cancel: std::sync::Arc<dyn lk_agents_abi::invoke::Cancellation>,
    call: E::Call,
) -> BoxFuture<'static, Result<E::Value, E::Error>> {
    match admit(scope, requires) {
        Ok(context) => entry.enter_cancellable(context, cx.span(), call, cancel),
        Err(refusal) => {
            let error = entry.refused(label, refusal);
            Box::pin(async move { Err(error) })
        }
    }
}

/// Runs `call` through `entry` in place, on the calling thread: a read the host answers at once.
pub fn call_now<E: HostEntry>(entry: &E, call: E::Call) -> Result<E::Value, E::Error> {
    entry.enter_now(call)
}

/// Queues `call` on `notices`, after the notifications before it, to run in a copy of the context
/// `scope` captured. Not awaited: a refusal or a host failure is logged.
pub fn notify<N: HostNotices>(
    notices: &N,
    label: &'static str,
    scope: &(impl HostContext + ?Sized),
    requires: &[Scope],
    lossy: bool,
    call: N::Call,
) {
    match admit(scope, requires) {
        Ok(context) => notices.notify(context.clone(), label, lossy, call),
        Err(refusal) => log::warn!(target: lk_agents_abi::LOG_TARGET, "{label} failed: {refusal}"),
    }
}

/// The context a call runs in, once `scope` holds every scope it requires.
fn admit<'a>(
    scope: &'a (impl HostContext + ?Sized),
    requires: &[Scope],
) -> Result<&'a HostRef, Refusal> {
    let context = scope.host_context().ok_or(Refusal::NoContext)?;
    match requires.iter().find(|s| !scope.holds(**s)) {
        Some(missing) => Err(Refusal::MissingScope(*missing)),
        None => Ok(context),
    }
}

pub struct HostInvoker {
    /// The operation's entry: identity, agent and captured host context.
    entry: Invocation,
    next_operation: u64,
    live: BTreeMap<u64, Invocation>,
    /// Each callback's task, yielding its operation once settled; dropping the invoker cancels
    /// them.
    tasks: FuturesUnordered<BoxFuture<'static, u64>>,
}

impl HostInvoker {
    /// `entry` is captured where the owning operation was created; its `operation` is unused.
    pub fn new(entry: Invocation) -> Self {
        HostInvoker {
            entry,
            next_operation: 0,
            live: BTreeMap::new(),
            tasks: FuturesUnordered::new(),
        }
    }

    /// Runs `call` with a new invocation derived from the entry. The returned id identifies
    /// the callback for [`live`](Self::live); `call` reports its own outcome and owns delivery.
    /// A callback that ends without completing (a panic, a cancel) delivers `lost` instead, so
    /// whoever waits on its report is never left waiting.
    pub fn invoke<F, Fut, L>(
        &mut self,
        function_call: Option<FunctionCall>,
        approval: Option<Approval>,
        runtime: &tokio::runtime::Handle,
        call: F,
        lost: L,
    ) -> u64
    where
        F: FnOnce(Invocation) -> Fut,
        Fut: Future<Output = ()> + Send + 'static,
        L: Future<Output = ()> + Send + 'static,
    {
        self.next_operation += 1;
        let operation = self.next_operation;
        let invocation = Invocation { operation, function_call, approval, ..self.entry.clone() };
        self.live.insert(operation, invocation.clone());
        let work = call(invocation);
        let task = TokioSpawner::new(runtime.clone()).spawn_settled(work, |outcome| async move {
            if !matches!(outcome, TaskOutcome::Completed) {
                lost.await;
            }
        });
        self.tasks.push(Box::pin(async move {
            let _ = task.await;
            operation
        }));
        operation
    }

    /// The invocation of a callback still in flight.
    pub fn live(&self, operation: u64) -> Option<&Invocation> {
        self.live.get(&operation)
    }

    /// Whether any callback is still in flight.
    pub fn is_idle(&self) -> bool {
        self.tasks.is_empty()
    }

    /// Waits for the next callback to settle and releases its invocation (and the host context
    /// it held). Pending forever while none is in flight.
    pub async fn settled(&mut self) -> u64 {
        match self.tasks.next().await {
            Some(operation) => {
                self.live.remove(&operation);
                operation
            }
            None => std::future::pending().await,
        }
    }
}

// Explicit cancellation of an owned host operation, acknowledged by its normal result.
pub use lk_agents_net::task::Cancellation;
