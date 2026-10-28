//! `HostInvoker` (docs/internals/rust-core.md "HostInvoker"): the one lifecycle for host callbacks.
//! An operation's [`Invocation`] is captured once, at its entry; each callback gets its own copy
//! with the scopes it declares (tool call, approval) and a fresh operation id, runs as a task this
//! invoker owns, is cancelled through the host (its call still settles), and releases its captured
//! context when it settles. Entering the host context is the generated adapter's job.

use std::collections::BTreeMap;

use lk_agents_abi::FunctionCall;
use lk_agents_abi::agent::Invocation;
use lk_agents_abi::llm::tool_executor::Approval;
use tokio::task::JoinSet;

pub struct HostInvoker {
    /// The operation's entry: identity, agent and captured host context.
    entry: Invocation,
    next_operation: u64,
    live: BTreeMap<u64, Invocation>,
    tasks: JoinSet<u64>,
}

impl HostInvoker {
    /// `entry` is captured where the owning operation was created; its `operation` is unused.
    pub fn new(entry: Invocation) -> Self {
        HostInvoker { entry, next_operation: 0, live: BTreeMap::new(), tasks: JoinSet::new() }
    }

    /// Runs `call` with a new invocation derived from the entry. The returned id names it for
    /// [`cancel`](Self::cancel); `call` must report its own outcome (it owns the delivery).
    pub fn invoke<F, Fut>(
        &mut self,
        function_call: Option<FunctionCall>,
        approval: Option<Approval>,
        runtime: &tokio::runtime::Handle,
        call: F,
    ) -> u64
    where
        F: FnOnce(Invocation) -> Fut,
        Fut: Future<Output = ()> + Send + 'static,
    {
        self.next_operation += 1;
        let operation = self.next_operation;
        let invocation = Invocation { operation, function_call, approval, ..self.entry.clone() };
        self.live.insert(operation, invocation.clone());
        let work = call(invocation);
        self.tasks.spawn_on(
            async move {
                work.await;
                operation
            },
            runtime,
        );
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
        loop {
            match self.tasks.join_next().await {
                Some(Ok(operation)) => {
                    self.live.remove(&operation);
                    return operation;
                }
                // A panicked delivery: its invocation is released when the invoker drops.
                Some(Err(_)) => continue,
                None => std::future::pending::<()>().await,
            }
        }
    }

    /// The scopes still captured: one per callback in flight, never more.
    pub fn captured(&self) -> usize {
        self.live.len()
    }
}
