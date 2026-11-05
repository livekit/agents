//! An `MCPToolset`'s lifecycle: its listing, the background recovery of an unreachable server
//! and the idle close after the last run. The host lists and closes; this decides when.

use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::errors::HostError;
use lk_agents_abi::invoke::CallContext;
use lk_agents_abi::llm::mcp::{McpToolsetHost, McpToolsetLifecycle as Port};
use lk_agents_abi::utils::stream::HOST_TEARDOWN_TIMEOUT;
use lk_agents_net::task::Cancellation;
use parking_lot::Mutex;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::telemetry;

pub struct McpToolsetLifecycle {
    server: String,
    idle_close: Duration,
    /// One setup at a time.
    setting_up: tokio::sync::Mutex<()>,
    state: Arc<Mutex<State>>,
}

#[derive(Default)]
struct State {
    listed: bool,
    recovery: Option<Recovery>,
    /// The armed idle close; taken by whichever comes first, a setup (disarming it) or the
    /// close itself (firing it).
    idle: Option<(u64, JoinHandle<()>)>,
    idle_generation: u64,
}

struct Recovery {
    stop: CancellationToken,
    task: JoinHandle<()>,
}

/// One cancellable listing: `Ok(true)` when the tools are listed, `Ok(false)` when `stop` fired
/// first (the listing cancelled and settled).
async fn list(
    host: &dyn McpToolsetHost,
    cx: &telemetry::Cx,
    context: &CallContext,
    stop: Option<&CancellationToken>,
) -> Result<bool, HostError> {
    let cancel = Arc::new(Cancellation::default());
    let mut listing = host.list(cx, context, cancel.clone());
    let Some(stop) = stop else { return listing.await.map(|()| true) };
    tokio::select! {
        biased;
        result = &mut listing => result.map(|()| true),
        () = stop.cancelled() => {
            cancel.cancel();
            if tokio::time::timeout(HOST_TEARDOWN_TIMEOUT, listing).await.is_err() {
                log::error!(target: "livekit.agents", "MCP listing cleanup incomplete");
            }
            Ok(false)
        }
    }
}

/// List again at the session's backoff until the server answers or `stop` fires.
async fn recover(
    host: Arc<dyn McpToolsetHost>,
    state: Arc<Mutex<State>>,
    stop: CancellationToken,
    context: CallContext,
    cx: telemetry::Cx,
) {
    loop {
        let wait = host.retry_in().unwrap_or_else(|error| {
            log::warn!(target: "livekit.agents", "MCP retry delay unavailable: {error}");
            1.0
        });
        tokio::select! {
            biased;
            () = stop.cancelled() => return,
            () = tokio::time::sleep(Duration::try_from_secs_f64(wait).unwrap_or_default()) => {}
        }
        match list(&*host, &cx, &context, Some(&stop)).await {
            Ok(true) => {
                let mut state = state.lock();
                state.listed = true;
                state.recovery = None;
                return;
            }
            Ok(false) => return,
            Err(_) => {}
        }
    }
}

impl Port for McpToolsetLifecycle {
    fn new(server: String, idle_close: f64) -> Self {
        Self {
            server,
            idle_close: Duration::try_from_secs_f64(idle_close).unwrap_or_default(),
            setting_up: tokio::sync::Mutex::new(()),
            state: Arc::default(),
        }
    }

    async fn setup(&self, host: Arc<dyn McpToolsetHost>, context: CallContext, cx: telemetry::Cx) {
        let _one = self.setting_up.lock().await;
        {
            let mut state = self.state.lock();
            if let Some((_, idle)) = state.idle.take() {
                idle.abort();
            }
            if state.listed || state.recovery.is_some() {
                return;
            }
        }
        match list(&*host, &cx, &context, None).await {
            Ok(_) => self.state.lock().listed = true,
            Err(error) => {
                log::warn!(
                    target: "livekit.agents",
                    "MCP server unavailable, continuing without its tools: {error}, mcp_server={}",
                    self.server
                );
                let stop = CancellationToken::new();
                let task =
                    tokio::spawn(recover(host, self.state.clone(), stop.clone(), context, cx));
                self.state.lock().recovery = Some(Recovery { stop, task });
            }
        }
    }

    async fn release(
        &self,
        host: Arc<dyn McpToolsetHost>,
        context: CallContext,
        cx: telemetry::Cx,
    ) {
        let recovery = self.state.lock().recovery.take();
        if let Some(Recovery { stop, task }) = recovery {
            stop.cancel();
            let _ = task.await;
        }
        let (state, idle) = (self.state.clone(), self.idle_close);
        let mut guard = self.state.lock();
        guard.idle_generation += 1;
        let generation = guard.idle_generation;
        let task = tokio::spawn(async move {
            tokio::time::sleep(idle).await;
            {
                let mut state = state.lock();
                // A setup that came first disarmed it.
                if state.idle.as_ref().is_none_or(|(armed, _)| *armed != generation) {
                    return;
                }
                state.idle = None;
                state.listed = false;
            }
            if let Err(error) = host.close_idle(&cx, &context).await {
                log::warn!(target: "livekit.agents", "MCP idle close failed: {error}");
            }
        });
        if let Some((_, previous)) = guard.idle.replace((generation, task)) {
            previous.abort();
        }
    }
}

#[cfg(test)]
#[path = "toolset_tests.rs"]
mod tests;
