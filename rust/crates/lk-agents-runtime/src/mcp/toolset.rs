use std::sync::Arc;
use std::time::Duration;

use lk_agents_abi::errors::HostError;
use lk_agents_abi::invoke::CallContext;
use lk_agents_abi::llm::mcp::{
    McpError, McpSession as _, McpTimeout, McpToolsetHost, McpToolsetLifecycle as Port,
};
use lk_agents_abi::utils::stream::HOST_TEARDOWN_TIMEOUT;
use lk_agents_abi::{Clock, timeout};
use lk_agents_core::mcp::timeout_duration;
use lk_agents_core::request::retrying;
use lk_agents_net::env::ProcessEnv;
use lk_agents_net::owned::Background;
use lk_agents_net::signal;
use lk_agents_net::task::{Cancellation, ScopedTask};
use parking_lot::Mutex;
use tokio_util::sync::CancellationToken;

use super::session::McpSession;
use crate::telemetry;

pub struct McpToolsetLifecycle {
    server: String,
    session: Arc<McpSession>,
    idle_close: Duration,
    /// Where the recovery and the idle close run, on the host's clock.
    background: Background,
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
    idle: Option<(u64, ScopedTask)>,
    idle_generation: u64,
}

struct Recovery {
    stop: CancellationToken,
    task: ScopedTask,
    /// Ends when the recovery's task does, its listing settled.
    done: signal::Cell<Option<()>>,
}

/// One cancellable listing: `Ok(true)` when the tools are listed, `Ok(false)` when `stop` fired
/// first (the listing cancelled and settled).
async fn list(
    host: &dyn McpToolsetHost,
    clock: &dyn Clock,
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
            if timeout(clock, HOST_TEARDOWN_TIMEOUT, listing).await.is_err() {
                log::error!(target: lk_agents_abi::LOG_TARGET, "MCP listing cleanup incomplete");
            }
            Ok(false)
        }
    }
}

/// List again at the session's backoff until the server answers or `stop` fires. The first failure
/// was a warning; each further one is logged at info, since the agent already runs without the
/// server's tools and the recovery repeats for as long as the server is down.
#[allow(clippy::too_many_arguments, reason = "the recovery's owner, inputs and its host")]
async fn recover(
    subject: String,
    host: Arc<dyn McpToolsetHost>,
    session: Arc<McpSession>,
    clock: Arc<dyn Clock>,
    state: Arc<Mutex<State>>,
    stop: CancellationToken,
    context: CallContext,
    cx: telemetry::Cx,
) {
    let mut delay = session.retry_delay();
    for attempt in 2_u32.. {
        let at = clock.now().saturating_add(delay);
        tokio::select! {
            biased;
            () = stop.cancelled() => return,
            () = clock.wait_until(at) => {}
        }
        match list(&*host, &*clock, &cx, &context, Some(&stop)).await {
            Ok(true) => {
                let mut state = state.lock();
                state.listed = true;
                // Ending on its own: its handle no longer cancels it.
                if let Some(recovery) = state.recovery.take() {
                    recovery.task.release();
                }
                return;
            }
            Ok(false) => return,
            Err(error) => {
                delay = session.retry_delay();
                retrying(log::Level::Info, &subject, attempt, delay, &error);
            }
        }
    }
}

/// After `idle` unless disarmed: drop the tools and close the server. A host that can no longer
/// run its close (its event loop is gone) leaves the native session to close here.
#[allow(clippy::too_many_arguments, reason = "the idle close's owner, inputs and its host")]
async fn close_when_idle(
    host: Arc<dyn McpToolsetHost>,
    session: Arc<McpSession>,
    clock: Arc<dyn Clock>,
    state: Arc<Mutex<State>>,
    idle: Duration,
    generation: u64,
    context: CallContext,
    cx: telemetry::Cx,
) {
    clock.wait_until(clock.now().saturating_add(idle)).await;
    {
        let mut state = state.lock();
        // A setup that came first disarmed it.
        if state.idle.as_ref().is_none_or(|(armed, _)| *armed != generation) {
            return;
        }
        // Ending on its own: its handle no longer cancels it.
        if let Some((_, task)) = state.idle.take() {
            task.release();
        }
        state.listed = false;
    }
    if let Err(error) = host.close_idle(&cx, &context).await {
        log::debug!(target: lk_agents_abi::LOG_TARGET, "MCP idle close ran natively, the host's failed: {error}");
        session.close().await;
    }
}

impl Port for McpToolsetLifecycle {
    type Env = ProcessEnv;
    type Session = McpSession;

    fn new(
        env: &ProcessEnv,
        server: String,
        session: Arc<Self::Session>,
        idle_timeout: f64,
    ) -> Result<Self, McpError> {
        Ok(Self {
            server,
            session,
            idle_close: timeout_duration(McpTimeout::Idle, idle_timeout)?,
            background: env.background().map_err(|e| HostError::new(e.to_string()))?,
            setting_up: tokio::sync::Mutex::new(()),
            state: Arc::default(),
        })
    }

    async fn setup(&self, host: Arc<dyn McpToolsetHost>, context: CallContext, cx: telemetry::Cx) {
        let _one = self.setting_up.lock().await;
        // A server that said its tools changed is listed again by the next run.
        let changed = self.session.tools_changed();
        let disarmed = {
            let mut state = self.state.lock();
            let disarmed = state.idle.take();
            if (state.listed && !changed) || state.recovery.is_some() {
                return;
            }
            disarmed
        };
        drop(disarmed);
        let clock = &*self.background.clock;
        match list(&*host, clock, &cx, &context, None).await {
            Ok(_) => self.state.lock().listed = true,
            Err(error) => {
                let subject = format!("MCP server {} (continuing without its tools)", self.server);
                retrying(log::Level::Warn, &subject, 1, self.session.retry_delay(), &error);
                let stop = CancellationToken::new();
                let (finished, done) = signal::once();
                let recovery = recover(
                    subject,
                    host,
                    self.session.clone(),
                    self.background.clock.clone(),
                    self.state.clone(),
                    stop.clone(),
                    context,
                    cx,
                );
                // Holds the state's lock until the recovery is recorded, so it cannot end first.
                // Refused only once the process is shutting down: no recovery then.
                let mut state = self.state.lock();
                let task = self.background.tasks.try_scoped(
                    self.background.spawner.as_ref(),
                    "MCP toolset recovery",
                    async move {
                        let _finished = finished;
                        recovery.await;
                    },
                );
                if let Some(task) = task {
                    state.recovery = Some(Recovery { stop, task, done });
                }
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
        if let Some(Recovery { stop, task, done }) = recovery {
            stop.cancel();
            let _ = done.wait().await;
            drop(task);
        }
        let mut state = self.state.lock();
        state.idle_generation += 1;
        let generation = state.idle_generation;
        let close = close_when_idle(
            host,
            self.session.clone(),
            self.background.clock.clone(),
            self.state.clone(),
            self.idle_close,
            generation,
            context,
            cx,
        );
        // Refused only once the process is shutting down; the session closes when dropped.
        let task = self.background.tasks.try_scoped(
            self.background.spawner.as_ref(),
            "MCP toolset idle close",
            close,
        );
        let previous = std::mem::replace(&mut state.idle, task.map(|task| (generation, task)));
        drop(state);
        drop(previous);
    }
}

#[cfg(test)]
#[path = "toolset_tests.rs"]
mod tests;
