use lk_agents_abi::a2a::turn::{HandlerOutput, HandlerPort, TurnEvent, TurnHost};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::{BoxFuture, Clock, Ts};
use lk_agents_net::task::Cancellation;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

// The turn reads reports in order and sees the handler's output only once every report it
// awaited before returning was read.

/// Reports a handler may have awaited and the turn not yet read.
pub const REPORT_QUEUE: usize = 64;

enum Report {
    Checkpoint(oneshot::Sender<bool>),
    Event(TurnEvent),
}

/// What the handler reports through.
pub struct HandlerTurn {
    task_id: String,
    reports: mpsc::Sender<Report>,
}

impl HandlerPort for HandlerTurn {
    fn checkpoint(&self) -> BoxFuture<'_, bool> {
        Box::pin(async move {
            let (verdict, answered) = oneshot::channel();
            if self.reports.send(Report::Checkpoint(verdict)).await.is_err() {
                return true;
            }
            answered.await.unwrap_or(true)
        })
    }
    fn progress(&self, text: String) -> BoxFuture<'_, ()> {
        let event = TurnEvent::Progress { task_id: self.task_id.clone(), text };
        Box::pin(async move {
            let _ = self.reports.send(Report::Event(event)).await;
        })
    }
    fn token(&self, delta: String) -> BoxFuture<'_, ()> {
        let event = TurnEvent::Token { task_id: self.task_id.clone(), delta };
        Box::pin(async move {
            let _ = self.reports.send(Report::Event(event)).await;
        })
    }
}

impl lk_agents_abi::a2a::turn::HandlerTurn for HandlerTurn {
    async fn checkpoint(&self) -> bool {
        HandlerPort::checkpoint(self).await
    }
    async fn progress(&self, text: String) {
        HandlerPort::progress(self, text).await
    }
    async fn token(&self, delta: String) {
        HandlerPort::token(self, delta).await
    }
}

/// What the turn reads next from its handler.
pub enum Read {
    Checkpoint,
    Event(TurnEvent),
    Output(Box<Result<HandlerOutput, ProviderError>>),
}

/// One open handler: its host call, its unread reports, the checkpoint awaiting a verdict.
pub struct Handler {
    call: Option<BoxFuture<'static, Result<HandlerOutput, ProviderError>>>,
    /// The call's result, until the reports it awaited before returning were read.
    output: Option<Result<HandlerOutput, ProviderError>>,
    reports: mpsc::Receiver<Report>,
    pending: Option<oneshot::Sender<bool>>,
    cancel: Arc<Cancellation>,
}

impl Handler {
    /// The handler's host call on `ctx`; it starts when first read.
    pub fn open(
        host: Arc<dyn TurnHost>,
        cx: crate::telemetry::Cx,
        ctx: lk_agents_abi::a2a::turn::TurnContext,
    ) -> Self {
        let (reports, unread) = mpsc::channel(REPORT_QUEUE);
        let turn = Arc::new(HandlerTurn { task_id: ctx.task_id.clone(), reports });
        let cancel = Arc::new(Cancellation::default());
        let signal = cancel.clone();
        let call = Box::pin(async move { host.handle(&cx, &ctx, turn, signal).await });
        Self { call: Some(call), output: None, reports: unread, pending: None, cancel }
    }

    /// Answers the checkpoint the handler waits at, if any.
    pub fn answer(&mut self, should_yield: bool) {
        if let Some(pending) = self.pending.take() {
            let _ = pending.send(should_yield);
        }
    }

    /// The handler's next report, or its output once every report before it was read; `None`
    /// once the output was read.
    pub async fn next(&mut self) -> Option<Read> {
        if let Some(call) = self.call.as_mut() {
            tokio::select! {
                biased;
                Some(report) = self.reports.recv() => return Some(self.read(report)),
                output = call => {
                    self.call = None;
                    self.output = Some(output);
                }
            }
        }
        // The call may return in the same poll as its last reports: those come first.
        let output = self.output.take()?;
        match self.reports.try_recv() {
            Ok(report) => {
                self.output = Some(output);
                Some(self.read(report))
            }
            Err(_) => Some(Read::Output(Box::new(output))),
        }
    }

    fn read(&mut self, report: Report) -> Read {
        match report {
            Report::Checkpoint(verdict) => {
                self.pending = Some(verdict);
                Read::Checkpoint
            }
            Report::Event(event) => Read::Event(event),
        }
    }

    /// Cancels a running handler and waits for its cleanup until `deadline`; `false` when it
    /// was still running then.
    pub async fn close(mut self, clock: &dyn Clock, deadline: Ts) -> bool {
        self.reports.close();
        self.pending = None;
        let Some(call) = self.call.take() else { return true };
        self.cancel.cancel();
        tokio::select! {
            biased;
            _ = call => true,
            () = clock.wait_until(deadline) => false,
        }
    }
}

#[cfg(test)]
#[path = "handler_tests.rs"]
mod tests;
