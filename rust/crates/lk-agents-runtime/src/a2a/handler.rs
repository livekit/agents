//! The host handler of one turn: one awaited host call, its reports read through a bounded queue.
//!
//! The handler awaits each report ([`HandlerPort`]): a checkpoint waits for the turn's verdict,
//! progress and tokens for room in the queue. The turn reads the queue in report order and
//! sees the handler's output only once every report it awaited before returning was read.

use lk_agents_abi::a2a::turn::{HandlerOutput, HandlerPort, TurnEvent, TurnHost, Verdict};
use lk_agents_abi::errors::ProviderError;
use lk_agents_abi::{BoxFuture, Clock, Ts};
use lk_agents_net::task::Cancellation;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};

/// Reports a handler may have awaited and the turn not yet read.
pub const REPORT_QUEUE: usize = 64;

enum Report {
    Checkpoint(oneshot::Sender<bool>),
    Event(TurnEvent),
}

/// What the handler reports through; handed to the host as `_core_HandlerTurn`.
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
    pub fn answer(&mut self, verdict: Verdict) {
        if let Some(pending) = self.pending.take() {
            let _ = pending.send(verdict == Verdict::Yield);
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
mod tests {
    use super::*;
    use lk_agents_net::clock::RealClock;
    use parking_lot::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    type Run = dyn Fn(
            Arc<dyn HandlerPort>,
            Arc<dyn lk_agents_abi::invoke::Cancellation>,
        ) -> BoxFuture<'static, Result<HandlerOutput, ProviderError>>
        + Send
        + Sync;

    struct Script(Box<Run>);

    impl TurnHost for Script {
        fn handle<'a>(
            &'a self,
            _cx: &crate::telemetry::Cx,
            _ctx: &'a lk_agents_abi::a2a::turn::TurnContext,
            turn: Arc<dyn HandlerPort>,
            cancel: Arc<dyn lk_agents_abi::invoke::Cancellation>,
        ) -> BoxFuture<'a, Result<HandlerOutput, ProviderError>> {
            (self.0)(turn, cancel)
        }
    }

    fn open(
        run: impl Fn(
            Arc<dyn HandlerPort>,
            Arc<dyn lk_agents_abi::invoke::Cancellation>,
        ) -> BoxFuture<'static, Result<HandlerOutput, ProviderError>>
        + Send
        + Sync
        + 'static,
    ) -> Handler {
        let ctx = lk_agents_abi::a2a::turn::TurnContext {
            thread_id: "AT_1".into(),
            lane: "desk".into(),
            message: Default::default(),
            batch: Vec::new(),
            history: Vec::new(),
            state: None,
            context_scope_id: String::new(),
            context_scopes: Vec::new(),
            read_only: false,
            task_id: "task".into(),
            tasks: Vec::new(),
            caller_history: None,
            answer_tool: None,
            host_context: None,
        };
        Handler::open(Arc::new(Script(Box::new(run))), crate::telemetry::Cx::NONE, ctx)
    }

    fn progress(text: &str) -> TurnEvent {
        TurnEvent::Progress { task_id: "task".into(), text: text.into() }
    }

    // Catches: a report overtaken by the output, or a verdict not reaching the checkpoint.
    #[tokio::test(start_paused = true)]
    async fn reports_arrive_in_order_and_the_output_after_them() {
        let mut handler = open(|turn, _| {
            Box::pin(async move {
                turn.progress("working".into()).await;
                turn.token("reply".into()).await;
                let interrupted = turn.checkpoint().await;
                turn.progress("after".into()).await;
                Ok(HandlerOutput { interrupted, ..Default::default() })
            })
        });
        assert!(matches!(handler.next().await, Some(Read::Event(e)) if e == progress("working")));
        let token = TurnEvent::Token { task_id: "task".into(), delta: "reply".into() };
        assert!(matches!(handler.next().await, Some(Read::Event(e)) if e == token));
        assert!(matches!(handler.next().await, Some(Read::Checkpoint)));
        handler.answer(Verdict::Yield);
        assert!(matches!(handler.next().await, Some(Read::Event(e)) if e == progress("after")));
        assert!(
            matches!(handler.next().await, Some(Read::Output(out)) if matches!(*out, Ok(ref out) if out.interrupted))
        );
        assert!(handler.next().await.is_none(), "the output is read once");
        assert!(handler.close(&RealClock::new(), Ts::from_nanos(u64::MAX)).await);
    }

    // Catches: an unbounded queue letting a handler run ahead of the turn.
    #[tokio::test(start_paused = true)]
    async fn a_full_queue_holds_the_handler_until_the_turn_reads() {
        let sent = Arc::new(AtomicUsize::new(0));
        let counted = sent.clone();
        let mut handler = open(move |turn, _| {
            let sent = counted.clone();
            Box::pin(async move {
                for n in 0..100 {
                    turn.progress(format!("{n}")).await;
                    sent.fetch_add(1, Ordering::Relaxed);
                }
                Ok(HandlerOutput::default())
            })
        });
        assert!(matches!(handler.next().await, Some(Read::Event(e)) if e == progress("0")));
        assert_eq!(sent.load(Ordering::Relaxed), REPORT_QUEUE);
        for n in 1..100 {
            let expected = progress(&format!("{n}"));
            assert!(matches!(handler.next().await, Some(Read::Event(e)) if e == expected));
        }
        assert!(matches!(handler.next().await, Some(Read::Output(out)) if out.is_ok()));
    }

    // Catches: a close that abandons the handler's cleanup, or waits past the teardown bound.
    #[tokio::test(start_paused = true)]
    async fn closing_cancels_the_handler_and_waits_for_its_cleanup_within_the_deadline() {
        for (cleanup, settles) in [(Duration::from_secs(1), true), (Duration::from_secs(10), false)]
        {
            let cleaned = Arc::new(AtomicBool::new(false));
            let flag = cleaned.clone();
            let mut handler = open(move |turn, cancel| {
                let cleaned = flag.clone();
                Box::pin(async move {
                    tokio::select! {
                        biased;
                        () = cancel.cancelled() => {
                            tokio::time::sleep(cleanup).await;
                            cleaned.store(true, Ordering::Relaxed);
                        }
                        _ = turn.checkpoint() => panic!("a closed checkpoint resumed"),
                    }
                    Ok(HandlerOutput::default())
                })
            });
            assert!(matches!(handler.next().await, Some(Read::Checkpoint)));
            let clock = RealClock::new();
            let deadline = clock.now().saturating_add(Duration::from_secs(5));
            assert_eq!(handler.close(&clock, deadline).await, settles);
            assert_eq!(cleaned.load(Ordering::Relaxed), settles);
        }
    }

    // Catches: a report from a task the handler left behind blocking, or a checkpoint after the
    // turn ended letting the work go on.
    #[tokio::test(start_paused = true)]
    async fn reports_after_the_turn_closed_are_dropped_and_checkpoints_yield() {
        let kept: Arc<Mutex<Option<Arc<dyn HandlerPort>>>> = Arc::default();
        let slot = kept.clone();
        let mut handler = open(move |turn, _| {
            *slot.lock() = Some(turn);
            Box::pin(async { Ok(HandlerOutput::default()) })
        });
        assert!(matches!(handler.next().await, Some(Read::Output(out)) if out.is_ok()));
        assert!(handler.close(&RealClock::new(), Ts::from_nanos(u64::MAX)).await);
        let turn = kept.lock().take().expect("the handler ran");
        for n in 0..2 * REPORT_QUEUE {
            turn.progress(format!("{n}")).await;
        }
        assert!(turn.checkpoint().await);
    }
}
