//! VoiceRuntime host callbacks follow their speech's cancellation lifetime, not the session's.

use super::*;
use crate::voice::session::{DriverBuilder, Timebase};
use lk_agents_abi::agent::run::ValidationId;
use lk_agents_abi::agent::{
    AgentCallbacks, FunctionToolInfo, Invocation, OutputCandidate, PreparedArguments,
    ReadBackAnswer, ToolArgumentError, ToolCompletion, ToolTemplate, ValidatedOutput,
};
use lk_agents_abi::errors::HostError;
use lk_agents_abi::{BoxFuture, HostRef};
use lk_agents_net::clock::RealClock;
use lk_agents_net::task::TokioSpawner;
use lk_agents_voice::agent_activity::DriverConfig;
use lk_agents_voice::agent_activity::scheduling::SpeechId;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;
use tokio::sync::Notify;

#[derive(Default)]
struct PendingHost {
    started: Notify,
    dropped: Notify,
    future_dropped: AtomicBool,
    cancellations: AtomicUsize,
}

struct PendingCallback<'a>(&'a PendingHost);

impl Drop for PendingCallback<'_> {
    fn drop(&mut self) {
        self.0.future_dropped.store(true, Ordering::SeqCst);
        self.0.dropped.notify_one();
    }
}

impl PendingHost {
    fn work<T: Send + 'static>(&self) -> BoxFuture<'_, Result<T, HostError>> {
        Box::pin(async move {
            let _pending = PendingCallback(self);
            self.started.notify_one();
            std::future::pending().await
        })
    }

    async fn await_drop(&self) {
        tokio::time::timeout(Duration::from_secs(1), self.dropped.notified())
            .await
            .expect("cancelled host future must be dropped without closing the session");
        assert_eq!(self.cancellations.load(Ordering::SeqCst), 1);
    }
}

impl AgentCallbacks for PendingHost {
    fn render_instructions<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a Invocation,
    ) -> BoxFuture<'a, Result<String, HostError>> {
        unreachable!("only the callback under test is invoked")
    }

    fn list_tools<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a Invocation,
    ) -> BoxFuture<'a, Result<Vec<FunctionToolInfo>, HostError>> {
        unreachable!("only the callback under test is invoked")
    }

    fn prepare_tool<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a Invocation,
        _: serde_json::Map<String, serde_json::Value>,
    ) -> BoxFuture<'a, Result<Result<PreparedArguments, ToolArgumentError>, HostError>> {
        unreachable!("only the callback under test is invoked")
    }

    fn read_back<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a Invocation,
        _: &'a HostRef,
    ) -> BoxFuture<'a, Result<ReadBackAnswer, HostError>> {
        unreachable!("only the callback under test is invoked")
    }

    fn run_tool<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a Invocation,
        _: &'a HostRef,
    ) -> BoxFuture<'a, Result<ToolCompletion, HostError>> {
        unreachable!("only the callback under test is invoked")
    }

    fn validate_output<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a Invocation,
        _: OutputCandidate,
    ) -> BoxFuture<'a, Result<ValidatedOutput, HostError>> {
        self.work()
    }

    fn render_template<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _: &'a Invocation,
        _: ToolTemplate,
    ) -> BoxFuture<'a, Result<Option<String>, HostError>> {
        self.work()
    }

    fn cancel(&self, _: &Invocation) {
        self.cancellations.fetch_add(1, Ordering::SeqCst);
    }
}

fn hosted(host: Arc<PendingHost>, speech: SpeechId, operation: u64) -> HostedTool {
    HostedTool {
        host,
        speech,
        invocation: Invocation {
            operation,
            invocation_id: "callback-lifetime".into(),
            agent_name: "agent".into(),
            execution_id: None,
            frame_id: None,
            context_scope: None,
            application_context: None,
            history: Vec::new(),
            function_call: None,
            approval: None,
            speech: None,
            host_context: None,
        },
    }
}

async fn runtime() -> VoiceRuntime {
    let time = Timebase::of_clock(Arc::new(RealClock::new()));
    let spawner = Arc::new(TokioSpawner::current());
    let builder = DriverBuilder::new(
        "agent".into(),
        time.clone(),
        spawner.clone(),
        DriverConfig::new(
            Default::default(),
            lk_agents_abi::voice::session_options::SessionLegs::none(),
        ),
    );
    let mut executor = crate::voice::Executor::new(time, spawner);
    let (cores, _handle) = builder.start(&mut executor).await.expect("session cores");
    VoiceRuntime::new(cores, Vec::new(), None)
}

#[tokio::test(start_paused = true)]
async fn cancelling_one_validation_preserves_other_callbacks_until_speech_cancellation() {
    let mut runtime = runtime().await;
    let speech = SpeechId(7);
    let first = Arc::new(PendingHost::default());
    let other = Arc::new(PendingHost::default());
    for (id, host) in [(1, first.clone()), (2, other.clone())] {
        runtime.perform_voice_effect(VoiceEffect::ValidateOutput {
            speech,
            validation: ValidationId(id),
            candidate: OutputCandidate::Text("candidate".into()),
            hosted: hosted(host, speech, id),
        });
    }
    first.started.notified().await;
    other.started.notified().await;

    runtime.perform_voice_effect(VoiceEffect::CancelOutputValidation {
        speech,
        validation: ValidationId(1),
    });
    first.await_drop().await;
    let receipts: Vec<_> = std::iter::from_fn(|| runtime.queue.try_recv().ok())
        .filter_map(|event| match event {
            crate::voice::VoiceRuntimeEvent::Input(VoiceEvent::Gen(
                GenEvent::OutputValidated { validation, verdict, value, .. },
            )) => Some((validation, verdict, value)),
            _ => None,
        })
        .collect();
    assert_eq!(receipts.len(), 1, "cancellation must settle the shared run's owed validator");
    assert_eq!(receipts[0].0, ValidationId(1));
    assert!(matches!(receipts[0].1, lk_agents_core::agent_run::OutputVerdict::Failed(_)));
    assert!(receipts[0].2.is_none());
    assert!(
        !other.future_dropped.load(Ordering::SeqCst),
        "exact cancellation killed another validator"
    );
    assert!(
        !runtime
            .host_callbacks
            .contains_key(&(speech, HostCallbackId::OutputValidation(ValidationId(1))))
    );
    assert!(
        runtime
            .host_callbacks
            .contains_key(&(speech, HostCallbackId::OutputValidation(ValidationId(2))))
    );
    assert!(!*runtime.shared.closing.borrow(), "callback cancellation closed the session");

    runtime.perform_voice_effect(VoiceEffect::CancelSpeech(speech));
    other.await_drop().await;
    assert!(runtime.host_callbacks.is_empty());
    assert!(!*runtime.shared.closing.borrow());
}

#[tokio::test(start_paused = true)]
async fn speech_cancellation_ends_refusal_rendering_but_preserves_deferred_reply_rendering() {
    let mut runtime = runtime().await;
    let speech = SpeechId(8);
    let refusal = Arc::new(PendingHost::default());
    let deferred = Arc::new(PendingHost::default());
    let template = || ToolTemplate::ReplyAtTail { call_ids: vec!["call".into()] };
    runtime.perform_voice_effect(VoiceEffect::RenderToolHandlingTemplate {
        rendering: Rendering::Refusal { speech, call: lk_agents_core::agent_run::CallId(1) },
        template: template(),
        hosted: hosted(refusal.clone(), speech, 1),
    });
    runtime.perform_voice_effect(VoiceEffect::RenderToolHandlingTemplate {
        rendering: Rendering::ToolReply { render: 2 },
        template: template(),
        hosted: hosted(deferred.clone(), speech, 2),
    });
    refusal.started.notified().await;
    deferred.started.notified().await;

    runtime.perform_voice_effect(VoiceEffect::CancelSpeech(speech));
    refusal.await_drop().await;
    assert!(
        !deferred.future_dropped.load(Ordering::SeqCst),
        "background tool reply lost its owner"
    );
    assert!(!*runtime.shared.closing.borrow());

    runtime.tasks.close();
    deferred.await_drop().await;
}

#[tokio::test(start_paused = true)]
async fn cancelling_one_refusal_reports_its_receipt_and_preserves_other_renderers() {
    let mut runtime = runtime().await;
    let speech = SpeechId(18);
    let call = lk_agents_core::agent_run::CallId(1);
    let first = Arc::new(PendingHost::default());
    let sibling = Arc::new(PendingHost::default());
    let another_speech = Arc::new(PendingHost::default());
    let deferred = Arc::new(PendingHost::default());
    for (operation, (rendering, host)) in [
        (Rendering::Refusal { speech, call }, first.clone()),
        (
            Rendering::Refusal { speech, call: lk_agents_core::agent_run::CallId(2) },
            sibling.clone(),
        ),
        (Rendering::Refusal { speech: SpeechId(19), call }, another_speech.clone()),
        (Rendering::ToolReply { render: 3 }, deferred.clone()),
    ]
    .into_iter()
    .enumerate()
    {
        let owner = match rendering {
            Rendering::Refusal { speech, .. } => speech,
            Rendering::ToolReply { .. } => speech,
        };
        runtime.perform_voice_effect(VoiceEffect::RenderToolHandlingTemplate {
            rendering,
            template: ToolTemplate::ReplyAtTail { call_ids: vec!["reused".into()] },
            hosted: hosted(host, owner, operation as u64 + 1),
        });
    }
    for host in [&first, &sibling, &another_speech, &deferred] {
        host.started.notified().await;
    }

    runtime.perform_voice_effect(VoiceEffect::CancelToolRefusal { speech, call });
    first.await_drop().await;
    let receipts: Vec<_> = std::iter::from_fn(|| runtime.queue.try_recv().ok())
        .filter_map(|event| match event {
            crate::voice::VoiceRuntimeEvent::Input(VoiceEvent::Gen(
                GenEvent::ToolHandlingTemplateRendered { rendering, text },
            )) => Some((rendering, text)),
            _ => None,
        })
        .collect();
    assert_eq!(receipts, [(Rendering::Refusal { speech, call }, None)]);
    for host in [&sibling, &another_speech, &deferred] {
        assert!(!host.future_dropped.load(Ordering::SeqCst));
        assert_eq!(host.cancellations.load(Ordering::SeqCst), 0);
    }
    runtime.perform_voice_effect(VoiceEffect::CancelToolRefusal { speech, call });
    assert_eq!(first.cancellations.load(Ordering::SeqCst), 1, "cancellation is idempotent");
    assert!(runtime.queue.try_recv().is_err(), "one renderer reports one terminal receipt");
    assert!(!*runtime.shared.closing.borrow(), "a call cancellation must not close the session");

    runtime.tasks.close();
    for host in [&sibling, &another_speech, &deferred] {
        host.await_drop().await;
    }
}

#[tokio::test(start_paused = true)]
async fn releasing_a_speech_cancels_its_callbacks_but_preserves_detached_tool_work() {
    let mut runtime = runtime().await;
    let speech = SpeechId(9);
    let validation = Arc::new(PendingHost::default());
    let refusal = Arc::new(PendingHost::default());
    let deferred = Arc::new(PendingHost::default());
    let body = Arc::new(PendingHost::default());
    runtime.perform_voice_effect(VoiceEffect::ValidateOutput {
        speech,
        validation: ValidationId(1),
        candidate: OutputCandidate::Text("candidate".into()),
        hosted: hosted(validation.clone(), speech, 1),
    });
    for (rendering, host) in [
        (
            Rendering::Refusal { speech, call: lk_agents_core::agent_run::CallId(1) },
            refusal.clone(),
        ),
        (Rendering::ToolReply { render: 2 }, deferred.clone()),
    ] {
        runtime.perform_voice_effect(VoiceEffect::RenderToolHandlingTemplate {
            rendering,
            template: ToolTemplate::ReplyAtTail { call_ids: vec!["call".into()] },
            hosted: hosted(host, speech, 2),
        });
    }
    let running_body = body.clone();
    runtime.speeches.spawn(speech, "detached tool body", async move {
        let _ = running_body.work::<()>().await;
    });
    for host in [&validation, &refusal, &deferred, &body] {
        host.started.notified().await;
    }

    runtime.perform_voice_effect(VoiceEffect::Speech(
        speech,
        lk_agents_voice::speech_handle::SpeechUpdate::Released,
    ));
    validation.await_drop().await;
    refusal.await_drop().await;
    assert!(runtime.host_callbacks.is_empty());
    assert!(!deferred.future_dropped.load(Ordering::SeqCst));
    assert!(!body.future_dropped.load(Ordering::SeqCst), "release cancelled a detached tool body");
    assert!(!*runtime.shared.closing.borrow());

    runtime.tasks.close();
    deferred.await_drop().await;
    tokio::time::timeout(Duration::from_secs(1), body.dropped.notified())
        .await
        .expect("session-owned work is cancelled by session teardown");
}
