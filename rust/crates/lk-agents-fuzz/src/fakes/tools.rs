use lk_agents_abi::voice::speech_handle::SpeechHandle as HostSpeech;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use lk_agents_abi::agent::{
    AgentCallbacks, AgentDefinition, AgentFrame, ArgumentIdentity, FunctionToolInfo, Invocation,
    OutputCandidate, OutputSpec, PreparedArguments, ReadBack, ReadBackAnswer, ToolArgumentError,
    ToolTemplate, ValidatedOutput, VoiceAgent,
};
use lk_agents_abi::llm::provider_format::tools::{DuplicateMode, DuplicateScope};
use lk_agents_abi::{FunctionCall, ToolSpec, errors::HostError};
use lk_agents_runtime::voice::speech_handle::SpeechHandle;
use lk_agents_voice::tool_executor::{ToolOutcome, ToolReport, VoiceAgentFrame};
use tokio::sync::oneshot;

use super::{Control, Registry, Slot};
use crate::{
    input::{AgentText, Input, ToolAnswer as Answer},
    record::{Detail, Kind},
};

/// The tool the runs declare with `needs_approval=True`.
pub const APPROVAL_TOOL: &str = "transfer";

/// A name-scoped fixture that rejects another call while its earlier body is still running.
pub const DUPLICATE_REJECT_TOOL: &str = "exclusive_lookup";

/// The tool whose toolset sets `tool_handling` templates: its deferred replies are
/// [`TEMPLATED_REPLY`] plus the updates' ids.
pub const TEMPLATED_TOOL: &str = "notify";

pub const TEMPLATED_REPLY: &str = "Tell the person what changed: ";

pub fn tool_spec(name: &str) -> ToolSpec {
    ToolSpec {
        name: name.to_owned(),
        provider_tool: None,
        description: format!("fixture tool {name}"),
        // A string first: an `InputRequired(tool=)` hands it the person's answer.
        json_schema: serde_json::json!({
            "type": "object",
            "properties": {"label": {"type": "string"}, "seconds": {"type": "integer"}},
        }),
        flags: Default::default(),
    }
}

pub struct FakeTools {
    registry: Arc<Registry>,
    /// Prepared calls' speeches until the body takes them or the preparation drops.
    prepared: Arc<Mutex<Preparations>>,
    refs: Arc<dyn lk_agents_abi::RefRegistry>,
    /// Bodies that answered `Detached` and run on, by call id.
    detached: Mutex<BTreeMap<String, Slot>>,
    /// A cancelled call's task is cancelled, so a body still running cannot detach afterwards
    /// (seed 5818).
    cancelled: Mutex<std::collections::BTreeSet<String>>,
    /// A cancel ends a running body at once, as the host cancels the callback's task, unless the
    /// bodies refuse it.
    running: Mutex<BTreeMap<String, oneshot::Sender<()>>>,
}

impl FakeTools {
    pub fn new(registry: Arc<Registry>) -> Arc<Self> {
        let prepared = Arc::new(Mutex::default());
        let refs: Arc<dyn lk_agents_abi::RefRegistry> =
            Arc::new(PreparedRefs(Arc::clone(&prepared)));
        Arc::new(FakeTools {
            registry,
            prepared,
            refs,
            detached: Mutex::default(),
            cancelled: Mutex::default(),
            running: Mutex::default(),
        })
    }
}

/// `agent`'s voice frame over the fixture `tools`, hosted by a fresh [`FakeTools`].
pub fn frame(registry: &Arc<Registry>, agent: &str, tools: &[String]) -> Arc<dyn AgentFrame> {
    let definition = AgentDefinition {
        name: agent.to_owned(),
        instructions: String::new(),
        dynamic_instructions: false,
        tools: tool_infos(tools),
        output: OutputSpec::Text { validated: false },
        retries: Default::default(),
    };
    let refs: Arc<dyn lk_agents_abi::RefRegistry> = Arc::new(NoRefs);
    let host: Arc<dyn AgentCallbacks> = FakeTools::new(Arc::clone(registry));
    Arc::new(VoiceAgentFrame::new(definition, host, lk_agents_abi::HostRef::new(0, &refs), 0))
}

pub fn tool_infos(tools: &[String]) -> Vec<FunctionToolInfo> {
    tools.iter().map(|name| tool_info(name)).collect()
}

fn tool_info(name: &str) -> FunctionToolInfo {
    FunctionToolInfo {
        spec: tool_spec(name),
        raw_schema: false,
        needs_approval: (name == APPROVAL_TOOL).then_some(ReadBack::Default),
        on_duplicate: if name == DUPLICATE_REJECT_TOOL {
            DuplicateMode::Reject
        } else {
            DuplicateMode::Allow
        },
        duplicate_scope: DuplicateScope::Name,
        retries: None,
        remote: None,
    }
}

fn call_of(invocation: &Invocation) -> &FunctionCall {
    invocation.function_call.as_ref().expect("a tool callback carries its call")
}

/// The value reported for `call_id`'s `Completed`, distinct per call: what a task's `run()` must
/// return.
pub fn completion_value(call_id: &str) -> u64 {
    call_id
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
}

struct NoRefs;

impl lk_agents_abi::RefRegistry for NoRefs {
    fn release(&self, _id: u64) {}
}

/// Prepared arguments no body ran with (a read-back, a refused duplicate) release their speech.
#[derive(Default)]
struct Preparations {
    // Never reuse an id: a body takes its entry before its HostRef drops, and that older ref may
    // release the id after another preparation was inserted.
    next: u64,
    speeches: BTreeMap<u64, Arc<SpeechHandle>>,
}

struct PreparedRefs(Arc<Mutex<Preparations>>);

impl lk_agents_abi::RefRegistry for PreparedRefs {
    fn release(&self, id: u64) {
        self.0.lock().unwrap().speeches.remove(&id);
    }
}

/// What the case tells a running body to do.
pub enum Body {
    Answer(ToolOutcome),
    /// `ctx.wait_for_idle()` first.
    AfterIdle(ToolOutcome),
    /// `await ctx.speech_handle` first: refused, or the body waits on its own speech forever.
    AwaitOwnSpeech(ToolOutcome),
}

impl AgentCallbacks for FakeTools {
    fn render_instructions<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _invocation: &'a Invocation,
    ) -> lk_agents_abi::BoxFuture<'a, Result<String, HostError>> {
        Box::pin(async { Ok(String::new()) })
    }

    fn list_tools<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        invocation: &'a Invocation,
    ) -> lk_agents_abi::BoxFuture<'a, Result<Vec<FunctionToolInfo>, HostError>> {
        let info = tool_info(&call_of(invocation).name);
        Box::pin(async move { Ok(vec![info]) })
    }

    /// The parsed arguments are the approval identity; the preparation keeps its speech for the
    /// body.
    fn prepare_tool<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        invocation: &'a Invocation,
        arguments: serde_json::Map<String, serde_json::Value>,
    ) -> lk_agents_abi::BoxFuture<'a, Result<Result<PreparedArguments, ToolArgumentError>, HostError>>
    {
        let handle = invocation
            .speech
            .as_ref()
            .and_then(|speech| {
                (Arc::clone(speech) as Arc<dyn std::any::Any + Send + Sync>)
                    .downcast::<SpeechHandle>()
                    .ok()
            })
            .filter(|h| h.created().is_some())
            .expect("a tool runs in a created speech");
        let mut prepared = self.prepared.lock().unwrap();
        let id = prepared.next;
        prepared.next = id.checked_add(1).expect("prepared argument ids exhausted");
        prepared.speeches.insert(id, handle);
        let args = PreparedArguments {
            prepared: lk_agents_abi::HostRef::new(id, &self.refs),
            identity: ArgumentIdentity::Json(serde_json::Value::Object(arguments)),
        };
        Box::pin(async move { Ok(Ok(args)) })
    }

    /// Never asked: the approval tool uses the framework's default read-back.
    fn read_back<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _invocation: &'a Invocation,
        _prepared: &'a lk_agents_abi::HostRef,
    ) -> lk_agents_abi::BoxFuture<'a, Result<ReadBackAnswer, HostError>> {
        Box::pin(async { Err(HostError::new("the fixture tools have no read-back function")) })
    }

    fn run_tool<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        invocation: &'a Invocation,
        prepared: &'a lk_agents_abi::HostRef,
    ) -> lk_agents_abi::BoxFuture<'a, Result<lk_agents_abi::agent::ToolCompletion, HostError>> {
        let handle = self
            .prepared
            .lock()
            .unwrap()
            .speeches
            .remove(&prepared.id())
            .expect("a body runs on prepared arguments");
        let call = call_of(invocation).clone();
        Box::pin(async move {
            let mut outcome = self.answer(invocation, &call, &handle).await;
            // A verbatim question names its own tool to take the answer.
            if let ToolOutcome::InputRequired { verbatim: true, tool, .. } = &mut outcome {
                *tool = Some(call.name.clone());
            }
            // Every completion carries a value: whether it ends a task is the core's decision,
            // so the rules see what it decides.
            let transition = matches!(outcome, ToolOutcome::Completed { .. }).then(|| {
                let refs: Arc<dyn lk_agents_abi::RefRegistry> = Arc::new(NoRefs);
                lk_agents_abi::HostRef::new(completion_value(&call.call_id), &refs)
            });
            let report = ToolReport { outcome, transition };
            Ok(lk_agents_abi::agent::ToolCompletion { report, extra: call.extra })
        })
    }

    /// Only [`TEMPLATED_TOOL`]'s deferred replies are rendered; others keep the framework's text.
    fn render_template<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        invocation: &'a Invocation,
        _: lk_agents_abi::agent::ToolTemplateKey,
        template: ToolTemplate,
    ) -> lk_agents_abi::BoxFuture<'a, Result<Option<String>, HostError>> {
        let templated = invocation.function_call.as_ref().is_some_and(|c| c.name == TEMPLATED_TOOL);
        let text = match template {
            ToolTemplate::ReplyAtTail { call_ids }
            | ToolTemplate::ReplyMaybeCovered { call_ids }
                if templated =>
            {
                Some(format!("{TEMPLATED_REPLY}{}", call_ids.join(", ")))
            }
            _ => None,
        };
        Box::pin(async move { Ok(text) })
    }

    fn validate_output<'a>(
        &'a self,
        _cx: &lk_agents_abi::telemetry::Cx,
        _invocation: &'a Invocation,
        _candidate: OutputCandidate,
    ) -> lk_agents_abi::BoxFuture<'a, Result<ValidatedOutput, HostError>> {
        Box::pin(async { Err(HostError::new("a voice reply validates no output")) })
    }

    fn cancel(&self, invocation: &Invocation) {
        let call_id = call_of(invocation).call_id.clone();
        self.cancelled.lock().unwrap().insert(call_id.clone());
        if let Some(cancel) = self.running.lock().unwrap().remove(&call_id) {
            if self.registry.refuses_cancel() {
                self.registry.record_cancel_refused(call_id.clone());
            } else {
                let _ = cancel.send(());
            }
        }
        if let Some(slot) = self.detached.lock().unwrap().remove(&call_id) {
            self.registry.mark_closed(slot);
        }
    }
}

impl FakeTools {
    /// The body's outcome, or `Cancelled` wherever it waits once the driver cancels it.
    async fn answer(
        &self,
        invocation: &Invocation,
        call: &FunctionCall,
        handle: &SpeechHandle,
    ) -> ToolOutcome {
        let speech = handle.id().0;
        let (sender, receiver) = oneshot::channel();
        let slot = self.registry.register(
            Kind::Tool,
            Detail::Tool {
                invocation_id: invocation.invocation_id.clone(),
                operation: invocation.operation,
                call_id: call.call_id.clone(),
                name: call.name.clone(),
                speech,
            },
            Control::Tool(sender),
        );
        let (cancel, cancelled) = oneshot::channel();
        self.running.lock().unwrap().insert(call.call_id.clone(), cancel);
        let outcome = tokio::select! {
            biased;
            Ok(()) = cancelled => None,
            outcome = self.body(invocation, call, handle, slot, receiver) => Some(outcome),
        };
        self.running.lock().unwrap().remove(&call.call_id);
        outcome.unwrap_or_else(|| {
            self.registry.mark_closed(slot);
            ToolOutcome::Cancelled
        })
    }

    async fn body(
        &self,
        invocation: &Invocation,
        call: &FunctionCall,
        handle: &SpeechHandle,
        slot: Slot,
        receiver: oneshot::Receiver<Body>,
    ) -> ToolOutcome {
        match receiver.await {
            Ok(Body::Answer(answer)) => {
                if let ToolOutcome::Update { .. } = answer {
                    // The task was cancelled first: the cancellation, not the detach, ends it.
                    if self.cancelled.lock().unwrap().contains(&call.call_id) {
                        self.registry.mark_closed(slot);
                        return ToolOutcome::Cancelled;
                    }
                    self.detached.lock().unwrap().insert(call.call_id.clone(), slot);
                }
                answer
            }
            Ok(Body::AfterIdle(answer)) => {
                // `async with ctx.foreground():`, asked from the tool's own origin so the driver
                // can tell it from its speech. A refusal at close ends the wait all the same.
                let origin = lk_agents_abi::voice::session::VoiceCommandOrigin::Tool {
                    invocation_id: invocation.invocation_id.clone(),
                    operation: invocation.operation,
                    call_id: call.call_id.clone(),
                };
                if let Some(session) = self.registry.session()
                    && let Ok(hold) =
                        lk_agents_abi::voice::session::VoiceSession::wait_for_idle_and_hold(
                            &**session,
                            Some(origin),
                        )
                    && let Ok(hold) = hold.await
                {
                    lk_agents_abi::voice::session::VoiceSession::release_idle_hold(
                        &**session, hold,
                    );
                }
                answer
            }
            Ok(Body::AwaitOwnSpeech(answer)) => {
                let origin = lk_agents_abi::voice::session::VoiceCommandOrigin::Tool {
                    invocation_id: invocation.invocation_id.clone(),
                    operation: invocation.operation,
                    call_id: call.call_id.clone(),
                };
                let error = HostSpeech::wait_for_playout(handle, Some(origin))
                    .await
                    .err()
                    .map(|e| e.to_string());
                self.registry
                    .record_boundary(crate::record::BoundaryOperation::ToolWaitForPlayout, error);
                answer
            }
            Err(_) => ToolOutcome::Cancelled,
        }
    }
}

/// One answer to a registered tool body. False when the body was already cancelled.
pub(super) fn deliver(sender: oneshot::Sender<Body>, what: &Input) -> bool {
    let Input::ToolAnswered { answer, text: AgentText(text), .. } = what else { return false };
    match answer {
        Answer::AfterIdle => sender.send(Body::AfterIdle(ToolOutcome::Nothing)).is_ok(),
        Answer::AwaitOwnSpeech => sender.send(Body::AwaitOwnSpeech(ToolOutcome::Nothing)).is_ok(),
        _ => sender.send(Body::Answer(outcome_of(*answer, text))).is_ok(),
    }
}

/// The runner's classification of a body that answered `answer` with `text`, as a Python
/// runner classifies what a tool returned or raised.
pub fn outcome_of(answer: Answer, text: &str) -> ToolOutcome {
    let text = text.to_owned();
    let value = |reply_required, verbatim| ToolOutcome::Value {
        text: text.clone(),
        reply_required,
        verbatim,
        more: None,
    };
    match answer {
        Answer::Return | Answer::AfterIdle | Answer::AwaitOwnSpeech => ToolOutcome::Nothing,
        Answer::Value => value(true, false),
        Answer::Recorded => value(false, false),
        Answer::Verbatim => value(true, true),
        Answer::Error => ToolOutcome::ToolError(text),
        Answer::StopResponse => ToolOutcome::StopResponse,
        Answer::Cancelled => ToolOutcome::Cancelled,
        Answer::Detached => ToolOutcome::Update { message: text.clone(), raw: text },
        Answer::Completed => ToolOutcome::Completed { text, verbatim: false },
        Answer::Failed => ToolOutcome::Failed(text),
        Answer::Rejected => ToolOutcome::Rejected(text),
        Answer::Ask => ToolOutcome::InputRequired { question: text, verbatim: false, tool: None },
        Answer::AskVerbatim => {
            ToolOutcome::InputRequired { question: text, verbatim: true, tool: None }
        }
        Answer::Working => ToolOutcome::Working(Some(text)),
    }
}

/// What the host reports for a `ToolUpdate`: its progress message, or its end answered as `end`.
pub fn update_outcome(progress: Option<&str>, end: Answer) -> ToolOutcome {
    match progress {
        Some(message) => ToolOutcome::Update { message: message.into(), raw: message.into() },
        None if end == Answer::Detached => ToolOutcome::Nothing,
        None => outcome_of(end, "the tool finished"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_runtime::net::task::TokioSpawner;
    use lk_agents_runtime::voice::session::Timebase;
    use lk_agents_runtime::voice::speech_handle::Requests;

    use lk_agents_voice::agent_session::machine::activity::scheduling::SpeechId;
    use lk_agents_voice::tool_executor::HostedTool;

    #[tokio::test]
    async fn dropping_started_preparation_does_not_release_a_later_call() {
        let time = Timebase::of_clock(Arc::new(lk_agents_runtime::net::clock::SimClock::new()));
        let (registry, mut records) =
            Registry::new(crate::record::Stamps::new(time), Arc::new(TokioSpawner::current()));
        let tools = FakeTools::new(registry.clone());
        let speech = Arc::new(SpeechHandle::new(SpeechId(1), true, Requests::disconnected()));
        let call = FunctionCall {
            id: "item_first".into(),
            call_id: "first".into(),
            name: "lookup".into(),
            arguments: "{}".into(),
            created_at: std::time::UNIX_EPOCH,
            extra: Default::default(),
            group_id: None,
        };
        let frame = frame(&registry, "agent", &[]);
        let invocation = |call: &FunctionCall| {
            let mut hosted = HostedTool::of(&frame, "s", SpeechId(1), 0, Some(call));
            hosted.invocation.speech = Some(speech.object());
            hosted.invocation
        };
        let first_invocation = invocation(&call);
        let first = tools
            .prepare_tool(
                &lk_agents_abi::telemetry::Cx::NONE,
                &first_invocation,
                Default::default(),
            )
            .await
            .unwrap()
            .unwrap();
        // A body takes its speech at once but its preparation lives until execution settles,
        // so another call may be prepared in between.
        drop(tools.run_tool(
            &lk_agents_abi::telemetry::Cx::NONE,
            &first_invocation,
            &first.prepared,
        ));
        let second_call = FunctionCall { call_id: "second".into(), ..call };
        let second_invocation = invocation(&second_call);
        let second = tools
            .prepare_tool(
                &lk_agents_abi::telemetry::Cx::NONE,
                &second_invocation,
                Default::default(),
            )
            .await
            .unwrap()
            .unwrap();
        drop(first);
        // Checked outside run's lock, so a failure does not panic again while unwinding.
        let second_is_prepared =
            tools.prepared.lock().unwrap().speeches.contains_key(&second.prepared.id());
        assert!(second_is_prepared, "the earlier body's cleanup released the later call");
        let second_body = tools.run_tool(
            &lk_agents_abi::telemetry::Cx::NONE,
            &second_invocation,
            &second.prepared,
        );
        let (report, ()) = tokio::join!(second_body, async {
            let super::super::Record::Opened {
                detail, control: Some(Control::Tool(answer)), ..
            } = records.recv().await.unwrap()
            else {
                panic!("the second body must open a tool invocation")
            };
            assert!(matches!(detail, Detail::Tool { call_id, .. } if call_id == "second"));
            assert!(answer.send(Body::Answer(ToolOutcome::Nothing)).is_ok());
        });
        assert!(matches!(report.unwrap().report.outcome, ToolOutcome::Nothing));
    }
}
