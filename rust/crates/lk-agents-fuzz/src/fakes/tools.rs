//! Tool bodies: every call the product hands to the runner is an invocation with the actual
//! `call_id`, name, speech and step; the body waits for whatever the case delivers. The runner is
//! the host's (`AgentHost`), classifying what the body did, as a Python agent's host does.
use lk_agents_abi::voice::speech_handle::SpeechHandle as HostSpeech;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use lk_agents_abi::agent::{
    AgentDefinition, AgentFrame, AgentHost, ArgumentIdentity, FunctionToolInfo, Invocation,
    OutputCandidate, OutputSpec, PreparedArguments, ReadBack, ReadBackAnswer, ToolArgumentError,
    ToolTemplate, ValidatedOutput, VoiceAgent,
};
use lk_agents_abi::llm::provider_format::tools::{DuplicateMode, DuplicateScope};
use lk_agents_abi::{FunctionCall, ToolSpec, errors::HostError};
use lk_agents_voice::speech_handle::SpeechHandle;
use lk_agents_voice::tool_executor::{HostedTool, ToolOutcome, ToolReport, VoiceAgentFrame};
use tokio::sync::oneshot;

use super::{Control, Registry, Slot};
use crate::{
    input::{AgentText, Input, ToolAnswer as Answer},
    record::{Detail, Kind},
};

/// The tool the runs declare with `needs_approval=True`.
pub const APPROVAL_TOOL: &str = "transfer";

/// The tool whose toolset sets `tool_handling` templates: its deferred replies are told
/// [`TEMPLATED_REPLY`] and the updates' ids.
pub const TEMPLATED_TOOL: &str = "notify";

/// What [`TEMPLATED_TOOL`]'s deferred reply instructions start with.
pub const TEMPLATED_REPLY: &str = "Tell the person what changed: ";

/// A fixture tool as the model sees it.
pub fn tool_spec(name: &str) -> ToolSpec {
    ToolSpec {
        name: name.to_owned(),
        provider_tool: None,
        description: format!("fixture tool {name}"),
        // The shapes `ToolArgs` draws: a string first, which an `InputRequired(tool=)` hands the
        // person's answer to.
        json_schema: serde_json::json!({
            "type": "object",
            "properties": {"label": {"type": "string"}, "seconds": {"type": "integer"}},
        }),
        flags: Default::default(),
    }
}

pub struct FakeTools {
    registry: Arc<Registry>,
    /// Prepared calls' speeches by prepared id, until the body takes them; what a dropped
    /// preparation releases.
    prepared: Arc<Mutex<Preparations>>,
    refs: Arc<dyn lk_agents_abi::RefRegistry>,
    /// The bodies that answered `Detached` and run on, by call id: what a cancel ends.
    detached: Mutex<BTreeMap<String, Slot>>,
    /// The calls the driver cancelled: their task is cancelled, so a body still running cannot
    /// detach afterwards (seed 5818).
    cancelled: Mutex<std::collections::BTreeSet<String>>,
    /// The bodies still running, by call id: a cancel ends one at once, as the host cancels the
    /// callback's task (`AgentHost.cancel`).
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
        tools: tools.iter().map(|name| tool_info(name)).collect(),
        output: OutputSpec::Text { validated: false },
        retries: Default::default(),
    };
    let refs: Arc<dyn lk_agents_abi::RefRegistry> = Arc::new(NoRefs);
    let host: Arc<dyn AgentHost> = FakeTools::new(Arc::clone(registry));
    Arc::new(VoiceAgentFrame::new(definition, host, lk_agents_abi::HostRef::new(0, &refs)))
}

/// The fixture tool `name`; [`APPROVAL_TOOL`] requires approval with the default read-back.
fn tool_info(name: &str) -> FunctionToolInfo {
    FunctionToolInfo {
        spec: tool_spec(name),
        raw_schema: false,
        needs_approval: (name == APPROVAL_TOOL).then_some(ReadBack::Default),
        on_duplicate: DuplicateMode::Allow,
        duplicate_scope: DuplicateScope::Name,
        retries: None,
        remote: None,
    }
}

/// The tool call a callback is about.
fn call_of(invocation: &Invocation) -> &FunctionCall {
    invocation.function_call.as_ref().expect("a tool callback carries its call")
}

/// The value the runner reports for call `call_id`'s `Completed` in a task: the object `run()` must
/// return, told apart from every other call's.
pub fn completion_value(call_id: &str) -> u64 {
    call_id
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
}

/// The side table a completion's value points into; nothing holds the object.
struct NoRefs;

impl lk_agents_abi::RefRegistry for NoRefs {
    fn release(&self, _id: u64) {}
}

/// Prepared arguments no body ran with (a read-back, a refused duplicate) release their speech.
#[derive(Default)]
struct Preparations {
    // A body takes the entry before the corresponding HostRef is dropped. Never reuse its id:
    // that older reference may still release it after another preparation has been inserted.
    next: u64,
    speeches: BTreeMap<u64, Arc<SpeechHandle>>,
}

struct PreparedRefs(Arc<Mutex<Preparations>>);

impl lk_agents_abi::RefRegistry for PreparedRefs {
    fn release(&self, id: u64) {
        self.0.lock().unwrap().speeches.remove(&id);
    }
}

/// What the case tells a running body to do: answer, or first wait for the agent to go idle --
/// `ctx.wait_for_idle()` -- and answer once it has.
pub enum Body {
    Answer(ToolOutcome),
    AfterIdle(ToolOutcome),
    /// `await ctx.speech_handle` first: refused, or the body waits on its own speech forever.
    AwaitOwnSpeech(ToolOutcome),
}

impl AgentHost for FakeTools {
    fn render_instructions<'a>(
        &'a self,
        _invocation: &'a Invocation,
    ) -> lk_agents_abi::BoxFuture<'a, Result<String, HostError>> {
        Box::pin(async { Ok(String::new()) })
    }

    /// The called fixture tool; [`APPROVAL_TOOL`] requires approval with the default read-back.
    fn list_tools<'a>(
        &'a self,
        invocation: &'a Invocation,
    ) -> lk_agents_abi::BoxFuture<'a, Result<Vec<FunctionToolInfo>, HostError>> {
        let info = tool_info(&call_of(invocation).name);
        Box::pin(async move { Ok(vec![info]) })
    }

    /// Binds nothing: the parsed arguments are the identity approval binds to, and the prepared
    /// call keeps its speech for the body.
    fn prepare_tool<'a>(
        &'a self,
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

    /// The runs' approval tool reads back with the framework's default: never asked here.
    fn read_back<'a>(
        &'a self,
        _invocation: &'a Invocation,
        _prepared: &'a lk_agents_abi::HostRef,
    ) -> lk_agents_abi::BoxFuture<'a, Result<ReadBackAnswer, HostError>> {
        Box::pin(async { Err(HostError::new("the fixture tools have no read-back function")) })
    }

    fn run_tool<'a>(
        &'a self,
        invocation: &'a Invocation,
        prepared: &'a lk_agents_abi::HostRef,
    ) -> lk_agents_abi::BoxFuture<
        'a,
        Result<(ToolReport, serde_json::Map<String, serde_json::Value>), HostError>,
    > {
        let handle = self
            .prepared
            .lock()
            .unwrap()
            .speeches
            .remove(&prepared.id())
            .expect("a body runs on prepared arguments");
        let call = call_of(invocation).clone();
        Box::pin(async move {
            let mut outcome = self.answer(&call, &handle).await;
            // A question asked as written names its own tool to take the answer
            // (`InputRequired(question, tool=..., verbatim=True)`); one in the agent's words does
            // not.
            if let ToolOutcome::InputRequired { verbatim: true, tool, .. } = &mut outcome {
                *tool = Some(call.name.clone());
            }
            // Every completion is handed over with a value, in a task's speech or not: whether it
            // ends a task is the core's to decide (a host that validated an ordinary tool's
            // `Completed` as a task's once silenced the call), so the rules see what it decides.
            let transition = matches!(outcome, ToolOutcome::Completed { .. }).then(|| {
                let refs: Arc<dyn lk_agents_abi::RefRegistry> = Arc::new(NoRefs);
                lk_agents_abi::HostRef::new(completion_value(&call.call_id), &refs)
            });
            Ok((ToolReport { outcome, transition }, call.extra))
        })
    }

    /// [`TEMPLATED_TOOL`]'s toolset renders its deferred replies; every other template keeps
    /// the framework's text.
    fn render_template<'a>(
        &'a self,
        invocation: &'a Invocation,
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
        _invocation: &'a Invocation,
        _candidate: OutputCandidate,
    ) -> lk_agents_abi::BoxFuture<'a, Result<ValidatedOutput, HostError>> {
        Box::pin(async { Err(HostError::new("a voice reply validates no output")) })
    }

    fn cancel(&self, invocation: &Invocation) {
        let call_id = call_of(invocation).call_id.clone();
        self.cancelled.lock().unwrap().insert(call_id.clone());
        if let Some(cancel) = self.running.lock().unwrap().remove(&call_id) {
            let _ = cancel.send(());
        }
        if let Some(slot) = self.detached.lock().unwrap().remove(&call_id) {
            self.registry.mark_closed(slot);
        }
    }
}

impl FakeTools {
    /// The body's outcome, or `Cancelled` once the driver cancels it: the callback's task is
    /// cancelled wherever it waits.
    async fn answer(&self, call: &FunctionCall, handle: &SpeechHandle) -> ToolOutcome {
        let speech = handle.id().0;
        let (sender, receiver) = oneshot::channel();
        let slot = self.registry.register(
            Kind::Tool,
            Detail::Tool { call_id: call.call_id.clone(), name: call.name.clone(), speech },
            Control::Tool(sender),
        );
        let (cancel, cancelled) = oneshot::channel();
        self.running.lock().unwrap().insert(call.call_id.clone(), cancel);
        let outcome = tokio::select! {
            biased;
            Ok(()) = cancelled => None,
            outcome = self.body(call, handle, slot, receiver) => Some(outcome),
        };
        self.running.lock().unwrap().remove(&call.call_id);
        outcome.unwrap_or_else(|| {
            self.registry.mark_closed(slot);
            ToolOutcome::Cancelled
        })
    }

    async fn body(
        &self,
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
                // `async with ctx.foreground():` in the body: idle asked from the tool's own
                // origin (so the driver can tell it apart from its speech), the floor held, and
                // released on the way out. Refused by the close, the wait ends all the same.
                let origin = format!("tool:{}", call.call_id);
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
                let origin = format!("tool:{}", call.call_id);
                let _ = HostSpeech::wait_for_playout(&*handle, Some(origin)).await;
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

/// What the runner reports for a body that answered `answer` with `text`, as a Python tool's
/// runner classifies the value it returned or raised.
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

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_runtime::net::task::TokioSpawner;
    use lk_agents_voice::{
        agent_activity::scheduling::SpeechId,
        agent_session::{Timebase, Tx},
    };

    #[tokio::test]
    async fn dropping_started_preparation_does_not_release_a_later_call() {
        let time = Timebase::of_clock(Arc::new(lk_agents_abi::SimClock::new()));
        let (registry, mut records) =
            Registry::new(crate::record::Stamps::new(time), Arc::new(TokioSpawner::current()));
        let tools = FakeTools::new(registry.clone());
        let speech = Arc::new(SpeechHandle::new(SpeechId(1), true, Tx::disconnected()));
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
        let invocation =
            |call: &FunctionCall| HostedTool::of(&frame, "s", &speech, 0, Some(call)).invocation;
        let first_invocation = invocation(&call);
        let first =
            tools.prepare_tool(&first_invocation, Default::default()).await.unwrap().unwrap();
        // A body takes its speech immediately, while its host preparation remains alive until
        // execution settles. Another call may be prepared during that interval.
        drop(tools.run_tool(&first_invocation, &first.prepared));
        let second_call = FunctionCall { call_id: "second".into(), ..call };
        let second_invocation = invocation(&second_call);
        let second =
            tools.prepare_tool(&second_invocation, Default::default()).await.unwrap().unwrap();
        drop(first);
        // Check before run's assertion acquires the mutex, so the failing regression leaves
        // preparation cleanup usable rather than causing a second panic during unwinding.
        let second_is_prepared =
            tools.prepared.lock().unwrap().speeches.contains_key(&second.prepared.id());
        assert!(second_is_prepared, "the earlier body's cleanup released the later call");
        let second_body = tools.run_tool(&second_invocation, &second.prepared);
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
        assert!(matches!(report.unwrap().0.outcome, ToolOutcome::Nothing));
    }
}
