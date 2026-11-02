use lk_agents_abi::agent::run::{StreamEvent, ToolResultStatus};
use lk_agents_abi::agent::{AgentDefinition, FunctionToolInfo, OutputSpec, ReadBack, Retries};
use lk_agents_abi::llm::provider_format::tools::{DuplicateMode, DuplicateScope};
use lk_agents_abi::resources::{LlmRef, ResourceId};
use lk_agents_abi::{
    ChatContext, ChoiceDelta, FunctionToolCall, LlmChunk, ProviderId, Role, ToolSpec,
};
use serde_json::json;

use super::*;

mod checkpoints {
    //! Yielding a child preserves its originating parent call and active frame.

    use super::*;

    fn verdict(driver: &mut Driver, yielded: bool) {
        let checkpoint = driver
            .effects
            .iter()
            .rev()
            .find_map(|effect| match effect {
                TextSessionEffect::Run {
                    effect: agent_run::AgentRunEffect::HostCheckpoint { checkpoint, .. },
                    ..
                } => Some(*checkpoint),
                _ => None,
            })
            .expect("the active frame awaits a checkpoint");
        driver.run(agent_run::AgentRunEvent::HostCheckpointDone {
            checkpoint,
            result: Ok(yielded),
            tools: None,
        });
    }

    #[test]
    fn a_yielded_child_remains_active_without_resolving_its_parent_call() {
        let config = config(false, ChatHistory::Fresh, false);
        let mut driver = Driver {
            session: TextSession::new(config.clone(), input(None)).unwrap().with_checkpoints(),
            config,
            effects: Vec::new(),
            auto_release: true,
        };
        driver.send(TextSessionEvent::Start(message("human", Role::User, "Book a room")));
        verdict(&mut driver, false);
        driver.reply("", &[("book", json!({"nights": 2}))]);
        driver.prepare(json!({"nights": 2}));
        verdict(&mut driver, true);

        let turn = driver.turn();
        assert!(turn.result.interrupted);
        assert_ne!(turn.state.active_frame_id, turn.state.root_frame_id);
        let active = &turn.state.frames[&turn.state.active_frame_id];
        assert!(!active.completed);
        let parent_call = active.parent.as_ref().expect("the originating call remains");
        let parent = &turn.state.frames[&parent_call.frame_id];
        assert!(parent.pending_calls.iter().any(|pending| pending.parent == *parent_call
            && pending.status == ChildCallStatus::Running { frame_id: active.id.clone() }));
        assert!(!parent.history.iter().any(|item| matches!(item,
            ChatItem::FunctionCallOutput(output) if output.call_id == parent_call.call_id
        )));
        assert_eq!(
            driver
                .effects
                .iter()
                .filter(|effect| matches!(
                    effect,
                    TextSessionEffect::Run {
                        effect: agent_run::AgentRunEffect::LlmRequestStart { .. },
                        ..
                    }
                ))
                .count(),
            1
        );
    }
}

fn message(id: &str, role: Role, text: &str) -> ChatMessage {
    ChatMessage::text(id, role, text, std::time::UNIX_EPOCH)
}

fn config(approval: bool, mode: ChatHistory, merge: bool) -> TextSessionConfig {
    let tool = FunctionToolInfo {
        spec: ToolSpec {
            provider_tool: None,
            name: "book".into(),
            description: "Book a room.".into(),
            json_schema: json!({"type":"object", "properties":{"nights":{"type":"integer"}}}),
            flags: Default::default(),
        },
        raw_schema: false,
        needs_approval: approval.then_some(ReadBack::Default),
        on_duplicate: DuplicateMode::Allow,
        duplicate_scope: DuplicateScope::Name,
        retries: None,
        remote: None,
    };
    let agent = |name: &str, tools, output| SessionAgent {
        definition: AgentDefinition {
            name: name.into(),
            instructions: format!("You are {name}."),
            dynamic_instructions: false,
            tools,
            output,
            retries: Retries::default(),
        },
        llm: Some(LlmRef { id: ResourceId(1), provider: ProviderId::of_label("fake") }),
        output_type_explicit: true,
        agent_tools: BTreeMap::new(),
    };
    let mut root = agent("desk", vec![tool], OutputSpec::Text { validated: false });
    root.agent_tools.insert(
        "book".into(),
        LocalAgentTool {
            agent_name: "booking".into(),
            chat_history: mode,
            merge_chat_history: merge,
            context_scope: Some("draft".into()),
        },
    );
    let child = agent(
        "booking",
        vec![],
        OutputSpec::Structured {
            schema: json!({"type":"object", "properties":{"reference":{"type":"string"}}}),
        },
    );
    TextSessionConfig::new(
        "desk".into(),
        BTreeMap::from([("desk".into(), root), ("booking".into(), child)]),
    )
}

fn input(state: Option<TextSessionState>) -> TextSessionInput {
    TextSessionInput {
        session_id: "conversation".into(),
        execution_id: "task-42".into(),
        task_id: None,
        root_scope: "customer".into(),
        available_scopes: BTreeSet::from(["customer".into(), "draft".into()]),
        separate_contexts: false,
        state,
        wall: WallAnchor { wall: 1_700_000_000.0, at: Ts::ZERO },
        cleanup_timeout: Duration::from_secs(1),
        delegated: true,
    }
}

struct Driver {
    session: TextSession,
    effects: Vec<TextSessionEffect>,
    config: TextSessionConfig,
    auto_release: bool,
}

impl Driver {
    fn new(config: TextSessionConfig, state: Option<TextSessionState>, text: &str) -> Self {
        let mut driver = Self {
            session: TextSession::new(config.clone(), input(state)).unwrap(),
            effects: Vec::new(),
            config,
            auto_release: true,
        };
        let id = format!("human-{}", driver.session.state.next_id);
        driver.send(TextSessionEvent::Start(message(&id, Role::User, text)));
        driver
    }

    fn send(&mut self, event: TextSessionEvent) -> Vec<TextSessionEffect> {
        let effects = self.session.handle(event, Ts::ZERO);
        self.effects.extend(effects.iter().cloned());
        if self.auto_release {
            for invocation_id in effects.iter().filter_map(|e| match e {
                TextSessionEffect::InvocationSettled { invocation_id, .. }
                | TextSessionEffect::InvocationParked { invocation_id, .. } => {
                    Some(invocation_id.clone())
                }
                _ => None,
            }) {
                let delivered = Default::default();
                self.send(TextSessionEvent::InvocationReleased { invocation_id, delivered });
            }
        }
        effects
    }

    fn invocation(&self) -> String {
        self.session.invocation.as_ref().unwrap().id.clone()
    }

    fn run(&mut self, event: agent_run::AgentRunEvent) -> Vec<TextSessionEffect> {
        self.send(TextSessionEvent::Run { invocation_id: self.invocation(), event })
    }

    fn request(&self) -> (agent_run::RequestId, ChatContext) {
        self.effects
            .iter()
            .rev()
            .find_map(|effect| match effect {
                TextSessionEffect::Run {
                    effect: agent_run::AgentRunEffect::LlmRequestStart { request, chat_ctx, .. },
                    ..
                } => Some((*request, chat_ctx.clone())),
                _ => None,
            })
            .unwrap()
    }

    fn reply(&mut self, text: &str, calls: &[(&str, Value)]) {
        let (request, _) = self.request();
        let id = self.invocation();
        let chunk = LlmChunk {
            id: "chunk".into(),
            delta: Some(ChoiceDelta {
                role: Some(Role::Assistant),
                content: (!text.is_empty()).then(|| text.into()),
                tool_calls: calls
                    .iter()
                    .enumerate()
                    .map(|(n, (name, args))| FunctionToolCall {
                        name: (*name).into(),
                        arguments: args.to_string(),
                        call_id: format!("{id}-{request:?}-{n}"),
                        extra: None,
                    })
                    .collect(),
                extra: None,
            }),
            usage: None,
            flush: false,
            node_text: false,
        };
        self.run(agent_run::AgentRunEvent::LlmChunk { request, chunk });
        self.run(agent_run::AgentRunEvent::LlmEnded { request });
    }

    fn prepare(&mut self, arguments: Value) {
        let invocation = self.invocation();
        let call = self
            .effects
            .iter()
            .rev()
            .find_map(|e| match e {
                TextSessionEffect::Run {
                    invocation_id,
                    effect: agent_run::AgentRunEffect::ToolPrepare { call, .. },
                    ..
                } if invocation_id == &invocation => Some(*call),
                _ => None,
            })
            .unwrap();
        self.run(agent_run::AgentRunEvent::ToolPrepared {
            call,
            prepared: ResourceId(100),
            identity: ArgumentIdentity::Json(arguments),
        });
    }

    fn accepted(&mut self, value: Value) {
        self.reply("", &[("final_result", value.clone())]);
        let validation = self
            .effects
            .iter()
            .rev()
            .find_map(|e| match e {
                TextSessionEffect::Run {
                    effect: agent_run::AgentRunEffect::ValidateOutput { validation, .. },
                    ..
                } => Some(*validation),
                _ => None,
            })
            .unwrap();
        self.send(TextSessionEvent::OutputValidated {
            invocation_id: self.invocation(),
            validation,
            verdict: OutputVerdict::Accepted { text: None },
            output: Some(value),
        });
    }

    fn turn(&self) -> TextSessionTurn {
        self.effects
            .iter()
            .find_map(|e| match e {
                TextSessionEffect::Finished { result: Ok(turn), .. } => Some(turn.clone()),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no successful turn: {:?}", self.effects))
    }

    fn reconstruct(&self, text: &str) -> Self {
        // All execution objects are dropped by the caller. Only the owned public snapshot
        // crosses this boundary, just as the host codec reconstructs its ChatItem records.
        Self::new(self.config.clone(), Some(self.turn().state), text)
    }
}

#[test]
fn text_session_reconstruction_preserves_approval_child_and_once_only_parent_resolution() {
    let cfg = config(true, ChatHistory::Fork, true);
    let mut first = Driver::new(cfg, None, "Book a room");
    first.reply("", &[("book", json!({"nights":"2"}))]);
    first.prepare(json!({"nights":2}));
    let waiting = first.turn();
    let root_id = waiting.state.root_frame_id.clone();
    let pending = &waiting.state.frames[&root_id].pending_calls[0];
    assert_eq!(pending.status, ChildCallStatus::Approval);
    assert_eq!(pending.execution_id, "task-42");
    assert_eq!(pending.validated_arguments, json!({"nights":2}));
    assert_eq!(waiting.state.frames.len(), 1);

    let mut second = first.reconstruct("yes");
    second.reply("", &[("book", json!({"nights":2}))]);
    second.prepare(json!({"nights":2}));
    let child_id = second.session.state.active_frame_id.clone();
    assert!(second.effects.iter().any(|e| matches!(e, TextSessionEffect::InvocationParked { .. })));
    assert_ne!(child_id, root_id);
    assert_eq!(second.session.state.ancestors.as_slice(), std::slice::from_ref(&root_id));
    assert_eq!(second.session.state.frames[&child_id].context_scope, "draft");
    let call = second.session.state.frames[&child_id].parent.clone().unwrap();
    assert_eq!(call.frame_id, root_id);
    assert_eq!(
        second.request().1.items.last().unwrap().as_message().unwrap().raw_text_content().unwrap(),
        "{\"nights\":2}"
    );
    second.auto_release = false;
    second.reply("Which date?", &[]);
    let result = second
        .effects
        .iter()
        .find_map(|e| match e {
            TextSessionEffect::InvocationSettled { result, .. } => Some(result),
            _ => None,
        })
        .unwrap();
    let question_message = result.all_messages.last().unwrap().as_message().unwrap().clone();
    let invocation_id = second.invocation();
    second.send(TextSessionEvent::DialogueDelivered {
        invocation_id: invocation_id.clone(),
        message: question_message.clone(),
    });
    second.send(TextSessionEvent::InvocationReleased {
        invocation_id,
        delivered: Default::default(),
    });
    let mut third = Driver::new(second.config.clone(), Some(second.turn().state), "Friday");
    assert_eq!(third.session.state.active_frame_id, child_id);
    assert_eq!(
        third.session.state.frames[&root_id].history,
        second.session.state.frames[&root_id].history
    );
    assert!(third.request().1.items.iter().any(|i| {
        i.as_message().is_some_and(|m| m.raw_text_content().as_deref() == Some("Friday"))
    }));
    assert!(!third.session.state.frames[&root_id].history.iter().any(|i| {
        i.as_message().is_some_and(|m| m.raw_text_content().as_deref() == Some("Friday"))
    }));
    third.accepted(json!({"reference":"R-1"}));
    let parent_request = third.request().1;
    assert_eq!(parent_request.items.iter().filter(|i| matches!(i, ChatItem::FunctionCallOutput(o) if o.call_id == call.call_id && o.output == "{\"reference\":\"R-1\"}")).count(), 1);
    assert!(third.session.state.frames[&child_id].completed);
    assert!(third.session.state.ancestors.is_empty());
    assert!(parent_request.items.iter().any(|i| i.id() == question_message.id));
    assert!(!parent_request.items.iter().any(|i| {
        i.as_message().is_some_and(|m| m.raw_text_content().as_deref() == Some("{\"nights\":2}"))
    }));
    let parent_invocation = third.invocation();
    assert_ne!(parent_invocation, pending.invocation_id);
    let stale = third.send(TextSessionEvent::Run {
        invocation_id: pending.invocation_id.clone(),
        event: agent_run::AgentRunEvent::Cancel,
    });
    assert!(stale.is_empty());
    third.reply("Reserved", &[]);
    let complete = third.turn();
    // usage.md "Scope and delegation": the turn counts the child's and the resumed parent's
    // requests once each, not only the final invocation's.
    assert_eq!((complete.result.usage.requests, complete.result.usage.settled), (2, true));
    let mut fourth = third.reconstruct("Thanks");
    fourth.reply("Welcome", &[]);
    assert_eq!(
        fourth.turn().state.frames[&root_id]
            .history
            .iter()
            .filter(|i| i.id() == question_message.id)
            .count(),
        1
    );
}

// An answered read-back cannot authorize a later turn, including one spent in a child.
#[test]
fn an_answer_without_a_repeated_call_consumes_the_child_approval() {
    for answer in ["no", "yes"] {
        for delegate in [false, true] {
            let mut cfg = config(true, ChatHistory::Fresh, false);
            let root = cfg.agents.get_mut("desk").unwrap();
            let mut details = root.definition.tools[0].clone();
            details.spec.name = "details".into();
            details.needs_approval = None;
            root.definition.tools.push(details);
            root.agent_tools.insert("details".into(), root.agent_tools["book"].clone());
            let mut first = Driver::new(cfg, None, "Book");
            first.reply("", &[("book", json!({"nights":2}))]);
            first.prepare(json!({"nights":2}));
            let asked = first.turn().state.frames.values().next().unwrap().pending_calls[0]
                .parent
                .call_id
                .clone();
            let mut next = first.reconstruct(answer);
            let request = next.request().1;
            assert!(request.items.iter().any(|i| matches!(i,
                ChatItem::FunctionCallOutput(o) if o.call_id == asked
                    && o.output == approval::answered_text("book", answer))));
            let mut later = if delegate {
                next.reply("", &[("details", json!({"nights":1}))]);
                next.prepare(json!({"nights":1}));
                next.reply("Which date?", &[]);
                let mut later = next.reconstruct("Friday");
                later.accepted(json!({"reference":"details"}));
                later
            } else {
                next.reply("Noted.", &[]);
                next.reconstruct("Book the same room again")
            };
            let root_id = later.session.state.root_frame_id.clone();
            let root = &later.session.state.frames[&root_id];
            assert!(root.pending_calls.is_empty());
            assert!(root.history.iter().any(|i| matches!(i,
                ChatItem::FunctionCallOutput(o) if o.call_id == asked && o.output == answer)));
            let frames = later.session.state.frames.len();
            later.reply("", &[("book", json!({"nights":2}))]);
            later.prepare(json!({"nights":2}));
            assert_eq!(
                later.session.state.frames.len(),
                frames,
                "a later call needs a fresh approval"
            );
            let waiting = later.turn();
            assert!(matches!(waiting.result.outcome, RunOutcome::InputRequired { .. }));
            let root = &waiting.state.frames[&root_id];
            assert_eq!(root.pending_calls.len(), 1);
            assert_eq!(root.pending_calls[0].status, ChildCallStatus::Approval);
            assert_ne!(root.pending_calls[0].parent.call_id, asked);
        }
    }
}

#[test]
fn fresh_and_fork_use_validated_task_arguments_once_and_exclude_unresolved_calls() {
    for mode in [ChatHistory::Fresh, ChatHistory::Fork] {
        for nights in [2, 7] {
            let mut first = Driver::new(config(false, mode, true), None, "Preferences: quiet room");
            first.reply("", &[("book", json!({"nights": nights.to_string()}))]);
            first.prepare(json!({"nights":nights}));
            let task = first.request().1;
            assert_eq!(
                task.items.last().unwrap().as_message().unwrap().raw_text_content().unwrap(),
                format!("{{\"nights\":{nights}}}")
            );
            assert!(
                !task.items.iter().any(|i| matches!(
                    i,
                    ChatItem::FunctionCall(_) | ChatItem::FunctionCallOutput(_)
                ))
            );
            assert_eq!(
                task.items
                    .iter()
                    .filter(|i| i.as_message().is_some_and(
                        |m| m.raw_text_content().as_deref() == Some("Preferences: quiet room")
                    ))
                    .count(),
                usize::from(mode == ChatHistory::Fork)
            );
            first.reply("Which date?", &[]);
            let mut second = first.reconstruct("Friday");
            assert_eq!(
                second
                    .request()
                    .1
                    .items
                    .iter()
                    .filter(|i| i.as_message().is_some_and(|m| m.raw_text_content().as_deref()
                        == Some(&format!("{{\"nights\":{nights}}}"))))
                    .count(),
                1
            );
            second.accepted(json!({"reference":format!("{nights} nights on Friday")}));
            assert!(second.request().1.items.iter().any(|i| matches!(i, ChatItem::FunctionCallOutput(o) if o.output == format!("{{\"reference\":\"{nights} nights on Friday\"}}"))));
            second.reply("Done", &[]);
            let state = second.turn().state;
            let root = &state.frames[&state.root_frame_id];
            assert_eq!(
                root.history
                    .iter()
                    .filter(|i| i.as_message().is_some_and(
                        |m| m.raw_text_content().as_deref() == Some("Preferences: quiet room")
                    ))
                    .count(),
                1
            );
            assert!(root.history.iter().any(|i| {
                i.as_message().is_some_and(|m| m.raw_text_content().as_deref() == Some("Friday"))
            }));
            // The child question was generated, but has no delivery receipt: it stays private.
            assert!(!root.history.iter().any(|i| {
                i.as_message()
                    .is_some_and(|m| m.raw_text_content().as_deref() == Some("Which date?"))
            }));
        }
    }
}

#[test]
fn fork_preserves_complete_exchanges_and_merge_requires_delivery_receipts() {
    let call = FunctionCall {
        id: "lookup".into(),
        call_id: "lookup-call".into(),
        name: "lookup".into(),
        arguments: "{}".into(),
        created_at: std::time::UNIX_EPOCH,
        extra: Default::default(),
        group_id: None,
    };
    let output = lk_agents_abi::FunctionCallOutput::new(
        "lookup-result",
        "lookup-call",
        "quiet",
        false,
        std::time::UNIX_EPOCH,
    );
    let mut driver = Driver::new(config(false, ChatHistory::Fork, true), None, "Book");
    let root = driver.session.state.active_frame_id.clone();
    let history =
        vec![ChatItem::FunctionCall(call.clone()), ChatItem::FunctionCallOutput(output.clone())];
    let mut incomplete = call;
    incomplete.id = "unresolved".into();
    incomplete.call_id = "unresolved".into();
    let mut parent_history = history.clone();
    parent_history.push(ChatItem::FunctionCall(incomplete));
    assert_eq!(initial(&parent_history, ChatHistory::Fork), history);
    driver.reply("", &[("book", json!({"nights":2}))]);
    driver.prepare(json!({"nights":2}));
    let child_invocation = driver.invocation();
    let delivered = message("spoken", Role::Assistant, "We have a quiet room");
    driver.send(TextSessionEvent::DialogueDelivered {
        invocation_id: child_invocation.clone(),
        message: delivered.clone(),
    });
    driver.send(TextSessionEvent::DialogueDelivered {
        invocation_id: child_invocation,
        message: delivered.clone(),
    });
    driver.accepted(json!({"reference":"R"}));
    let request = driver.request().1;
    assert_eq!(request.items.iter().filter(|i| i.id() == "spoken").count(), 1);
    let call_index =
        request.items.iter().position(|i| matches!(i, ChatItem::FunctionCall(_))).unwrap();
    let output_index =
        request.items.iter().position(|i| matches!(i, ChatItem::FunctionCallOutput(_))).unwrap();
    let dialogue_index = request.items.iter().position(|i| i.id() == "spoken").unwrap();
    assert!(call_index < output_index && output_index < dialogue_index);
    assert_eq!(
        driver.session.state.frames[&root]
            .visible_dialogue
            .iter()
            .filter(|m| m.id == "spoken")
            .count(),
        1
    );
}

#[test]
fn unknown_agents_scopes_and_implicit_child_outputs_refuse_routing() {
    let mut cfg = config(false, ChatHistory::Fresh, false);
    cfg.agents.get_mut("booking").unwrap().output_type_explicit = false;
    assert!(matches!(TextSession::new(cfg, input(None)), Err(RoutingError::MissingOutputType(_))));
    let mut first = Driver::new(config(false, ChatHistory::Fresh, false), None, "Book");
    first.reply("", &[("book", json!({"nights":2}))]);
    first.prepare(json!({"nights":2}));
    first.reply("Which date?", &[]);
    let saved = first.turn().state;
    let mut missing = saved.clone();
    let active = missing.active_frame_id.clone();
    missing.frames.get_mut(&active).unwrap().agent_name = "removed".into();
    assert!(matches!(
        TextSession::new(first.config.clone(), input(Some(missing))),
        Err(RoutingError::UnknownAgent(_))
    ));
    let mut request = input(Some(saved));
    request.available_scopes.remove("draft");
    assert!(matches!(
        TextSession::new(first.config.clone(), request),
        Err(RoutingError::UnknownScope(_))
    ));
    let mut config = first.config.clone();
    config.agents.get_mut("desk").unwrap().agent_tools.get_mut("book").unwrap().context_scope =
        Some("missing".into());
    let mut refused = Driver::new(config, None, "Book");
    refused.reply("", &[("book", json!({"nights":2}))]);
    refused.prepare(json!({"nights":2}));
    assert_eq!(refused.session.state.frames.len(), 1);
    assert!(refused.effects.iter().any(|e| matches!(
        e,
        TextSessionEffect::RoutingRefused { error: RoutingError::UnknownScope(_), .. }
    )));
}

#[test]
fn direct_child_execution_reconciles_with_its_original_parent_call() {
    let mut first = Driver::new(config(false, ChatHistory::Fresh, false), None, "Book");
    first.reply("", &[("book", json!({"nights":2}))]);
    first.prepare(json!({"nights":2}));
    first.reply("Which date?", &[]);
    let turn = first.turn();
    let frame_id = turn.state.active_frame_id.clone();
    let mut session = TextSession::new(first.config.clone(), input(Some(turn.state))).unwrap();
    let result = RunResult {
        pending_call: None,
        input_requirements: Vec::new(),
        interrupted: false,
        outcome: RunOutcome::Completed { output: Output::Validated(ValidationId(99)) },
        all_messages: turn.result.all_messages,
        new_messages: Vec::new(),
        usage: Default::default(),
        verbatim: None,
    };
    let effects = session.handle(
        TextSessionEvent::Reconcile {
            frame_id: frame_id.clone(),
            result,
            output: Some(json!({"reference":"direct"})),
        },
        Ts::ZERO,
    );
    assert!(session.state.frames[&frame_id].completed);
    assert!(effects.iter().any(|e| matches!(e, TextSessionEffect::Run { effect:agent_run::AgentRunEffect::LlmRequestStart { chat_ctx, .. }, .. } if chat_ctx.items.iter().any(|i| matches!(i, ChatItem::FunctionCallOutput(o) if o.output == "{\"reference\":\"direct\"}")))));
    assert!(
        session
            .handle(
                TextSessionEvent::Reconcile {
                    frame_id,
                    result: RunResult {
                        pending_call: None,
                        input_requirements: Vec::new(),
                        interrupted: false,
                        outcome: RunOutcome::NoOutput,
                        all_messages: Vec::new(),
                        new_messages: Vec::new(),
                        usage: Default::default(),
                        verbatim: None
                    },
                    output: None
                },
                Ts::ZERO
            )
            .is_empty()
    );
}

#[test]
fn child_completion_waits_for_delivery_receipts_before_merging_and_starting_parent() {
    let mut driver = Driver::new(config(false, ChatHistory::Fresh, true), None, "Book");
    driver.reply("", &[("book", json!({"nights":2}))]);
    driver.prepare(json!({"nights":2}));
    let child_invocation = driver.invocation();
    let child_frame = driver.session.state.active_frame_id.clone();
    driver.auto_release = false;
    driver.accepted(json!({"reference":"R"}));
    // The generation/validation has ended, but the endpoint's already queued dialogue has
    // not reached the person yet. This late receipt must precede the parent model request.
    assert_eq!(driver.session.state.active_frame_id, child_frame);
    driver.send(TextSessionEvent::DialogueDelivered {
        invocation_id: child_invocation.clone(),
        message: message("late-delivery", Role::Assistant, "A quiet room is available"),
    });
    let delivered = Default::default();
    driver
        .send(TextSessionEvent::InvocationReleased { invocation_id: child_invocation, delivered });
    assert_eq!(driver.request().1.items.iter().filter(|i| i.id() == "late-delivery").count(), 1);
    assert_ne!(driver.session.state.active_frame_id, child_frame);
}

#[test]
fn sibling_children_follow_model_order_across_reconstruction_and_resolve_once() {
    let mut driver = Driver::new(config(false, ChatHistory::Fresh, false), None, "Book twice");
    driver.reply("", &[("book", json!({"nights":2})), ("book", json!({"nights":7}))]);
    let parent_invocation = driver.invocation();
    let calls: Vec<_> = driver
        .effects
        .iter()
        .filter_map(|e| match e {
            TextSessionEffect::Run {
                effect: agent_run::AgentRunEffect::ToolPrepare { call, arguments, .. },
                ..
            } => Some((*call, Value::Object(arguments.clone()))),
            _ => None,
        })
        .collect();
    for (call, arguments) in calls.iter().rev() {
        driver.run(agent_run::AgentRunEvent::ToolPrepared {
            call: *call,
            prepared: ResourceId(100 + call.0),
            identity: ArgumentIdentity::Json(arguments.clone()),
        });
        if *call == calls[1].0 {
            // A duplicated host receipt must release its resource and cannot replace the
            // arguments that the shared gate already admitted for this pending child.
            driver.run(agent_run::AgentRunEvent::ToolPrepared {
                call: *call,
                prepared: ResourceId(999),
                identity: ArgumentIdentity::Json(json!({"nights":99})),
            });
        }
    }
    assert!(driver.effects.iter().any(|e| matches!(
        e,
        TextSessionEffect::Run {
            effect: agent_run::AgentRunEffect::ToolArgumentsRelease { prepared: ResourceId(999) },
            ..
        }
    )));
    assert_eq!(
        driver
            .request()
            .1
            .items
            .last()
            .unwrap()
            .as_message()
            .unwrap()
            .raw_text_content()
            .as_deref(),
        Some("{\"nights\":2}")
    );
    driver.reply("Which date?", &[]);
    let waiting = driver.turn();
    let root = waiting.state.root_frame_id.clone();
    assert_eq!(waiting.state.frames[&root].pending_calls.len(), 2);
    let mut next = driver.reconstruct("Friday");
    next.accepted(json!({"reference":"first"}));
    assert_eq!(
        next.request().1.items.last().unwrap().as_message().unwrap().raw_text_content().as_deref(),
        Some("{\"nights\":7}")
    );
    next.accepted(json!({"reference":"second"}));
    let outputs: Vec<_> = next
        .request()
        .1
        .items
        .iter()
        .filter_map(|i| match i {
            ChatItem::FunctionCallOutput(o) => Some(o.output.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(outputs, ["{\"reference\":\"first\"}", "{\"reference\":\"second\"}"]);
    assert!(next.session.state.frames[&root].pending_calls.is_empty());
    assert!(!next.request().1.items.iter().any(|i| {
        i.as_message().is_some_and(|m| m.raw_text_content().as_deref() == Some("Friday"))
    }));
    let sequences: Vec<_> = next
        .effects
        .iter()
        .filter_map(|e| match e {
            TextSessionEffect::Emit { sequence, .. } => Some(*sequence),
            _ => None,
        })
        .collect();
    assert!(sequences.windows(2).all(|w| w[0] < w[1]));
    assert_eq!(next.effects.iter().filter(|e| matches!(e, TextSessionEffect::Emit { invocation_id, event:StreamEvent::ToolResult { status:ToolResultStatus::Completed, .. }, .. } if invocation_id == &parent_invocation)).count(), 2);
}

#[test]
fn invalid_parameters_and_changed_approval_arguments_never_start_a_child() {
    let mut first = Driver::new(config(true, ChatHistory::Fresh, false), None, "Book");
    first.reply("", &[("book", json!({"nights":2}))]);
    first.prepare(json!({"nights":2}));
    let mut next = first.reconstruct("yes");
    next.reply("", &[("book", json!({"nights":7}))]);
    next.prepare(json!({"nights":7}));
    assert_eq!(next.turn().state.frames.len(), 1);
    assert!(
        matches!(&next.turn().result.outcome, RunOutcome::InputRequired { question, .. } if question.contains("nights=7"))
    );
    let mut invalid = Driver::new(config(false, ChatHistory::Fresh, false), None, "Book");
    invalid.reply("", &[("book", json!({"nights":"invalid"}))]);
    let call = invalid
        .effects
        .iter()
        .find_map(|e| match e {
            TextSessionEffect::Run {
                effect: agent_run::AgentRunEffect::ToolPrepare { call, .. },
                ..
            } => Some(*call),
            _ => None,
        })
        .unwrap();
    invalid.run(agent_run::AgentRunEvent::ToolPrepareFailed {
        call,
        failure: crate::tool_calls::PrepareFailure::Refused("nights must be an integer".into()),
    });
    assert_eq!(invalid.session.state.frames.len(), 1);
    assert!(invalid.request().1.items.iter().any(|i| matches!(i, ChatItem::FunctionCallOutput(o) if o.is_error && o.output.contains("integer"))));
}

#[test]
fn cancellation_after_generation_settlement_still_returns_no_envelope() {
    let mut driver = Driver::new(config(false, ChatHistory::Fresh, false), None, "Hello");
    driver.auto_release = false;
    driver.reply("Hello", &[]);
    let invocation_id = driver.invocation();
    driver.send(TextSessionEvent::Cancel);
    assert!(driver.effects.iter().any(|e| matches!(
        e,
        TextSessionEffect::Finished { result: Err(TextSessionError::Run(RunError::Cancelled)), .. }
    )));
    assert!(
        !driver
            .effects
            .iter()
            .any(|e| matches!(e, TextSessionEffect::Finished { result: Ok(_), .. }))
    );
    let delivered = Default::default();
    assert!(
        driver.send(TextSessionEvent::InvocationReleased { invocation_id, delivered }).is_empty()
    );
}

#[test]
fn parking_waits_for_host_siblings_and_cancellation_settles_only_owned_child_work() {
    for cancel in [false, true] {
        let mut cfg = config(false, ChatHistory::Fresh, false);
        let mut lookup = cfg.agents["desk"].definition.tools[0].clone();
        lookup.spec.name = "lookup".into();
        cfg.agents.get_mut("desk").unwrap().definition.tools.push(lookup);
        let mut driver = Driver::new(cfg, None, "Book");
        let root = driver.session.state.active_frame_id.clone();
        driver.reply("", &[("book", json!({"nights":2})), ("lookup", json!({"nights":2}))]);
        let prepares: Vec<_> = driver
            .effects
            .iter()
            .filter_map(|e| match e {
                TextSessionEffect::Run {
                    effect: agent_run::AgentRunEffect::ToolPrepare { call, function_call, .. },
                    ..
                } => Some((*call, function_call.name.clone())),
                _ => None,
            })
            .collect();
        for (call, _) in &prepares {
            driver.run(agent_run::AgentRunEvent::ToolPrepared {
                call: *call,
                prepared: ResourceId(100 + call.0),
                identity: ArgumentIdentity::Json(json!({"nights":2})),
            });
        }
        assert_eq!(driver.session.state.active_frame_id, root);
        assert!(
            !driver.effects.iter().any(|e| matches!(e, TextSessionEffect::InvocationParked { .. }))
        );
        let lookup = prepares.iter().find(|(_, name)| name == "lookup").unwrap().0;
        if cancel {
            let cancellation = driver.send(TextSessionEvent::Cancel);
            assert_eq!(
                cancellation
                    .iter()
                    .filter(|e| matches!(
                        e,
                        TextSessionEffect::Run {
                            effect: agent_run::AgentRunEffect::ToolCancel { .. },
                            ..
                        }
                    ))
                    .count(),
                1
            );
            assert!(!cancellation.iter().any(|e| matches!(e, TextSessionEffect::Finished { .. })));
            driver.run(agent_run::AgentRunEvent::ToolSettled {
                call: lookup,
                outcome: lk_agents_abi::llm::tool_executor::ToolOutcome::Cancelled,
            });
            assert!(driver.effects.iter().any(|e| matches!(
                e,
                TextSessionEffect::Finished {
                    result: Err(TextSessionError::Run(RunError::Cancelled)),
                    ..
                }
            )));
            assert!(!driver.effects.iter().any(|e| matches!(
                e,
                TextSessionEffect::InvocationParked { .. }
                    | TextSessionEffect::Finished { result: Ok(_), .. }
            )));
        } else {
            driver.run(agent_run::AgentRunEvent::ToolSettled {
                call: lookup,
                outcome: lk_agents_abi::llm::tool_executor::ToolOutcome::Completed {
                    text: "available".into(),
                    verbatim: false,
                },
            });
            assert_ne!(driver.session.state.active_frame_id, root);
            driver.reply("Which date?", &[]);
            let state = driver.turn().state;
            assert!(
                state.frames[&root].history.iter().any(
                    |i| matches!(i, ChatItem::FunctionCallOutput(o) if o.output == "available")
                )
            );
        }
    }
}

#[test]
fn child_output_without_a_serialized_accepted_value_cannot_resolve_parent() {
    let mut driver = Driver::new(config(false, ChatHistory::Fresh, false), None, "Book");
    driver.reply("", &[("book", json!({"nights":2}))]);
    driver.prepare(json!({"nights":2}));
    driver.reply("", &[("final_result", json!({"reference":"R"}))]);
    let validation = driver
        .effects
        .iter()
        .rev()
        .find_map(|e| match e {
            TextSessionEffect::Run {
                effect: agent_run::AgentRunEffect::ValidateOutput { validation, .. },
                ..
            } => Some(*validation),
            _ => None,
        })
        .unwrap();
    driver.send(TextSessionEvent::OutputValidated {
        invocation_id: driver.invocation(),
        validation,
        verdict: OutputVerdict::Accepted { text: None },
        output: None,
    });
    assert!(driver.effects.iter().any(|e| matches!(
        e,
        TextSessionEffect::Finished {
            result: Err(TextSessionError::Routing(RoutingError::MissingAcceptedOutput)),
            ..
        }
    )));
    assert!(!driver.effects.iter().any(|e| {
        matches!(e, TextSessionEffect::Finished { result: Ok(_), .. })
            || matches!(e, TextSessionEffect::Emit {
                event: StreamEvent::ToolResult { tool_name, status: ToolResultStatus::Completed, .. },
                ..
            } if tool_name == "book")
    }));
}

#[test]
fn a_failed_child_keeps_its_invocation_identity_on_the_terminal_effect() {
    let mut driver = Driver::new(config(false, ChatHistory::Fresh, false), None, "Book");
    let parent = driver.invocation();
    driver.reply("", &[("book", json!({"nights":2}))]);
    driver.prepare(json!({"nights":2}));
    let child = driver.invocation();
    assert_ne!(parent, child);
    let (request, _) = driver.request();
    let effects = driver.run(agent_run::AgentRunEvent::LlmFailed {
        request,
        error: "child provider failed".into(),
    });
    assert!(effects.iter().any(|effect| matches!(
        effect,
        TextSessionEffect::Finished {
            invocation_id: Some(invocation_id),
            result: Err(TextSessionError::Run(RunError::Model(_))),
            ..
        } if *invocation_id == child
    )));
    assert!(
        driver.session.invocation.is_none(),
        "the terminal retains an identity, not a live run"
    );
}

#[test]
fn late_prepared_arguments_are_released_even_after_the_session_settles() {
    let mut driver = Driver::new(config(false, ChatHistory::Fresh, false), None, "Hello");
    let invocation_id = driver.invocation();
    driver.reply("Hello", &[]);
    let effects = driver.send(TextSessionEvent::Run {
        invocation_id,
        event: agent_run::AgentRunEvent::ToolPrepared {
            call: CallId(999),
            prepared: ResourceId(999),
            identity: ArgumentIdentity::Json(json!({})),
        },
    });
    assert!(effects.iter().any(|e| matches!(
        e,
        TextSessionEffect::Run {
            effect: agent_run::AgentRunEffect::ToolArgumentsRelease { prepared: ResourceId(999) },
            ..
        }
    )));
    assert!(!effects.iter().any(|e| matches!(e, TextSessionEffect::Finished { .. })));
}
