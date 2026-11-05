//! Concurrent sibling calls, duplicate admission, cancellation and the cleanup deadline.

use super::*;

fn ask(question: &str) -> ToolOutcome {
    ToolOutcome::InputRequired { question: question.into(), verbatim: true, tool: None }
}

/// Prepares every call of the step with its arguments as sent.
fn prepare_all(d: &mut Driver) {
    for (call, arguments) in d.prepares() {
        d.prepared(call, Value::Object(arguments));
    }
}

// Catches: a later sibling's progress or completion
// recognised only after an earlier stalled sibling ends, or history assembled in arrival order.
#[test]
fn siblings_are_recognised_on_arrival_and_recorded_in_model_order() {
    let mut d = Driver::new(
        text_agent(vec![string_tool("slow", "x"), string_tool("fast", "x")]),
        "go",
        vec![],
    );
    d.reply("", &[("slow", json!({"x": "1"})), ("fast", json!({"x": "2"}))]);
    prepare_all(&mut d);
    let calls: Vec<CallId> = d.executes().iter().map(|(c, ..)| *c).collect();
    let fx = d.send(AgentRunEvent::ToolProgress { call: calls[1], message: "halfway".into() });
    assert!(
        matches!(&fx[..], [AgentRunEffect::Emit(StreamEvent::Progress { tool_name, .. })] if tool_name == "fast")
    );
    let fx = d.send(AgentRunEvent::ToolSettled { call: calls[1], outcome: value("fast done") });
    assert!(
        matches!(&fx[..], [AgentRunEffect::Emit(StreamEvent::ToolResult { tool_name, output, .. })]
        if tool_name == "fast" && output == "fast done")
    );
    assert_eq!(d.requests(), 1, "the step waits for its stalled sibling");
    d.send(AgentRunEvent::ToolSettled { call: calls[0], outcome: value("slow done") });
    let (_, ctx, _) = d.request();
    assert_eq!(
        transcript(&ctx.items)[2..],
        pairs(&[
            ("call", r#"slow({"x":"1"})"#),
            ("call", r#"fast({"x":"2"})"#),
            ("output", "slow done"),
            ("output", "fast done"),
        ])
    );
}

// Catches: duplicate policies carried but not enforced: `reject` running the
// second body, `replace` leaving the original running, `confirm` running before the model
// re-asserts, `name_and_args` treating different arguments as duplicates.
#[test]
fn duplicate_calls_follow_the_tool_policy() {
    let run = |mode: DuplicateMode, scope: DuplicateScope, second: Value| {
        let mut t = string_tool("lookup", "id");
        t.on_duplicate = mode;
        t.duplicate_scope = scope;
        let mut d = Driver::new(text_agent(vec![t]), "go", vec![]);
        d.reply("", &[("lookup", json!({"id": "1"})), ("lookup", second)]);
        let calls = d.prepares();
        d.prepared(calls[0].0, json!({"id": "1"}));
        let mut arguments = calls[1].1.clone();
        arguments.remove(CONFIRM_DUPLICATE_PARAM);
        d.prepared(calls[1].0, Value::Object(arguments));
        let executed: Vec<CallId> = d.executes().iter().map(|(c, ..)| *c).collect();
        let cancelled: Vec<CallId> = d
            .effects
            .iter()
            .filter_map(|e| match e {
                AgentRunEffect::ToolCancel { call } => Some(*call),
                _ => None,
            })
            .collect();
        (calls.iter().map(|c| c.0).collect::<Vec<_>>(), executed, cancelled, d)
    };
    use lk_agents_abi::llm::provider_format::tools::CONFIRM_DUPLICATE_PARAM;
    let same = json!({"id": "1"});

    let (calls, executed, cancelled, _) =
        run(DuplicateMode::Allow, DuplicateScope::Name, same.clone());
    assert_eq!((executed, cancelled), (calls.clone(), vec![]));

    let (calls, executed, _, mut d) =
        run(DuplicateMode::Reject, DuplicateScope::Name, same.clone());
    assert_eq!(executed, [calls[0]]);
    d.send(AgentRunEvent::ToolSettled { call: calls[0], outcome: value("found") });
    d.send(AgentRunEvent::ToolHandlingTemplateRendered { call: calls[1], text: None });
    let (_, ctx, _) = d.request();
    assert!(
        transcript(&ctx.items)
            .last()
            .unwrap()
            .1
            .starts_with("Same tool `lookup` is already running")
    );

    // Catches: the agent's or toolset's `duplicate_reject_template` ignored.
    let (calls, _, _, mut d) = run(DuplicateMode::Reject, DuplicateScope::Name, same.clone());
    let template = d.effects.iter().find_map(|e| match e {
        AgentRunEffect::RenderToolHandlingTemplate { call, template } if *call == calls[1] => {
            Some(template.clone())
        }
        _ => None,
    });
    assert!(matches!(
        template,
        Some(ToolTemplate::DuplicateReject { function_name, fnc_calls_json, .. })
            if function_name == "lookup" && fnc_calls_json.len() == 1
    ));
    d.send(AgentRunEvent::ToolSettled { call: calls[0], outcome: value("found") });
    d.send(AgentRunEvent::ToolHandlingTemplateRendered {
        call: calls[1],
        text: Some("Still looking.".into()),
    });
    let (_, ctx, _) = d.request();
    assert_eq!(transcript(&ctx.items).last().unwrap().1, "Still looking.");

    let (calls, executed, cancelled, mut d) =
        run(DuplicateMode::Replace, DuplicateScope::Name, same.clone());
    assert_eq!((executed, cancelled), (calls.clone(), vec![calls[0]]));
    d.send(AgentRunEvent::ToolSettled { call: calls[0], outcome: ToolOutcome::Cancelled });
    d.send(AgentRunEvent::ToolSettled { call: calls[1], outcome: value("found") });
    assert_eq!(d.requests(), 2, "the replaced original settles before the step continues");

    let (calls, executed, ..) = run(DuplicateMode::Confirm, DuplicateScope::Name, same.clone());
    assert_eq!(executed, [calls[0]]);
    let confirmed = json!({"id": "1", CONFIRM_DUPLICATE_PARAM: true});
    let (calls, executed, ..) = run(DuplicateMode::Confirm, DuplicateScope::Name, confirmed);
    assert_eq!(executed, calls);

    let (calls, executed, ..) =
        run(DuplicateMode::Reject, DuplicateScope::NameAndArgs, json!({"id": "2"}));
    assert_eq!(executed, calls);
}

// Catches: a cancelled run settling before the work it started reported back, a late result
// adopted after settlement, or prepared arguments leaked by a cancellation.
#[test]
fn cancel_settles_once_started_work_reports_back() {
    let mut d =
        Driver::new(text_agent(vec![string_tool("a", "x"), string_tool("b", "x")]), "go", vec![]);
    d.reply("", &[("a", json!({"x": "1"})), ("b", json!({"x": "2"}))]);
    let calls = d.prepares();
    d.prepared(calls[0].0, json!({"x": "1"}));
    let fx = d.send(AgentRunEvent::Cancel);
    assert_eq!(
        fx,
        [
            AgentRunEffect::ToolCancel { call: calls[0].0 },
            AgentRunEffect::ToolCancel { call: calls[1].0 }
        ]
    );
    assert!(d.finished().is_none());
    d.send(AgentRunEvent::ToolSettled { call: calls[0].0, outcome: ToolOutcome::Cancelled });
    let fx = d.send(AgentRunEvent::ToolPrepared {
        call: calls[1].0,
        prepared: ResourceId(99),
        identity: ArgumentIdentity::Json(json!({})),
    });
    assert_eq!(fx[0], AgentRunEffect::ToolArgumentsRelease { prepared: ResourceId(99) });
    assert_eq!(
        fx[1],
        AgentRunEffect::Finished {
            result: Box::new(Err(RunError::Cancelled)),
            input_requirements: Vec::new()
        }
    );
    assert!(d.executes().len() == 1 && d.run.wake_at().is_none());
    assert!(
        d.send(AgentRunEvent::ToolSettled { call: calls[0].0, outcome: value("late") }).is_empty()
    );
}

// Catches: the settled history or result depending on the order the host's answers arrive in
// (a decision keyed on arrival rather than model order), or a run settling twice.
#[test]
fn every_arrival_order_settles_the_same_history_once() {
    let tools = || vec![string_tool("a", "x"), string_tool("b", "x"), string_tool("c", "x")];
    let mut outcomes = Vec::new();
    // A small deterministic generator: each seed is one interleaving of preparations, progress
    // and settlements across the three siblings.
    for seed in 0u64..64 {
        let mut state = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let mut next = |n: usize| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as usize % n
        };
        let mut d = Driver::new(text_agent(tools()), "go", vec![]);
        d.reply(
            "",
            &[("a", json!({"x": "1"})), ("b", json!({"x": "2"})), ("c", json!({"x": "3"}))],
        );
        let mut pending: Vec<(CallId, u8)> = d.prepares().iter().map(|(c, _)| (*c, 0)).collect();
        while !pending.is_empty() {
            let i = next(pending.len());
            let (call, stage) = pending[i];
            match stage {
                0 => {
                    d.prepared(call, json!({}));
                }
                1 => {
                    d.send(AgentRunEvent::ToolProgress { call, message: "working".into() });
                }
                _ => {
                    let outcome = if call.0 % 2 == 0 {
                        ask("Which?")
                    } else {
                        value(&format!("r{}", call.0))
                    };
                    d.send(AgentRunEvent::ToolSettled { call, outcome });
                }
            }
            if stage == 2 {
                pending.remove(i);
            } else {
                pending[i].1 += 1;
            }
        }
        let finished =
            d.effects.iter().filter(|e| matches!(e, AgentRunEffect::Finished { .. })).count();
        assert_eq!(finished, 1, "seed {seed}");
        outcomes.push(d.result().clone());
    }
    assert!(outcomes.windows(2).all(|w| w[0] == w[1]));
}

// Catches: a cancelled request that never reports back holding the run forever, a wake before the
// deadline changing anything, a receipt processed at the deadline beating incomplete cleanup, or a
// late receipt settling the run twice.
#[test]
fn the_cleanup_deadline_decides_between_a_late_receipt_and_incomplete_cleanup() {
    for millis in [4_999, 5_000, 5_001] {
        let mut d = Driver::new(text_agent(Vec::new()), "go", Vec::new());
        let (request, _, _) = d.request();
        d.send(AgentRunEvent::LlmChunk { request, chunk: chunk(Some("Partial"), vec![], None) });
        let fx = d.send(AgentRunEvent::Cancel);
        assert!(matches!(
            &fx[0],
            AgentRunEffect::Emit(StreamEvent::TextEnd { interrupted: true, .. })
        ));
        assert_eq!(fx[1], AgentRunEffect::LlmRequestCancel { request });
        assert_eq!(d.run.wake_at(), Some(Ts::ZERO.saturating_add(Duration::from_secs(5))));
        d.advance(Duration::from_millis(millis));
        let woke = d.send(AgentRunEvent::Wake);
        let receipt = d.send(AgentRunEvent::LlmEnded { request });
        let cancelled = AgentRunEffect::Finished {
            result: Box::new(Err(RunError::Cancelled)),
            input_requirements: Vec::new(),
        };
        if millis < 5_000 {
            assert!(woke.is_empty(), "an early wake changes nothing");
            assert_eq!(receipt, [cancelled]);
        } else {
            let pending = vec![format!("llm request {}", request.0)];
            assert_eq!(woke, [AgentRunEffect::IncompleteCleanup { pending }, cancelled]);
            assert!(receipt.is_empty(), "a receipt processed after settlement is inert");
        }
        assert!(d.run.wake_at().is_none());
        assert!(d.send(AgentRunEvent::Wake).is_empty(), "settlement happens exactly once");
    }
}

/// The first duplicate body finishes while the host is still rendering the second's refusal.
fn waiting_for_refusal_template() -> (Driver, CallId) {
    let mut lookup = string_tool("lookup", "x");
    lookup.on_duplicate = DuplicateMode::Reject;
    let mut driver = Driver::new(text_agent(vec![lookup]), "go", Vec::new());
    driver.reply("", &[("lookup", json!({"x": "1"})), ("lookup", json!({"x": "2"}))]);
    let calls = driver.prepares();
    driver.prepared(calls[0].0, json!({"x": "1"}));
    let effects = driver.prepared(calls[1].0, json!({"x": "2"}));
    assert!(effects.iter().any(|effect| matches!(
        effect,
        AgentRunEffect::RenderToolHandlingTemplate { call, .. } if *call == calls[1].0
    )));
    driver.send(AgentRunEvent::ToolSettled { call: calls[0].0, outcome: value("found") });
    (driver, calls[1].0)
}

#[test]
fn cancellation_waits_for_the_refusal_renderer_to_report_its_end() {
    let (mut driver, call) = waiting_for_refusal_template();
    let effects = driver.send(AgentRunEvent::Cancel);
    assert_eq!(effects, [AgentRunEffect::ToolCancel { call }]);
    assert!(driver.finished().is_none(), "the host renderer still owns an invocation");
    assert_eq!(driver.run.wake_at(), Some(Ts::ZERO.saturating_add(Duration::from_secs(5))));

    driver.send(AgentRunEvent::ToolHandlingTemplateRendered { call, text: None });
    assert!(matches!(driver.finished(), Some(Err(RunError::Cancelled))));
    assert!(driver.run.wake_at().is_none());
    assert!(
        driver.send(AgentRunEvent::ToolHandlingTemplateRendered { call, text: None }).is_empty(),
        "a duplicate receipt cannot settle the run again"
    );
}

#[test]
fn a_refusal_renderer_missing_its_cleanup_deadline_is_reported_incomplete() {
    let (mut driver, call) = waiting_for_refusal_template();
    driver.send(AgentRunEvent::Cancel);
    driver.advance(Duration::from_secs(5));
    let effects = driver.send(AgentRunEvent::Wake);
    assert!(matches!(
        effects.as_slice(),
        [AgentRunEffect::IncompleteCleanup { pending }, AgentRunEffect::Finished { .. }]
            if pending == &vec![format!("tool call {}", call.0)]
    ));
    assert!(matches!(driver.finished(), Some(Err(RunError::Cancelled))));
    assert!(
        driver.send(AgentRunEvent::ToolHandlingTemplateRendered { call, text: None }).is_empty()
    );
}

#[test]
fn a_refusal_renderer_settles_normally_before_the_next_model_step() {
    let (mut driver, call) = waiting_for_refusal_template();
    assert!(driver.finished().is_none());
    driver.send(AgentRunEvent::ToolHandlingTemplateRendered {
        call,
        text: Some("This lookup is already running.".into()),
    });
    driver.reply("Found it.", &[]);
    assert_eq!(
        driver.result().outcome,
        RunOutcome::Completed { output: Output::Text("Found it.".into()) }
    );
    assert!(driver.run.wake_at().is_none());
}
