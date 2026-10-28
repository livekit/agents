//! Model steps, tools, outputs and questions.

use super::*;

// Catches: a tool result not reaching the next request, the incoming message missing from
// `new_messages`, instructions leaking into `all_messages`, or usage counted per run instead of
// per request.
#[test]
fn a_tool_runs_and_the_model_continues_from_its_output() {
    let mut d =
        Driver::new(text_agent(vec![string_tool("find_slot", "day")]), "Book me Monday", vec![]);
    let (_, ctx, tools) = d.request();
    assert_eq!(
        transcript(&ctx.items),
        pairs(&[("system", "Book appointments."), ("user", "Book me Monday")])
    );
    assert_eq!(tools.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(), ["find_slot"]);

    d.reply("Let me look.", &[("find_slot", json!({"day": "Monday"}))]);
    d.run_tools(|_| value("9:00 is free"));
    let (_, ctx, _) = d.request();
    assert_eq!(
        transcript(&ctx.items)[2..],
        pairs(&[
            ("assistant", "Let me look."),
            ("call", r#"find_slot({"day":"Monday"})"#),
            ("output", "9:00 is free"),
        ])
    );

    d.reply("Monday 9:00 is free.", &[]);
    let result = d.result();
    assert_eq!(
        result.outcome,
        RunOutcome::Completed { output: Output::Text("Monday 9:00 is free.".into()) }
    );
    assert_eq!(transcript(&result.new_messages)[0], ("user".into(), "Book me Monday".into()));
    assert_eq!(result.new_messages.len(), 5);
    assert_eq!(result.all_messages, result.new_messages);
    assert_eq!(
        result.usage,
        RunUsage { requests: 2, input_tokens: 20, output_tokens: 6, tool_calls: 1 }
    );
}

// Catches: text withheld until the response ends (no incremental output), or a step's text
// stream not closed before its tools run.
#[test]
fn text_streams_as_it_arrives_before_the_model_ends() {
    let mut d = Driver::new(text_agent(vec![]), "hi", vec![]);
    let (request, _, _) = d.request();
    let fx = d.send(Event::LlmChunk { request, chunk: chunk(Some("Hel"), vec![], None) });
    assert!(
        matches!(&fx[..], [Effect::Emit(StreamEvent::TextDelta { delta, .. })] if delta == "Hel")
    );
    assert!(d.finished().is_none());
    d.send(Event::LlmChunk { request, chunk: chunk(Some("lo"), vec![], None) });
    let fx = d.send(Event::LlmEnded { request });
    assert!(matches!(&fx[0], Effect::Emit(StreamEvent::TextEnd { interrupted: false, .. })));
    assert_eq!(d.result().outcome, RunOutcome::Completed { output: Output::Text("Hello".into()) });
}

// Catches: identical inputs producing different ids, effects or history (ambient time, entropy
// or iteration order in a decision).
#[test]
fn identical_inputs_replay_identically() {
    let play = || {
        let mut d = Driver::new(
            text_agent(vec![string_tool("a", "x"), string_tool("b", "y")]),
            "go",
            vec![],
        );
        d.reply("", &[("b", json!({"y": "2"})), ("a", json!({"x": "1"}))]);
        d.run_tools(value);
        d.reply("done", &[]);
        d.effects
    };
    assert_eq!(play(), play());
}

// Catches (c7 defect 2): a direct run ignoring `ToolResult(reply_required=False)` and asking the
// model again.
#[test]
fn a_tool_that_asks_for_no_reply_ends_the_turn() {
    let mut d =
        Driver::new(text_agent(vec![string_tool("send_dtmf", "digits")]), "press 1", vec![]);
    d.reply("", &[("send_dtmf", json!({"digits": "1"}))]);
    d.run_tools(|_| ToolOutcome::Value {
        text: "sent".into(),
        reply_required: false,
        verbatim: false,
        more: None,
    });
    assert_eq!(d.requests(), 1);
    assert_eq!(d.result().outcome, RunOutcome::Completed { output: Output::Nothing });
}

// Catches: `StopResponse` continuing the loop, or a verbatim result losing its delivered text or
// its exact words on the result.
#[test]
fn stop_response_and_verbatim_results_end_or_lead_the_turn() {
    let mut d = Driver::new(text_agent(vec![string_tool("stop", "x")]), "stop", vec![]);
    d.reply("", &[("stop", json!({"x": "1"}))]);
    d.run_tools(|_| ToolOutcome::StopResponse);
    assert_eq!(
        (d.requests(), &d.result().outcome),
        (1, &RunOutcome::Completed { output: Output::Nothing })
    );

    let mut d = Driver::new(text_agent(vec![string_tool("code", "x")]), "my code?", vec![]);
    d.reply("", &[("code", json!({"x": "1"}))]);
    d.run_tools(|_| ToolOutcome::Value {
        text: "Your code is 1234.".into(),
        reply_required: true,
        verbatim: true,
        more: None,
    });
    let (_, ctx, _) = d.request();
    assert_eq!(
        ctx.items.last().map(|i| transcript(std::slice::from_ref(i))[0].1.clone()).unwrap(),
        "delivered to the person as written: Your code is 1234."
    );
    d.reply("Anything else?", &[]);
    let result = d.result();
    assert_eq!(
        result.outcome,
        RunOutcome::Completed {
            output: Output::Text("Your code is 1234.\n\nAnything else?".into())
        }
    );
    assert_eq!(result.verbatim.as_deref(), Some("Your code is 1234."));
}

// Catches (f2, c7 #8): `ModelRetry` treated as an internal error, or retried without a budget.
#[test]
fn model_retry_is_feedback_within_the_tool_retry_budget() {
    let mut d = Driver::new(text_agent(vec![string_tool("book", "day")]), "book", vec![]);
    d.reply("", &[("book", json!({"day": "someday"}))]);
    d.run_tools(|_| ToolOutcome::ModelRetry("use a weekday".into()));
    let (_, ctx, _) = d.request();
    let ChatItem::FunctionCallOutput(output) = ctx.items.last().unwrap() else { panic!() };
    assert_eq!((output.output.as_str(), output.is_error), ("use a weekday", true));

    d.reply("", &[("book", json!({"day": "never"}))]);
    d.run_tools(|_| ToolOutcome::ModelRetry("use a weekday".into()));
    assert_eq!(
        d.finished(),
        Some(&Err(RunError::ToolRetriesExceeded { tool: "book".into(), retries: 1 }))
    );
}

// Catches: invalid arguments running the body, or an unknown tool failing the run instead of
// telling the model.
#[test]
fn invalid_arguments_and_unknown_tools_are_feedback_and_never_run_a_body() {
    let mut d = Driver::new(text_agent(vec![string_tool("book", "day")]), "book", vec![]);
    d.reply("", &[("book", json!({"day": 5})), ("cancel", json!({}))]);
    let (call, _) = d.prepares()[0].clone();
    d.send(Event::ToolPrepareFailed { call, error: "day: Input should be a valid string".into() });
    assert!(d.executes().is_empty());
    let (_, ctx, _) = d.request();
    let outputs: Vec<_> =
        transcript(&ctx.items).into_iter().filter(|(k, _)| k == "output").collect();
    assert_eq!(
        outputs,
        pairs(&[
            ("output", "day: Input should be a valid string"),
            ("output", "unknown tool `cancel`")
        ])
    );
}

// Catches: a non-object output schema offered unwrapped, its value not unwrapped, an accepted
// candidate not settling the run, or a rejected one not fed back.
#[test]
fn structured_output_is_accepted_through_final_result() {
    let mut d = Driver::new(
        agent(vec![], OutputSpec::Structured { schema: json!({"type": "integer"}) }),
        "count",
        vec![],
    );
    let (_, _, tools) = d.request();
    assert_eq!(tools[0].name, "final_result");
    assert_eq!(tools[0].json_schema["properties"]["result"], json!({"type": "integer"}));

    d.reply("", &[("final_result", json!({"result": 3}))]);
    let validation = d.effects.iter().find_map(|e| match e {
        Effect::OutputValidate { validation, candidate } => {
            assert_eq!(candidate, &OutputCandidate::Value(json!(3)));
            Some(*validation)
        }
        _ => None,
    });
    d.send(Event::OutputValidated {
        validation: validation.unwrap(),
        verdict: OutputVerdict::Retry("must be even".into()),
    });
    let (_, ctx, _) = d.request();
    assert_eq!(transcript(&ctx.items).last().unwrap().1, "invalid final_result: must be even");

    d.reply("", &[("final_result", json!({"result": 4}))]);
    let validation = d.effects.iter().rev().find_map(|e| match e {
        Effect::OutputValidate { validation, .. } => Some(*validation),
        _ => None,
    });
    d.send(Event::OutputValidated {
        validation: validation.unwrap(),
        verdict: OutputVerdict::Accepted { text: None },
    });
    assert_eq!(
        d.result().outcome,
        RunOutcome::Completed { output: Output::Validated(validation.unwrap()) }
    );
}

// Catches (c7 defect 4): an output validator's replacement value discarded.
#[test]
fn a_validator_replacement_is_the_accepted_output() {
    let mut d = Driver::new(agent(vec![], OutputSpec::Text { validated: true }), "hi", vec![]);
    d.reply("hello", &[]);
    let validation = d.effects.iter().find_map(|e| match e {
        Effect::OutputValidate { validation, .. } => Some(*validation),
        _ => None,
    });
    d.send(Event::OutputValidated {
        validation: validation.unwrap(),
        verdict: OutputVerdict::Accepted { text: Some("Hello!".into()) },
    });
    assert_eq!(d.result().outcome, RunOutcome::Completed { output: Output::Text("Hello!".into()) });
}

// Catches: conversational text where a typed output is expected settling as output, or a
// question being worded twice.
#[test]
fn text_where_a_typed_output_is_expected_asks_the_person() {
    let mut d = Driver::new(
        agent(vec![], OutputSpec::Structured { schema: json!({"type": "object"}) }),
        "book",
        vec![],
    );
    d.reply("Which day works for you?", &[]);
    assert_eq!(d.requests(), 1);
    assert_eq!(
        d.result().outcome,
        RunOutcome::InputRequired {
            question: "Which day works for you?".into(),
            verbatim: false,
            answer_tool: None
        }
    );
}

// Catches: the facing agent passing a tool's question on unworded, a delegated run wording it,
// or the worded question missing from the history.
#[test]
fn a_tool_question_is_worded_by_the_facing_agent_only() {
    let ask = |_: &str| ToolOutcome::InputRequired {
        question: "photo of the damage".into(),
        verbatim: false,
        tool: None,
    };
    let mut d = Driver::new(text_agent(vec![string_tool("photo", "x")]), "claim", vec![]);
    d.reply("", &[("photo", json!({"x": "1"}))]);
    d.run_tools(ask);
    let (_, ctx, tools) = d.request();
    assert!(tools.is_empty());
    assert!(
        transcript(&ctx.items).last().unwrap().1.ends_with("nothing else: photo of the damage")
    );
    d.reply("Could you send a photo of the damage?", &[]);
    let result = d.result();
    assert_eq!(
        result.outcome,
        RunOutcome::InputRequired {
            question: "Could you send a photo of the damage?".into(),
            verbatim: false,
            answer_tool: None
        }
    );
    assert_eq!(
        transcript(&result.all_messages).last().unwrap().1,
        "Could you send a photo of the damage?"
    );

    let mut input = Driver::new(text_agent(vec![]), "x", vec![]).run.input.clone();
    input.delegated = true;
    input.message = user("m", "claim");
    let mut d = Driver::with_input(text_agent(vec![string_tool("photo", "x")]), input);
    d.reply("", &[("photo", json!({"x": "1"}))]);
    d.run_tools(ask);
    assert_eq!(d.requests(), 1);
    assert!(
        matches!(&d.result().outcome, RunOutcome::InputRequired { question, .. } if question == "photo of the damage")
    );
}

// Catches: `InputRequired(tool=)`'s answer going to the model instead of the named tool, or the
// waiting placeholder left in the history once answered.
#[test]
fn an_answer_for_a_named_tool_is_handed_to_it_without_a_model_turn() {
    let tools =
        || vec![string_tool("ask_expert", "question"), string_tool("reply_expert", "answer")];
    let mut d = Driver::new(text_agent(tools()), "is it covered?", vec![]);
    d.reply("", &[("ask_expert", json!({"question": "covered?"}))]);
    d.run_tools(|_| ToolOutcome::InputRequired {
        question: "Which policy?".into(),
        verbatim: true,
        tool: Some("reply_expert".into()),
    });
    let first = d.result().clone();
    let RunOutcome::InputRequired { answer_tool, .. } = &first.outcome else { panic!() };
    assert_eq!(answer_tool.as_deref(), Some("reply_expert"));

    let mut input = d.run.input.clone();
    input.history = first.all_messages.clone();
    input.message = user("msg_2", "Gold plan");
    input.answer_tool = answer_tool.clone();
    let mut d = Driver::with_input(text_agent(tools()), input);
    assert_eq!(d.requests(), 0);
    assert_eq!(d.prepares()[0].1, Map::from_iter([("answer".to_owned(), json!("Gold plan"))]));
    d.run_tools(|_| value("covered"));
    let (_, ctx, _) = d.request();
    let outputs: Vec<_> =
        transcript(&ctx.items).into_iter().filter(|(k, _)| k == "output").collect();
    assert_eq!(outputs, pairs(&[("output", "Gold plan"), ("output", "covered")]));
}

// Catches (decision 47): an implicit loop cap ending a long, finite model/tool sequence.
#[test]
fn a_long_finite_tool_sequence_completes_without_a_loop_cap() {
    let mut d = Driver::new(text_agent(vec![string_tool("step", "n")]), "go", vec![]);
    for n in 0..40 {
        d.reply("", &[("step", json!({"n": n.to_string()}))]);
        d.run_tools(|_| value("ok"));
    }
    d.reply("done", &[]);
    assert_eq!(d.result().usage.requests, 41);
}

// Catches: dynamic instructions not rendered before each request, or rendered ones entering the
// history.
#[test]
fn dynamic_instructions_are_rendered_before_every_request() {
    let mut a = text_agent(vec![string_tool("t", "x")]);
    a.dynamic_instructions = true;
    let mut d = Driver::new(a, "hi", vec![]);
    for (n, rendered) in ["Patient: none.", "Patient: Ada."].into_iter().enumerate() {
        let render = d.effects.iter().rev().find_map(|e| match e {
            Effect::InstructionsRender { render } => Some(*render),
            _ => None,
        });
        assert_eq!(d.requests(), n);
        d.send(Event::InstructionsRendered {
            render: render.unwrap(),
            result: Ok(rendered.into()),
        });
        let (_, ctx, _) = d.request();
        assert_eq!(transcript(&ctx.items)[0].1, format!("Book appointments.\n\n{rendered}"));
        if n == 0 {
            d.reply("", &[("t", json!({"x": "1"}))]);
            d.run_tools(|_| value("ok"));
        }
    }
    d.reply("hello", &[]);
    assert!(d.result().all_messages.iter().all(|i| !i.is_instruction()));
}
