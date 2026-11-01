//! Realtime tools regression tests.

mod realtime_cut_reply_tool_runs {
    //! A realtime reply whose audio the output cut still has its tools running.
    use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer};
    use lk_agents_voice::trace::{SpeechStep, Step};

    // Catches: the realtime step deciding `tools_running` from its playout (`!heard.complete`), so a
    // reply whose audio the speaker cut (a republished track, a sink reset) reported its tools as
    // not running while they ran. `wait_for_idle` then counted the working speech as busy, a tool
    // inside `ctx.foreground()` waited on its own speech forever, and the reply never ended: the
    // agent idled silent for the rest of the call (survey seed 4224, once the fake realtime server
    // called tools). The pipeline step decides it from the speech's interruption, as the reference
    // does; the realtime one now does too.
    #[test]
    fn a_cut_realtime_reply_still_runs_its_tools() {
        // A model whose replies speak and call `lookup`.
        let setup = Setup {
            realtime: Some(RealtimeSetup {
                turn_detection: false,
                tool_calls: Some(100),
                ..RealtimeSetup::default()
            }),
            tools: vec!["lookup".into()],
            ..Setup::default()
        };
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            run.apply(&Input::GenerateReply { user_input: None, instructions: None, allow: None })
                .await;
            // The regression needs a tool already running when its audio is cut. Wait for both
            // external resources while the healthy speaker reports real PCM progress; fixed delays
            // can reset the reply before this server has even delivered its tool call.
            let mut running_tool = None;
            for _ in 0..200 {
                crate::playout::apply_with_playout(&mut run, &Input::Advance { ms: Millis(100) })
                    .await;
                let tool =
                    run.record.live().of(lk_agents_fuzz::record::Kind::Tool).first().copied();
                if tool.is_some()
                    && !run.record.live().of(lk_agents_fuzz::record::Kind::Segment).is_empty()
                {
                    running_tool = tool;
                    break;
                }
            }
            let tool = running_tool.expect("the server's tool must overlap its admitted audio");
            // The speaker cuts its playback, while the speech and its tool remain live.
            run.apply(&Input::SinkReset { output: Ref::new(0) }).await;
            run.apply(&Input::Advance { ms: Millis(200) }).await;
            run.apply(&Input::ToolAnswered {
                tool: Ref::new(tool),
                answer: ToolAnswer::AfterIdle,
                text: AgentText(String::new()),
            })
            .await;
            run.apply(&Input::Advance { ms: Millis(2_000) }).await;
            run
        });
        assert!(run.record.sink().iter().any(|(_, sink)| {
            matches!(sink, lk_agents_fuzz::record::Sink::Reset { segments, .. } if !segments.is_empty())
        }), "the regression did not cut an admitted audio segment");
        let granted = run.record.journal().iter().any(|(_, step)| {
            matches!(step, Step::Speech(SpeechStep::IdleAnswered { hold: Some(_), closing: false }))
        });
        let failures: Vec<_> = run
            .record
            .host_view()
            .speeches()
            .into_iter()
            .filter_map(|speech| speech.error().map(|error| (speech.id, error.to_owned())))
            .collect();
        let tools: Vec<_> = run
            .record
            .resources(lk_agents_fuzz::record::Kind::Tool)
            .map(|tool| (tool.nth(), tool.opened_at()))
            .collect();
        let answers: Vec<_> = run
            .record
            .inputs_view()
            .sent()
            .iter()
            .filter_map(|sent| {
                matches!(sent.input, Input::ToolAnswered { .. }).then_some((sent.at, sent.accepted))
            })
            .collect();
        assert!(
            granted,
            "the tool's foreground was never granted; failures={failures:?}, tools={tools:?}, answers={answers:?}"
        );
    }
}

mod realtime_tool_steps_bounded {
    //! A realtime model's tool rounds stop at `max_tool_steps`.
    use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer};
    use lk_agents_voice::trace::{Step, ToolStep};

    // Catches: the realtime follow-up to a tool step sent with no `tool_choice`, so neither a spent
    // step budget nor a draining agent stopped the model calling tools again: a model whose prompt
    // has it call a tool every response looped, tool after tool, and the reply never ended (found
    // once the fake realtime server called tools). The reference's `_realtime_generation_task` asks
    // for the follow-up with `tool_choice="none"` when `max_tool_steps` is reached or the activity
    // drains; the pipeline step already did.
    #[test]
    fn a_realtime_reply_stops_calling_tools_at_its_budget() {
        let setup = Setup {
            realtime: Some(RealtimeSetup {
                turn_detection: false,
                tool_calls: Some(100),
                ..RealtimeSetup::default()
            }),
            tools: vec!["lookup".into()],
            max_tool_steps: Some(1),
            ..Setup::default()
        };
        let mut inputs =
            vec![Input::GenerateReply { user_input: None, instructions: None, allow: None }];
        // Every call the model makes is answered and every segment played, for a few rounds.
        for round in 0..4 {
            inputs.push(Input::Advance { ms: Millis(10_000) });
            inputs.push(Input::PlayoutStarted { segment: Ref::new(round) });
            inputs.push(Input::PlayoutFinished { segment: Ref::new(round) });
            inputs.push(Input::ToolAnswered {
                tool: Ref::new(round),
                answer: ToolAnswer::Value,
                text: AgentText("ok".into()),
            });
        }
        inputs.push(Input::Advance { ms: Millis(10_000) });
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                crate::playout::apply_with_playout(&mut run, input).await;
            }
            run
        });
        let calls = run
            .record
            .journal()
            .iter()
            .filter(|(_, s)| matches!(s, Step::Tool(ToolStep::CallStarted { .. })))
            .count();
        // The first reply's call, then one more step's (`max_tool_steps=1`), then the reply that may call
        // none.
        assert_eq!(calls, 2, "the model called tools past its budget of one step");
    }
}

mod realtime_tool_update_replies {
    //! A realtime model answers a background tool's final update on its own socket.
    use lk_agents_abi::RealtimeCommandKind;
    use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
    use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer};
    use lk_agents_voice::trace::{RealtimeStep, Step, ToolStep};

    // Catches: the deferred reply to a detached tool's update (`ctx.update()`, then its final
    // result) built as a pipeline reply on a realtime session, which has no LLM: it failed at once
    // with "cannot generate a reply: the session has no llm", the update was reported skipped, and
    // the person never heard the tool's result (survey seed 769, once the fake realtime server called
    // tools). The reference's deferred reply is `session.generate_reply()`, which a realtime model
    // answers on its own socket.
    #[test]
    fn a_detached_realtime_tool_is_answered_on_the_socket() {
        let setup = Setup {
            realtime: Some(RealtimeSetup {
                turn_detection: false,
                tool_calls: Some(100),
                ..RealtimeSetup::default()
            }),
            tools: vec!["lookup".into()],
            // One round of tools: the follow-up speaks without calling another.
            max_tool_steps: Some(1),
            ..Setup::default()
        };
        let inputs = [
            Input::GenerateReply { user_input: None, instructions: None, allow: None },
            Input::Advance { ms: Millis(3_000) },
            Input::PlayoutStarted { segment: Ref::new(0) },
            Input::PlayoutFinished { segment: Ref::new(0) },
            Input::Advance { ms: Millis(100) },
            // The tool answers with a progress update and keeps running in the background.
            Input::ToolAnswered {
                tool: Ref::new(0),
                answer: ToolAnswer::Detached,
                text: AgentText("Checking.".into()),
            },
            // The model speaks the progress, and calls the tool again, which returns; the speaker
            // plays it.
            Input::Advance { ms: Millis(3_000) },
            Input::ToolAnswered {
                tool: Ref::new(1),
                answer: ToolAnswer::Return,
                text: AgentText(String::new()),
            },
            Input::PlayoutStarted { segment: Ref::new(1) },
            Input::PlayoutFinished { segment: Ref::new(1) },
            Input::Advance { ms: Millis(3_000) },
            Input::PlayoutStarted { segment: Ref::new(2) },
            Input::PlayoutFinished { segment: Ref::new(2) },
            Input::Advance { ms: Millis(100) },
            // News from the background, then its end, once the agent is idle.
            Input::ToolUpdate {
                tool: Ref::new(0),
                progress: Some(AgentText("It shipped.".into())),
                end: ToolAnswer::Return,
            },
            Input::Advance { ms: Millis(100) },
            Input::ToolUpdate { tool: Ref::new(0), progress: None, end: ToolAnswer::Return },
            Input::Advance { ms: Millis(1_000) },
        ];
        let run = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            for input in &inputs {
                run.apply(input).await;
            }
            run
        });
        let journal = run.record.journal();
        let scheduled = journal
            .iter()
            .position(|(_, s)| matches!(s, Step::Tool(ToolStep::ReplyScheduled { .. })))
            .expect("the update's reply was scheduled");
        let asked = journal[scheduled..].iter().any(|(_, s)| {
            matches!(
                s,
                Step::Realtime(RealtimeStep::Sent { kind: RealtimeCommandKind::GenerateReply, .. })
            )
        });
        assert!(asked, "the update's reply never went to the realtime server");
        let failed = run
            .record
            .host_view()
            .speeches()
            .iter()
            .any(|s| s.error().is_some_and(|e| e.contains("has no llm")));
        assert!(!failed, "the update's reply failed as a pipeline reply");
    }
}
