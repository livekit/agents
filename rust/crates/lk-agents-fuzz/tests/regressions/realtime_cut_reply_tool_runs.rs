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
            super::playout::apply_with_playout(&mut run, &Input::Advance { ms: Millis(100) }).await;
            let tool = run.record.live().of(lk_agents_fuzz::record::Kind::Tool).first().copied();
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
