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
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(3_000) },
        Input::PlayoutStarted { segment: Ref::new(0) },
        Input::Advance { ms: Millis(200) },
        // The speaker throws its playback away: the step's audio is cut, the speech is not.
        Input::SinkReset { output: Ref::new(0) },
        Input::Advance { ms: Millis(200) },
        // The tool asks the agent to go idle from inside its body (`ctx.foreground()`).
        Input::ToolAnswered {
            tool: Ref::new(0),
            answer: ToolAnswer::AfterIdle,
            text: AgentText(String::new()),
        },
        Input::Advance { ms: Millis(2_000) },
    ];
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    });
    let granted = run.record.journal().iter().any(|(_, step)| {
        matches!(step, Step::Speech(SpeechStep::IdleAnswered { hold: Some(_), closing: false }))
    });
    assert!(granted, "the tool's foreground was never granted");
}
