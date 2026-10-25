//! What a closing session cut short ends with it: a tool body it answered for, a detached tool,
//! a hook it issued.
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, ToolAnswer, ToolArgs};
use lk_agents_fuzz::record::{Host, Kind};
use lk_agents_voice::agent_session::HookKind;
use lk_agents_voice::events::{SessionEvent, ToolEndStatus, ToolLifecycle};
use lk_agents_voice::trace::{HandoffStep, HookEnd, Step};

fn run(setup: &Setup, inputs: &[Input]) -> Run {
    simulate(async {
        let mut run = Run::start(setup, 0, Default::default()).await;
        for input in inputs {
            run.apply(input).await;
        }
        run
    })
}

// Catches: a blocking tool the close's bound answered with "session closing" left running at the
// host. The reference's close cancels its tool tasks (`_ToolExecutor.aclose`); here the body
// outlived the session and took a late answer (survey seed 189).
#[test]
fn a_tool_the_close_answered_is_cancelled() {
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(20) },
        Input::LlmToolCall {
            chat: Ref::new(0),
            name: "lookup".into(),
            args: ToolArgs("{}".into()),
        },
        Input::LlmEnd { chat: Ref::new(0) },
        Input::Advance { ms: Millis(20) },
        Input::Close { drain: false },
        Input::Advance { ms: Millis(20_000) },
        Input::ToolAnswered {
            tool: Ref::new(0),
            answer: ToolAnswer::Return,
            text: AgentText(String::new()),
        },
    ];
    let run = run(&Setup::default(), &inputs);
    assert!(run.record.driver_finished(), "the session closed");
    let answered = run.record.applied().last().expect("the late answer was applied");
    assert!(!answered.accepted, "the body was cancelled with the session");
    assert!(!run.record.live().has(Kind::Tool, 0), "nothing of the tool is left");
}

// Catches: a tool that detached (`ctx.update()`) still running at the host after the session
// closed: the reference's close cancels it (`_ToolExecutor.aclose`) and tells its end, cancelled,
// before the session's `close`.
#[test]
fn a_detached_tool_is_cancelled_before_the_close() {
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(20) },
        Input::LlmToolCall {
            chat: Ref::new(0),
            name: "lookup".into(),
            args: ToolArgs("{}".into()),
        },
        Input::LlmEnd { chat: Ref::new(0) },
        Input::Advance { ms: Millis(20) },
        Input::ToolAnswered {
            tool: Ref::new(0),
            answer: ToolAnswer::Detached,
            text: AgentText("Looking it up.".into()),
        },
        Input::Advance { ms: Millis(20) },
        Input::Close { drain: false },
        Input::Advance { ms: Millis(20_000) },
    ];
    let run = run(&Setup::default(), &inputs);
    assert!(run.record.driver_finished(), "the session closed");
    assert!(!run.record.live().has(Kind::Tool, 0), "the body was cancelled");
    let host = run.record.host();
    let ended: Vec<usize> = (0..host.len())
        .filter(|&i| {
            matches!(
                &host[i].1,
                Host::Session(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Ended { .. }))
            )
        })
        .collect();
    let [ended] = ended[..] else { panic!("one end told, got {}", ended.len()) };
    let Host::Session(SessionEvent::ToolExecutionUpdated(ToolLifecycle::Ended {
        id,
        call_id,
        status,
        ..
    })) = &host[ended].1
    else {
        unreachable!()
    };
    assert_eq!(*status, ToolEndStatus::Cancelled);
    assert_eq!(*id, format!("{call_id}_final"), "the end of a call that detached");
    let close = host
        .iter()
        .position(|(_, e)| matches!(e, Host::Session(SessionEvent::Close { .. })))
        .expect("the session's close");
    assert!(ended < close, "the end is told before the session's close");
}

// Catches: a cut step's `function_tools_executed` issued past the edge, after the edge abandoned
// every hook, still waited on when the loop stopped (survey seeds 15, 335).
#[test]
fn a_hook_issued_past_the_edge_ends_with_the_session() {
    let inputs = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(20) },
        Input::LlmToolCall {
            chat: Ref::new(0),
            name: "unknown".into(),
            args: ToolArgs("{}".into()),
        },
        Input::Advance { ms: Millis(20) },
        Input::Close { drain: true },
        Input::Advance { ms: Millis(40_000) },
    ];
    let setup = Setup { hook_bound_ms: Some(10_000), ..Setup::default() };
    let run = run(&setup, &inputs);
    assert!(run.record.driver_finished(), "the session closed");
    let journal = run.record.journal();
    let issued = journal
        .iter()
        .find_map(|(_, s)| match s {
            Step::Handoff(HandoffStep::HookIssued {
                hook,
                kind: HookKind::FunctionToolsExecuted,
                ..
            }) => Some(*hook),
            _ => None,
        })
        .expect("the cut step asked its handlers");
    let ended: Vec<_> = journal
        .iter()
        .filter_map(|(_, s)| match s {
            Step::Handoff(HandoffStep::HookFinished { hook, how, .. }) if *hook == issued => {
                Some(*how)
            }
            _ => None,
        })
        .collect();
    assert_eq!(ended, [HookEnd::Abandoned], "abandoned once, by the close");
}
