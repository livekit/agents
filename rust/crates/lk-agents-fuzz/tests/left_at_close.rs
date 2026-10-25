//! What a closing session cut short ends with it: a hook it issued.
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{Input, Millis, Ref, ToolArgs};
use lk_agents_voice::agent_session::HookKind;
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
