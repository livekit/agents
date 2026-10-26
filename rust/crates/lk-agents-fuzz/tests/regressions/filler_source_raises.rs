//! A host filler callable that raises ends the filler, as the reference's loop dies on it.
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, HookAnswer, Input, Millis, Ref, ToolArgs};
use lk_agents_voice::agent_session::HookKind;
use lk_agents_voice::trace::{FillerEnd, GenerationStep, HandoffStep, Step};

/// The `FillerSource` hooks' positions among the hooks issued, as a `Ref<Hook>` counts them.
fn sources(run: &Run) -> Vec<usize> {
    run.record
        .journal()
        .iter()
        .filter_map(|(_, s)| match s {
            Step::Handoff(HandoffStep::HookIssued { kind, .. }) => Some(*kind),
            _ => None,
        })
        .enumerate()
        .filter_map(|(nth, kind)| (kind == HookKind::FillerSource).then_some(nth))
        .collect()
}

// Catches: a `Failed` answer to `FillerSource` skipping the fire and rescheduling it. In the
// reference `src(step)` raising ends `_FillerScheduler._loop`, so no further filler plays.
#[test]
fn a_raising_filler_callable_ends_the_filler() {
    let opening = [
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(20) },
        Input::LlmToolCall {
            chat: Ref::new(0),
            name: "lookup".into(),
            args: ToolArgs("{}".into()),
        },
        Input::LlmEnd { chat: Ref::new(0) },
        Input::Advance { ms: Millis(20) },
        Input::FillerStart {
            tool: Ref::new(0),
            speech: Ref::new(0),
            delay: Millis(10),
            interval: Some(Millis(30)),
            text: None,
        },
        Input::Advance { ms: Millis(20) },
    ];
    let setup = Setup { hook_bound_ms: Some(1_000), ..Setup::default() };
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in &opening {
            run.apply(input).await;
        }
        let [nth] = sources(&run)[..] else { panic!("one filler source asked") };
        let hook = Ref::new(nth as u32);
        let text = AgentText(String::new());
        run.apply(&Input::HookDone { hook, answer: HookAnswer::Failed, text }).await;
        run.apply(&Input::Advance { ms: Millis(200) }).await;
        run
    });
    assert_eq!(sources(&run).len(), 1, "the callable is not asked again");
    let ends: Vec<_> = run
        .record
        .journal()
        .iter()
        .filter_map(|(_, s)| match s {
            Step::Generation(GenerationStep::FillerFired { .. }) => Some(None),
            Step::Generation(GenerationStep::FillerStopped { cause, .. }) => Some(Some(*cause)),
            _ => None,
        })
        .collect();
    assert_eq!(ends, [Some(FillerEnd::Refused)], "no fire, the filler ends");
}
