//! A developer's context edits leave the realtime session sending only what the server takes.
use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, Role, ToolAnswer};
use lk_agents_fuzz::record::{Breach, Recording};

fn realtime(setup: Setup, inputs: &[Input]) -> Recording {
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in inputs {
            run.apply(input).await;
        }
        run
    });
    run.record
}

fn refused(r: &Recording) -> Vec<&Breach> {
    let breaches = r.inference_view().realtime_breaches();
    breaches.iter().map(|(_, _, b)| b).filter(|b| matches!(b, Breach::Refused { .. })).collect()
}

fn edit() -> Input {
    Input::UpdateChatCtx { items: vec![Role::System, Role::User], text: AgentText("Hello.".into()) }
}

// Catches: the output of a tool whose call the developer's `update_chat_ctx` removed while it
// ran, pushed to the server anyway: it names a call the server no longer holds
// (`invalid_tool_call_id`). It is not sent.
#[test]
fn an_output_for_a_removed_call_is_not_sent() {
    let setup = Setup {
        tools: vec!["lookup".into()],
        realtime: Some(RealtimeSetup {
            turn_detection: false,
            tool_calls: Some(100),
            ..RealtimeSetup::default()
        }),
        ..Setup::default()
    };
    let r = realtime(setup, &[
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(3_000) },
        Input::PlayoutStarted { segment: Ref::new(0) },
        Input::PlayoutFinished { segment: Ref::new(0) },
        // The developer replaces the context, call included, while the tool runs.
        edit(),
        Input::Advance { ms: Millis(1_000) },
        Input::ToolAnswered {
            tool: Ref::new(0),
            answer: ToolAnswer::Value,
            text: AgentText("shipped".into()),
        },
        Input::Advance { ms: Millis(5_000) },
    ]);
    assert!(refused(&r).is_empty(), "{:?}", refused(&r));
}

// Catches: a context update sent while the previous one was unconfirmed, diffed against what
// the server had confirmed rather than what it was being brought to: it deleted again the items
// the first one had deleted (`item_delete_invalid_item_id`).
#[test]
fn an_edit_over_an_unconfirmed_edit_deletes_once() {
    let setup = Setup {
        realtime: Some(RealtimeSetup { turn_detection: false, ..RealtimeSetup::default() }),
        ..Setup::default()
    };
    let r = realtime(setup, &[
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(5_000) },
        edit(),
        Input::UpdateChatCtx { items: vec![Role::User], text: AgentText("Again.".into()) },
        Input::Advance { ms: Millis(2_000) },
    ]);
    assert!(refused(&r).is_empty(), "{:?}", refused(&r));
}
