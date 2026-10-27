//! A reply asked while the realtime server generates one of its own waits for it, and is heard.
use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, RtEvent, ToolAnswer};
use lk_agents_fuzz::record::{Breach, Recording, Tag};

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

/// Whether the n-th realtime generation's audio reached the speaker.
fn heard(r: &Recording, n: u32) -> bool {
    r.host_view().audio_pushed().iter().any(|p| {
        p.tags.iter().any(|t| matches!(t, Tag::Generated { generation, .. } if *generation == n))
    })
}

// Catches: a host `generate_reply` whose `response.create` crossed a reply the server started on
// its own: refused (`conversation_already_has_active_response`), and the host's reply was lost.
// It waits for the server's reply and goes after it, one at a time, as speeches are scheduled.
#[test]
fn a_reply_crossing_the_servers_own_waits_and_is_heard() {
    // Only a server with its own turn detection opens a reply unasked (its idle timeout).
    let setup = Setup {
        realtime: Some(RealtimeSetup { turn_detection: true, ..RealtimeSetup::default() }),
        ..Setup::default()
    };
    let r = realtime(setup, &[
        // The server opens a reply of its own as the host asks for one.
        Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::SecondGeneration },
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(3_000) },
        Input::PlayoutStarted { segment: Ref::new(0) },
        Input::PlayoutFinished { segment: Ref::new(0) },
        Input::Advance { ms: Millis(10_000) },
    ]);
    let failed: Vec<_> =
        r.host_view().speeches().into_iter().filter_map(|s| s.error().map(str::to_owned)).collect();
    assert!(failed.is_empty(), "{failed:?}");
    assert!(heard(&r, 1), "the host's reply, asked again after the server's, is heard");
}

// Catches: the tools' follow-up asked over a reply the server was still generating (one the host
// asked for while the tool ran): refused (`conversation_already_has_active_response`), the tools'
// result never spoken. It waits for that reply to be done, which is not cut.
#[test]
fn a_tool_follow_up_waits_for_the_reply_generating() {
    let setup = Setup {
        tools: vec!["lookup_order".into()],
        realtime: Some(RealtimeSetup {
            turn_detection: false,
            tool_calls: Some(100),
            ..RealtimeSetup::default()
        }),
        ..Setup::default()
    };
    let r = realtime(setup, &[
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(5_000) },
        // The reply's words play out; the tool still runs, and the host's reply goes.
        Input::PlayoutStarted { segment: Ref::new(0) },
        Input::PlayoutFinished { segment: Ref::new(0) },
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(600) },
        Input::ToolAnswered {
            tool: Ref::new(0),
            answer: ToolAnswer::Value,
            text: AgentText("shipped".into()),
        },
        Input::Advance { ms: Millis(5_000) },
    ]);
    assert!(refused(&r).is_empty(), "{:?}", refused(&r));
    // The host's reply (the second generation) played on; the follow-up came after it.
    let cut = r.journal().iter().any(|(_, s)| {
        matches!(s, lk_agents_voice::trace::Step::Realtime(lk_agents_voice::trace::RealtimeStep::Sent {
            kind: lk_agents_abi::RealtimeCommandKind::Interrupt,
            ..
        }))
    });
    assert!(!cut && heard(&r, 1), "the host's reply is not cut");
}
