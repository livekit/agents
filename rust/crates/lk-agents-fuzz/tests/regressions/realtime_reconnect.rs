//! A realtime session reconnecting restores the server's state once, and only what it holds.
use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
use lk_agents_fuzz::input::{AgentText, Count, Input, Millis, Ref, Role, RtEvent, UserText};
use lk_agents_fuzz::record::{Breach, Recording};

fn realtime(inputs: &[Input]) -> Recording {
    let setup = Setup {
        realtime: Some(RealtimeSetup { turn_detection: false, ..RealtimeSetup::default() }),
        ..Setup::default()
    };
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in inputs {
            run.apply(input).await;
        }
        run
    });
    run.record
}

/// What the server refused of the client.
fn refused(r: &Recording) -> Vec<&Breach> {
    let breaches = r.inference_view().realtime_breaches();
    breaches.iter().map(|(_, _, b)| b).filter(|b| matches!(b, Breach::Refused { .. })).collect()
}

/// The server drops the socket and takes connections again.
fn drop_socket() -> [Input; 3] {
    [
        Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::Disconnected },
        Input::Advance { ms: Millis(100) },
        Input::RealtimeEvent { socket: Ref::new(0), event: RtEvent::Reconnected },
    ]
}

// Catches: the driver replaying its conversation onto the new connection after the session had
// replayed it itself (`_reconnect`, which the reference leaves alone): every item went twice, and
// the server refused each second one (`item_create_duplicate_item_id`).
#[test]
fn a_reconnect_restores_the_conversation_once() {
    let mut inputs = vec![
        Input::GenerateReply { user_input: Some(UserText("where is my order".into())), instructions: None, allow: None },
        Input::Advance { ms: Millis(10_000) },
    ];
    inputs.extend(drop_socket());
    inputs.push(Input::Advance { ms: Millis(5_000) });
    let r = realtime(&inputs);
    assert!(refused(&r).is_empty(), "{:?}", refused(&r));
}

// Catches: the native session counting the audio it sent before a reconnect as still in the
// server's buffer. The new connection starts empty, so the commit of the person's turn went out
// over nothing and the server refused it (`input_audio_buffer_commit_empty`).
#[test]
fn a_commit_after_a_reconnect_names_audio_the_server_has() {
    let mut inputs = vec![
        Input::MicFrames { mic: Ref::new(0), chunks: Count(30) },
        Input::Advance { ms: Millis(300) },
    ];
    inputs.extend(drop_socket());
    inputs.extend([
        Input::Advance { ms: Millis(5_000) },
        Input::CommitUserTurn {
            transcript_timeout: Millis(0),
            stt_flush: Millis(0),
            skip_reply: false,
        },
        Input::Advance { ms: Millis(1_000) },
    ]);
    let r = realtime(&inputs);
    assert!(refused(&r).is_empty(), "{:?}", refused(&r));
}

// Catches: a reconnect that crossed a context update the lost connection had not confirmed
// replaying the confirmed conversation and then resending that update's frames: what the update
// deletes and creates again under the same id was created twice, and what was sent right after
// the replay was diffed against nothing (`item_create_duplicate_item_id`). The new connection
// gets the conversation the server was being brought to, once.
#[test]
fn a_reconnect_across_an_unconfirmed_edit_creates_each_item_once() {
    let edit = |text: &str| Input::UpdateChatCtx {
        items: vec![Role::System, Role::User],
        text: AgentText(text.into()),
    };
    let mut inputs = vec![
        Input::GenerateReply { user_input: None, instructions: None, allow: None },
        Input::Advance { ms: Millis(5_000) },
        edit("Hello."),
        Input::Advance { ms: Millis(1_000) },
        // A second edit, its items' ids the first's, and the socket goes before it is confirmed.
        edit("Hello again."),
    ];
    inputs.extend(drop_socket());
    inputs.extend([edit("Hello once more."), Input::Advance { ms: Millis(5_000) }]);
    let r = realtime(&inputs);
    assert!(refused(&r).is_empty(), "{:?}", refused(&r));
}
