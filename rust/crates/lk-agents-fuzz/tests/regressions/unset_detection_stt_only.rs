//! With an STT, no VAD and no turn detection asked for, turns end on finals and the STT's speech
//! events leave the user state alone; under a realtime model the STT's words stay out of history.
use lk_agents_abi::{ChatItem, Role};
use lk_agents_fuzz::drive::{RealtimeSetup, Run, Setup, simulate};
use lk_agents_fuzz::input::{Input, Millis, Ref, UserText};
use lk_agents_voice::agent::Source;
use lk_agents_voice::{events::SessionEvent, trace::Step};

fn utterance(text: &str) -> [Input; 4] {
    [
        Input::SttStart { stt: Ref::new(0) },
        Input::SttFinal { utterance: Ref::new(0), text: UserText(text.into()), speaker: None },
        Input::SttEnd { utterance: Ref::new(0) },
        Input::Advance { ms: Millis(3_000) },
    ]
}

/// (user state changes, user items committed from a turn) over the run.
fn observed(setup: &Setup) -> (usize, usize) {
    let run = simulate(async {
        let mut run = Run::start(setup, 0, Default::default()).await;
        for input in &utterance("hello there") {
            run.apply(input).await;
        }
        run
    });
    let (mut states, mut users) = (0, 0);
    for (_, step) in run.record.journal().iter() {
        match step {
            Step::Emitted(SessionEvent::UserStateChanged { .. }) => states += 1,
            Step::Emitted(SessionEvent::ConversationItemAdded { item, source, .. }) => {
                let user = matches!(&**item, ChatItem::Message(m) if m.role == Role::User);
                if user && matches!(source, Source::UserTurn { .. }) {
                    users += 1;
                }
            }
            _ => {}
        }
    }
    (states, users)
}

// Catches: the unset mode resolved to `stt`, so an STT-only session moved the user state on the
// STT's start and end of speech, which the reference's VAD-based path (`_turn_detection_mode =
// None`) never reads; found by `scripts/parity.py user_turn_default_detection`.
#[test]
fn an_stt_only_turn_ends_on_its_final() {
    let setup = Setup { stt: true, ..Setup::default() };
    assert_eq!(observed(&setup), (0, 1), "no user state change, one user item");
}

// Catches: a realtime turn pushed the STT's words to the server as an item, beside the audio it
// had heard, and kept them as the user item. The reference ignores the STT transcription for a
// realtime model and answers from the committed audio; found by
// `scripts/parity.py realtime_stt_default_detection`.
#[test]
fn a_realtime_turn_keeps_no_stt_words() {
    let realtime = Some(RealtimeSetup { turn_detection: false, ..RealtimeSetup::default() });
    let setup = Setup { stt: true, realtime, ..Setup::default() };
    assert_eq!(observed(&setup), (0, 0), "no user state change, no user item");
}
