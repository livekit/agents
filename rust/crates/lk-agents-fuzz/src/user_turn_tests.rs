//! User turns (`start_user_turn()`) and push-to-talk, end to end over the real executors.

use lk_agents_abi::voice::session::UserTurnFailed;
use lk_agents_abi::{ChatItem, Role};
use lk_agents_fuzz::drive::{Case, Run, Setup, run, simulate};
use lk_agents_fuzz::input::{Count, Input, Knob, Millis, Ref, TurnMode, UserText};
use lk_agents_fuzz::record::{Answer, Kind};
use lk_agents_voice::agent::Source;
use lk_agents_voice::{events::SessionEvent, trace::Step};

fn push_to_talk() -> Setup {
    Setup { stt: true, push_to_talk: true, ..Setup::default() }
}

fn mic(chunks: u32) -> Input {
    Input::MicFrames { mic: Ref::new(0), chunks: Count(chunks) }
}

fn said(stt: u32, utterance: u32, text: &str) -> [Input; 2] {
    [
        Input::SttStart { stt: Ref::new(stt) },
        Input::SttFinal {
            utterance: Ref::new(utterance),
            text: UserText(text.into()),
            speaker: None,
        },
    ]
}

fn wait(ms: u64) -> Input {
    Input::Advance { ms: Millis(ms) }
}

fn played(setup: &Setup, inputs: Vec<Input>) -> Run {
    simulate(async {
        let mut run = Run::start(setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    })
}

/// What each `end()` answered, in order.
fn ends(run: &Run) -> Vec<Answer> {
    let calls = run.record.host_view().api_calls();
    let answers = calls.into_iter().filter_map(|call| match call.returned() {
        Some((_, answer @ Answer::UserTurn(_))) => Some(answer.clone()),
        _ => None,
    });
    answers.collect()
}

/// The user messages turns added to the conversation.
fn user_items(run: &Run) -> Vec<String> {
    let items = run.record.journal().iter().filter_map(|(_, step)| match step {
        Step::Emitted(SessionEvent::ConversationItemAdded { item, source, .. }) => match &**item {
            ChatItem::Message(m)
                if m.role == Role::User && matches!(source, Source::UserTurn { .. }) =>
            {
                m.raw_text_content()
            }
            _ => None,
        },
        _ => None,
    });
    items.collect()
}

fn ended_events(run: &Run) -> Vec<bool> {
    let ended = run.record.journal().iter().filter_map(|(_, step)| match step {
        Step::Emitted(SessionEvent::UserTurnEnded(e)) => Some(!e.cancelled),
        _ => None,
    });
    ended.collect()
}

fn ok_message(answer: &Answer) -> Option<String> {
    match answer {
        Answer::UserTurn(answer) => match &**answer {
            Ok(result) => result.message.as_ref().and_then(|m| m.raw_text_content()),
            Err(_) => None,
        },
        _ => None,
    }
}

// Catches: a press-and-release not committing what was said, or committing it twice.
#[test]
fn a_released_turn_commits_its_words_once() {
    let mut inputs = vec![Input::StartUserTurn {}, mic(30)];
    inputs.extend(said(0, 0, "book a table for two"));
    inputs.extend([Input::EndUserTurn { reply: false }, mic(30), wait(100)]);
    inputs.extend([Input::SttEnd { utterance: Ref::new(0) }, wait(500)]);
    let run = played(&push_to_talk(), inputs);
    let ends = ends(&run);
    assert_eq!(ends.len(), 1, "{ends:?}");
    assert_eq!(ok_message(&ends[0]).as_deref(), Some("book a table for two"));
    assert_eq!(user_items(&run), ["book a table for two"]);
    assert_eq!(ended_events(&run), [true]);
}

// Catches: `end()` resolving on the first final after release while the stream still had words,
// so the late final leaks into the next turn.
#[test]
fn a_final_after_the_release_belongs_to_the_turn() {
    let mut inputs = vec![Input::StartUserTurn {}, mic(30), Input::SttStart { stt: Ref::new(0) }];
    inputs.extend([Input::EndUserTurn { reply: false }, mic(30), wait(100)]);
    inputs.extend([
        Input::SttFinal {
            utterance: Ref::new(0),
            text: UserText("cancel my order".into()),
            speaker: None,
        },
        Input::SttEnd { utterance: Ref::new(0) },
        wait(500),
    ]);
    let run = played(&push_to_talk(), inputs);
    assert_eq!(ends(&run).first().and_then(ok_message).as_deref(), Some("cancel my order"));
    assert_eq!(user_items(&run), ["cancel my order"]);
}

// Catches: a cancelled turn's words transcribed into history, or its stream finalized.
#[test]
fn a_cancelled_turn_commits_nothing() {
    let mut inputs = vec![Input::StartUserTurn {}, mic(30)];
    inputs.extend(said(0, 0, "never mind"));
    inputs.extend([Input::CancelUserTurn {}, wait(500)]);
    let run = played(&push_to_talk(), inputs);
    assert!(user_items(&run).is_empty());
    assert_eq!(ended_events(&run), [false]);
}

// Catches: an empty tap replying to an empty user message.
#[test]
fn an_empty_tap_adds_nothing_and_replies_nothing() {
    let inputs =
        vec![Input::StartUserTurn {}, Input::EndUserTurn { reply: true }, mic(30), wait(500)];
    let run = played(&push_to_talk(), inputs);
    let ends = ends(&run);
    assert!(
        matches!(&ends[..], [Answer::UserTurn(a)] if matches!(&**a, Ok(r) if r.empty && r.message.is_none()))
    );
    assert!(user_items(&run).is_empty());
    assert!(run.record.live().of(Kind::Chat).is_empty(), "no reply was asked of the model");
}

// Catches: audio outside a push-to-talk turn reaching the recogniser or the VAD.
#[test]
fn push_to_talk_hears_nothing_outside_a_turn() {
    let setup = Setup { vad: true, ..push_to_talk() };
    let run = played(&setup, vec![mic(50), wait(1_000)]);
    assert!(run.record.inference_view().frames_heard().is_empty());
}

// Catches: a second press while a turn is open opening a second turn.
#[test]
fn a_second_start_is_refused_while_a_turn_is_open() {
    let inputs = vec![Input::StartUserTurn {}, Input::StartUserTurn {}, wait(100)];
    let run = played(&push_to_talk(), inputs);
    let accepted = run.record.inputs_view().accepted().count();
    let started = run.record.journal().iter().filter(|(_, step)| {
        matches!(step, Step::Turn(lk_agents_voice::trace::TurnStep::UserTurnStarted { .. }))
    });
    assert_eq!(started.count(), 1, "{accepted} inputs accepted");
}

// Catches: a close leaving a released turn's `end()` waiting forever.
#[test]
fn the_close_fails_a_released_turn() {
    let mut inputs = vec![Input::StartUserTurn {}, mic(30), Input::SttStart { stt: Ref::new(0) }];
    inputs.extend([Input::EndUserTurn { reply: true }, Input::Close { drain: false }, wait(5_000)]);
    let run = played(&push_to_talk(), inputs);
    assert!(matches!(&ends(&run)[..], [Answer::UserTurn(a)] if **a == Err(UserTurnFailed::Closed)));
}

// Catches: a provider that never finalizes holding `end()` past its bound, or losing the words
// that did arrive.
#[test]
fn a_provider_that_never_closes_commits_what_arrived_at_the_bound() {
    let mut inputs = vec![Input::StartUserTurn {}, mic(30)];
    inputs.extend(said(0, 0, "where is my parcel"));
    // No end of speech: the stream stays open after its input ended.
    inputs.extend([Input::EndUserTurn { reply: false }, mic(30), wait(6_000)]);
    let run = played(&push_to_talk(), inputs);
    assert_eq!(ends(&run).first().and_then(ok_message).as_deref(), Some("where is my parcel"));
}

// Catches: a stuck button holding a turn open past `user_turn_limit`.
#[test]
fn the_turn_limit_ends_an_open_turn_as_a_release() {
    let setup = Setup { user_turn_max_words: Some(3), ..push_to_talk() };
    let mut inputs = vec![Input::StartUserTurn {}, mic(30)];
    inputs.extend(said(0, 0, "one two three four five"));
    inputs.extend([mic(30), Input::SttEnd { utterance: Ref::new(0) }, wait(500)]);
    let run = played(&setup, inputs);
    assert_eq!(user_items(&run), ["one two three four five"]);
    assert_eq!(ended_events(&run), [true]);
}

// Catches: turning push-to-talk off mid-turn leaving the turn open forever.
#[test]
fn turning_push_to_talk_off_ends_the_open_turn() {
    let mut inputs = vec![Input::StartUserTurn {}, mic(30)];
    inputs.extend(said(0, 0, "hello"));
    inputs.extend([
        Input::Configure {
            knob: Knob::PushToTalk,
            mode: TurnMode::Default,
            ms: Millis(0),
            on: false,
            words: Count(0),
        },
        mic(30),
        Input::SttEnd { utterance: Ref::new(0) },
        wait(500),
    ]);
    let run = played(&push_to_talk(), inputs);
    assert_eq!(user_items(&run), ["hello"]);
}

// Every rule holds over a press, a cancel and a press under push-to-talk, closed.
#[test]
fn user_turns_keep_every_rule() {
    let mut inputs = vec![Input::StartUserTurn {}, mic(20)];
    inputs.extend(said(0, 0, "first"));
    inputs.extend([
        Input::EndUserTurn { reply: true },
        mic(30),
        Input::SttEnd { utterance: Ref::new(0) },
    ]);
    inputs.extend([wait(500), Input::StartUserTurn {}, mic(10), Input::CancelUserTurn {}]);
    inputs.extend([wait(500), Input::Close { drain: false }, wait(20_000), Input::AwaitExit {}]);
    let outcome = run(Case { setup: push_to_talk(), inputs, ..Case::default() });
    assert!(outcome.failures.is_empty(), "{:#?}", outcome.failures);
}

// Catches: a client that stops sending audio on release (it unmutes only while pressed) leaving
// the tail waiting for audio that never comes, so every turn ends on the 5 s bound.
#[test]
fn a_muted_release_ends_its_tail_on_time() {
    let mut inputs = vec![Input::StartUserTurn {}, mic(30), wait(100)];
    inputs.extend(said(0, 0, "two coffees please"));
    inputs.extend([Input::SttEnd { utterance: Ref::new(0) }, Input::EndUserTurn { reply: false }]);
    inputs.push(wait(1_000));
    let run = played(&push_to_talk(), inputs);
    assert_eq!(ends(&run).first().and_then(ok_message).as_deref(), Some("two coffees please"));
}
