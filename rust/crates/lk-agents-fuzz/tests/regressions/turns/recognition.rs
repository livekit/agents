//! What a recogniser reports reaches the turn through the production speech rules: a connection
//! that drops mid-speech keeps the words it had, an end its provider reports before the words
//! waits for them, and a recogniser without its own detection starts and ends speech with words.

use lk_agents_abi::{ChatItem, Role};
use lk_agents_fuzz::check::check_all;
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::fakes::SttSpeech;
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, TurnMode, UserText};
use lk_agents_fuzz::record::Kind;
use lk_agents_voice::events::SessionEvent;
use lk_agents_voice::trace::Step;

fn user_messages(run: &Run) -> Vec<String> {
    let events = run.record.host_view().events();
    events
        .into_iter()
        .filter_map(|(_, event)| match event {
            SessionEvent::ConversationItemAdded { item, .. } => match &**item {
                ChatItem::Message(m) if m.role == Role::User => m.raw_text_content(),
                _ => None,
            },
            _ => None,
        })
        .filter(|text| !text.is_empty())
        .collect()
}

fn run_all(setup: &Setup, inputs: &[Input]) -> Run {
    let run = simulate(async {
        let mut run = Run::start(setup, 0, Default::default()).await;
        for input in inputs {
            run.apply(input).await;
        }
        run
    });
    let (failures, _) = check_all(&run.record);
    assert!(failures.is_empty(), "{failures:#?}");
    run
}

fn words(text: &str) -> UserText {
    UserText(text.into())
}

fn setup(stt_speech: SttSpeech) -> Setup {
    Setup { stt: true, turn: TurnMode::Stt, tts: false, stt_speech, ..Setup::default() }
}

/// A provider's end must not leave an STT-only turn waiting forever for its missing final.
/// The late final corrects the original item rather than adding a second user turn.
#[test]
fn a_missing_final_commits_the_interim_and_a_late_final_corrects_it() {
    let run = run_all(
        &setup(SttSpeech::default()),
        &[
            Input::SttStart { stt: Ref::new(0) },
            Input::SttInterim {
                utterance: Ref::new(0),
                text: words("book a table"),
                speaker: None,
            },
            Input::SttEnd { utterance: Ref::new(0) },
            Input::Advance { ms: Millis(4_000) },
            Input::SttFinal {
                utterance: Ref::new(0),
                text: words("book a table for two"),
                speaker: None,
            },
            Input::Advance { ms: Millis(1_000) },
        ],
    );
    let messages: Vec<_> = run
        .record
        .host_view()
        .events()
        .into_iter()
        .filter_map(|(_, event)| match event {
            SessionEvent::ConversationItemAdded { item, .. } => item
                .as_message()
                .filter(|m| m.role == Role::User)
                .map(|m| (m.id.clone(), m.raw_text_content().unwrap_or_default())),
            _ => None,
        })
        .collect();
    assert!(
        messages.iter().any(|(_, text)| text == "book a table"),
        "the deadline committed provisional words: {messages:?}"
    );
    assert!(
        messages.iter().any(|(_, text)| text == "book a table for two"),
        "the late final corrected the words: {messages:?}"
    );
    assert!(
        messages.iter().all(|(id, _)| id == &messages[0].0),
        "the correction kept its item identity: {messages:?}"
    );
}

/// Catches: a connection boundary that ended the speech without its interim, so the turn lost
/// the words the provider had sent before it dropped.
#[test]
fn a_dropped_connection_keeps_the_interim_in_the_turn() {
    let run = run_all(
        &setup(SttSpeech::default()),
        &[
            Input::SttStart { stt: Ref::new(0) },
            Input::SttInterim {
                utterance: Ref::new(0),
                text: words("book a table"),
                speaker: None,
            },
            Input::Advance { ms: Millis(200) },
            Input::SttDrop { stt: Ref::new(0) },
            Input::Advance { ms: Millis(2_000) },
        ],
    );
    assert_eq!(user_messages(&run), ["book a table"]);
}

/// Catches: an end of speech a provider that transcribes after the speech ends reports before its
/// words, taken at its word: the person stopped with nothing said, and their words came as a
/// speech of their own.
#[test]
fn an_end_before_the_words_waits_for_the_final() {
    let speech = SttSpeech { provider_vad: true, transcribes_after_speech_ends: true };
    let run = run_all(
        &setup(speech),
        &[
            Input::SttStart { stt: Ref::new(0) },
            Input::Advance { ms: Millis(800) },
            Input::SttEnd { utterance: Ref::new(0) },
            Input::Advance { ms: Millis(600) },
            Input::SttFinal {
                utterance: Ref::new(0),
                text: words("what time is it"),
                speaker: None,
            },
            Input::Advance { ms: Millis(2_000) },
            Input::LlmText { chat: Ref::new(0), text: AgentText("Noon.".into()) },
            Input::LlmEnd { chat: Ref::new(0) },
            Input::Advance { ms: Millis(1_000) },
        ],
    );
    assert_eq!(user_messages(&run), ["what time is it"]);
    assert_eq!(run.record.resources(Kind::Chat).count(), 1, "one reply, to the words");
    // One speech, ended with its words: the session heard the end only once the final came.
    let heard: Vec<&str> = run
        .record
        .journal()
        .iter()
        .filter_map(|(_, step)| match step {
            Step::Emitted(SessionEvent::StartOfSpeech(_)) => Some("start"),
            Step::Emitted(SessionEvent::UserInputTranscribed(t)) if t.is_final => Some("final"),
            Step::Emitted(SessionEvent::EndOfSpeech(_)) => Some("end"),
            _ => None,
        })
        .collect();
    assert_eq!(heard, ["start", "final", "end"]);
}

/// Catches: a recogniser without its own detection whose speech waited on an end it never
/// reports: the person stayed speaking past their final and the turn never ended.
#[test]
fn without_its_own_detection_the_final_ends_the_speech() {
    let speech = SttSpeech { provider_vad: false, transcribes_after_speech_ends: false };
    let run = run_all(
        &setup(speech),
        &[
            Input::SttStart { stt: Ref::new(0) },
            Input::SttInterim { utterance: Ref::new(0), text: words("what time"), speaker: None },
            Input::Advance { ms: Millis(400) },
            Input::SttFinal {
                utterance: Ref::new(0),
                text: words("what time is it"),
                speaker: None,
            },
            Input::Advance { ms: Millis(2_000) },
        ],
    );
    assert_eq!(user_messages(&run), ["what time is it"]);
    assert_eq!(run.record.resources(Kind::Chat).count(), 1, "the turn ended on the final");
}
