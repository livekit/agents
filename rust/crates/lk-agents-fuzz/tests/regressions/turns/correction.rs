//! A late final: a turn committed with the utterance's interim, and the final that comes after
//! replaces those words in that turn, by identity.

use lk_agents_abi::{ChatItem, Role};
use lk_agents_fuzz::check::check_all;
use lk_agents_fuzz::drive::{RoomSetup, Run, Setup, StorageSetup, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, RoomIn, TurnMode, UserText};
use lk_agents_fuzz::record::{Kind, StorageStep};
use lk_agents_voice::events::SessionEvent;
use lk_agents_voice::trace::{InterruptCause, InterruptStep, Step};

/// Every version of the user's messages, by id, in order; a manual close's empty turn left out.
fn user_versions(run: &Run) -> Vec<(String, String)> {
    let events = run.record.host_view().events();
    let messages = events.into_iter().filter_map(|(_, event)| match event {
        SessionEvent::ConversationItemAdded { item, .. } => match &**item {
            ChatItem::Message(m) if m.role == Role::User => {
                Some((m.id.clone(), m.raw_text_content().unwrap_or_default()))
            }
            _ => None,
        },
        _ => None,
    });
    messages.filter(|(_, text)| !text.is_empty()).collect()
}

/// Each user message as it reads last, in the order they first appeared.
fn user_messages(run: &Run) -> Vec<String> {
    let mut last: Vec<(String, String)> = Vec::new();
    for (id, text) in user_versions(run) {
        match last.iter_mut().find(|(seen, _)| *seen == id) {
            Some((_, said)) => *said = text,
            None => last.push((id, text)),
        }
    }
    last.into_iter().map(|(_, text)| text).collect()
}

/// The corrections the session made.
fn corrections(run: &Run) -> usize {
    let journal = run.record.journal();
    let corrected = |s: &Step| matches!(s, Step::Emitted(SessionEvent::UserInputCorrected { .. }));
    journal.iter().filter(|(_, s)| corrected(s)).count()
}

/// A thread database that meets no fault: what the call saved is what the rule reads.
fn saved() -> Option<StorageSetup> {
    Some(StorageSetup { faults: false, outage: None, take_answer_ms: None })
}

/// The text the thread saved each user message with.
fn saved_users(run: &Run) -> Vec<String> {
    let storage = run.record.host_view().storage();
    storage
        .iter()
        .rev()
        .find_map(|(_, step)| match step {
            StorageStep::Saved { user_texts, .. } => Some(
                user_texts.iter().map(|(_, text)| text.clone()).filter(|t| !t.is_empty()).collect(),
            ),
            _ => None,
        })
        .unwrap_or_default()
}

fn interim(utterance: u32, text: &str) -> Input {
    Input::SttInterim { utterance: Ref::new(utterance), text: UserText(text.into()), speaker: None }
}

fn final_(utterance: u32, text: &str) -> Input {
    Input::SttFinal { utterance: Ref::new(utterance), text: UserText(text.into()), speaker: None }
}

/// "book a table" heard as an interim, and the host commits the turn before its final comes.
fn committed_as_interim() -> [Input; 5] {
    [
        Input::SttStart { stt: Ref::new(0) },
        interim(0, "book a table"),
        Input::Advance { ms: Millis(100) },
        Input::CommitTurn { skip_reply: false },
        Input::Advance { ms: Millis(1_000) },
    ]
}

/// Replies a correction cut.
fn interrupted(run: &Run) -> usize {
    let journal = run.record.journal();
    let superseded = |s: &Step| {
        matches!(
            s,
            Step::Interrupt(InterruptStep::Interrupted { cause: InterruptCause::Superseded, .. })
        )
    };
    journal.iter().filter(|(_, s)| superseded(s)).count()
}

fn run_all(setup: &Setup, inputs: Vec<Input>) -> Run {
    simulate(async {
        let mut run = Run::start(setup, 0, Default::default()).await;
        for input in &inputs {
            run.apply(input).await;
        }
        run
    })
}

fn closed(mut inputs: Vec<Input>) -> Vec<Input> {
    inputs.extend([
        Input::Close { drain: false },
        Input::Advance { ms: Millis(5_000) },
        Input::AwaitExit {},
    ]);
    inputs
}

/// The final lands while the reply is still being thought: the message reads the full words,
/// and the reply is made again from them, as no turn of its own.
#[test]
fn a_late_final_before_the_reply_is_heard_makes_it_again_from_the_full_words() {
    let setup = Setup {
        stt: true,
        turn: TurnMode::Manual,
        tts: false,
        storage: saved(),
        ..Setup::default()
    };
    let mut inputs = committed_as_interim().to_vec();
    inputs.extend([
        final_(0, "book a table for two"),
        Input::Advance { ms: Millis(100) },
        Input::LlmText { chat: Ref::new(1), text: AgentText("For two, done.".into()) },
        Input::LlmEnd { chat: Ref::new(1) },
        Input::Advance { ms: Millis(1_000) },
    ]);
    let run = run_all(&setup, closed(inputs));

    let versions = user_versions(&run);
    assert_eq!(versions.len(), 2, "{versions:?}");
    assert_eq!(versions[0].0, versions[1].0, "the final rewrote the turn's message");
    assert_eq!(versions[1].1, "book a table for two");
    assert_eq!(run.record.resources(Kind::Chat).count(), 2, "the reply was made again");
    assert_eq!(interrupted(&run), 1, "the unheard reply was cut once");
    assert_eq!(saved_users(&run), ["book a table for two"], "the thread saved the full words");
    let (failures, _) = check_all(&run.record);
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The final lands while the reply plays: the message reads the full words, and the reply
/// plays on.
#[test]
fn a_late_final_while_the_reply_plays_changes_only_the_record() {
    let setup = Setup { stt: true, turn: TurnMode::Manual, storage: saved(), ..Setup::default() };
    let mut inputs = committed_as_interim().to_vec();
    inputs.extend([
        Input::LlmText { chat: Ref::new(0), text: AgentText("Booked, see you soon.".into()) },
        Input::LlmEnd { chat: Ref::new(0) },
        Input::Advance { ms: Millis(200) },
        Input::PlayoutStarted { segment: Ref::new(0) },
        Input::Advance { ms: Millis(200) },
        final_(0, "book a table for two"),
        Input::Advance { ms: Millis(100) },
    ]);
    let run = run_all(&setup, closed(inputs));

    assert_eq!(user_messages(&run), ["book a table for two"]);
    assert_eq!(run.record.resources(Kind::Chat).count(), 1, "the playing reply was not redone");
    assert_eq!(interrupted(&run), 0, "the playing reply was not cut");
    assert_eq!(saved_users(&run), ["book a table for two"], "the thread saved the full words");
    let (failures, _) = check_all(&run.record);
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The words also appear elsewhere in the turn: only the span the interim took changes.
#[test]
fn a_late_final_replaces_only_the_words_the_interim_took() {
    let setup = Setup { stt: true, turn: TurnMode::Manual, tts: false, ..Setup::default() };
    let inputs = vec![
        Input::SttStart { stt: Ref::new(0) },
        final_(0, "book a table"),
        Input::SttEnd { utterance: Ref::new(0) },
        Input::SttStart { stt: Ref::new(0) },
        interim(1, "book a table"),
        Input::Advance { ms: Millis(100) },
        Input::CommitTurn { skip_reply: false },
        Input::Advance { ms: Millis(1_000) },
        final_(1, "book a table for two"),
        Input::Advance { ms: Millis(100) },
    ];
    let run = run_all(&setup, closed(inputs));

    assert_eq!(user_messages(&run), ["book a table book a table for two"]);
    let (failures, _) = check_all(&run.record);
    assert!(failures.is_empty(), "{failures:#?}");
}

/// An empty final for the words a turn took takes nothing back.
#[test]
fn an_empty_final_after_the_turn_took_its_interim_corrects_nothing() {
    let setup = Setup { stt: true, turn: TurnMode::Manual, tts: false, ..Setup::default() };
    let mut inputs = committed_as_interim().to_vec();
    inputs.extend([final_(0, ""), Input::Advance { ms: Millis(100) }]);
    let run = run_all(&setup, closed(inputs));

    assert_eq!(user_messages(&run), ["book a table"]);
    assert_eq!(corrections(&run), 0);
    let (failures, _) = check_all(&run.record);
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A short final-only utterance after the turn took an interim is a turn of its own.
#[test]
fn a_final_only_utterance_after_the_turn_took_an_interim_is_a_new_turn() {
    let setup = Setup { stt: true, turn: TurnMode::Stt, tts: false, ..Setup::default() };
    let mut inputs = committed_as_interim().to_vec();
    inputs.extend([
        Input::LlmText { chat: Ref::new(0), text: AgentText("For how many?".into()) },
        Input::LlmEnd { chat: Ref::new(0) },
        Input::Advance { ms: Millis(500) },
        Input::SttStart { stt: Ref::new(0) },
        final_(1, "yes"),
        Input::SttEnd { utterance: Ref::new(1) },
        Input::Advance { ms: Millis(2_000) },
    ]);
    let run = run_all(&setup, closed(inputs));

    assert_eq!(user_messages(&run), ["book a table", "yes"]);
    assert_eq!(corrections(&run), 0);
    let (failures, _) = check_all(&run.record);
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A pause inside a turn: the provider's late final lands while the turn is open, joins it, and
/// the reply answers the whole turn.
#[test]
fn a_pause_inside_a_turn_keeps_the_late_final_in_it() {
    let setup = Setup { stt: true, turn: TurnMode::Stt, tts: false, ..Setup::default() };
    let inputs = vec![
        Input::SttStart { stt: Ref::new(0) },
        interim(0, "book a flight to Par"),
        Input::Advance { ms: Millis(2_000) },
        final_(0, "book a flight to Paris"),
        Input::SttEnd { utterance: Ref::new(0) },
        Input::Advance { ms: Millis(100) },
        Input::SttStart { stt: Ref::new(0) },
        final_(1, "on Monday"),
        Input::SttEnd { utterance: Ref::new(1) },
        Input::Advance { ms: Millis(2_000) },
    ];
    let run = run_all(&setup, closed(inputs));

    assert_eq!(user_messages(&run), ["book a flight to Paris on Monday"]);
    assert_eq!(corrections(&run), 0);
    assert_eq!(run.record.resources(Kind::Chat).count(), 1, "one reply, to the whole turn");
    let (failures, _) = check_all(&run.record);
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The room shows the full words in the segment that showed the interim the turn took, and in no
/// segment of their own.
#[test]
fn the_rooms_segment_reads_the_corrected_words() {
    let setup = Setup {
        stt: true,
        turn: TurnMode::Manual,
        tts: false,
        room: Some(RoomSetup { close_on_disconnect: false, delete_room_on_close: false }),
        ..Setup::default()
    };
    let mut inputs =
        vec![Input::Room { event: RoomIn::Linked }, Input::Advance { ms: Millis(100) }];
    inputs.extend(committed_as_interim());
    inputs.extend([final_(0, "book a table for two"), Input::Advance { ms: Millis(100) }]);
    let run = run_all(&setup, inputs);

    let mut segments: Vec<(String, String)> = Vec::new();
    for stream in run.room_text_streams() {
        let attributes = &stream.opts.attributes;
        let Some(id) = attributes.get(lk_agents_abi::ATTRIBUTE_TRANSCRIPTION_SEGMENT_ID) else {
            continue;
        };
        match segments.iter_mut().find(|(seen, _)| seen == id) {
            Some((_, text)) => *text = stream.text(),
            None => segments.push((id.clone(), stream.text())),
        }
    }
    let user: Vec<&str> = segments
        .iter()
        .map(|(_, text)| text.as_str())
        .filter(|text| text.starts_with("book"))
        .collect();
    assert_eq!(user, ["book a table for two"], "{segments:?}");
}

/// A late final after the turn was saved and answered: the next turn reloads the thread, which
/// still holds the words it replaced, and saves the corrected message, not the stale one.
#[test]
fn a_late_final_after_the_turn_was_saved_reaches_the_thread_with_the_next_turn() {
    let setup = Setup {
        stt: true,
        turn: TurnMode::Manual,
        tts: false,
        storage: saved(),
        ..Setup::default()
    };
    let mut inputs = committed_as_interim().to_vec();
    inputs.extend([
        Input::LlmText { chat: Ref::new(0), text: AgentText("Booked.".into()) },
        Input::LlmEnd { chat: Ref::new(0) },
        Input::Advance { ms: Millis(1_000) },
        final_(0, "book a table for two"),
        Input::Advance { ms: Millis(100) },
        Input::SttStart { stt: Ref::new(0) },
        final_(1, "thanks"),
        Input::SttEnd { utterance: Ref::new(1) },
        Input::Advance { ms: Millis(100) },
        Input::CommitTurn { skip_reply: false },
        Input::Advance { ms: Millis(1_000) },
        Input::LlmText { chat: Ref::new(1), text: AgentText("Anytime.".into()) },
        Input::LlmEnd { chat: Ref::new(1) },
        Input::Advance { ms: Millis(1_000) },
    ]);
    let run = run_all(&setup, closed(inputs));

    assert_eq!(saved_users(&run), ["book a table for two", "thanks"]);
    let (failures, _) = check_all(&run.record);
    assert!(failures.is_empty(), "{failures:#?}");
}
