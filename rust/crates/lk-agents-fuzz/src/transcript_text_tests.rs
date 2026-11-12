//! Transcript text: what the user and the agent said, as the conversation and the room show it.

use lk_agents_abi::Role;
use lk_agents_fuzz::check::check_all;
use lk_agents_fuzz::drive::{RoomLegs, Run, Setup, TtsAlignment, simulate};
use lk_agents_fuzz::input::{AgentText, Input, Millis, Ref, SayInput, TurnMode, UserText};
use lk_agents_fuzz::record::{Recording, Sink};
use lk_agents_voice::events::SessionEvent;
use lk_agents_voice::trace::{PlayoutStep, Step};

/// Consecutive final transcripts join with a space, and streamed punctuation survives in the
/// agent's message and its shown transcript. https://github.com/livekit/agents/pull/566 https://github.com/livekit/agents/issues/587
#[test]
fn split_finals_and_streamed_punctuation_keep_their_boundaries() {
    let setup = Setup { stt: true, turn: TurnMode::Stt, ..Setup::default() };
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        for input in [
            Input::SttStart { stt: Ref::new(0) },
            Input::SttFinal {
                utterance: Ref::new(0),
                text: UserText("I need".into()),
                speaker: None,
            },
            Input::SttFinal {
                utterance: Ref::new(0),
                text: UserText("a table.".into()),
                speaker: None,
            },
            Input::SttEnd { utterance: Ref::new(0) },
            Input::Advance { ms: Millis(1_000) },
        ] {
            run.apply(&input).await;
        }
        for piece in ["Sure", ",", " one", " moment", "."] {
            let text = AgentText(piece.into());
            run.apply(&Input::LlmText { chat: Ref::new(0), text }).await;
        }
        run.apply(&Input::LlmEnd { chat: Ref::new(0) }).await;
        crate::playout::apply_with_playout(&mut run, &Input::Advance { ms: Millis(5_000) }).await;
        run
    });
    let events = run.record.host_view().events();
    let said = |role: Role| -> Vec<String> {
        events
            .iter()
            .filter_map(|(_, event)| match event {
                SessionEvent::ConversationItemAdded { item, .. } => {
                    item.as_message().filter(|m| m.role == role).and_then(|m| m.raw_text_content())
                }
                _ => None,
            })
            .collect()
    };
    assert_eq!(said(Role::User), ["I need a table."]);
    assert_eq!(said(Role::Assistant), ["Sure, one moment."]);
    let shown: String = run
        .record
        .sink()
        .iter()
        .filter_map(|(_, sink)| match sink {
            Sink::Text { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(shown.trim(), "Sure, one moment.");
}

const TWELVE: &str = "one two three four five six seven eight nine ten eleven twelve";

/// A synchronised session says `text`, played by a speaker that reports its progress or not, then
/// closes.
fn say_synchronised(text: &str, progress: bool) -> Recording {
    let setup = Setup { sync_transcription: true, ..Setup::default() };
    let run = simulate(async {
        let mut run = Run::start(&setup, 0, Default::default()).await;
        let text = AgentText(text.into());
        run.apply(&Input::Say { text, input: SayInput::Words, allow: None, to_ctx: true }).await;
        let advance = Input::Advance { ms: Millis(10_000) };
        if progress {
            crate::playout::apply_with_playout(&mut run, &advance).await;
        } else {
            crate::playout::apply_with_bare_playout(&mut run, &advance).await;
        }
        run.apply(&Input::Close { drain: true }).await;
        run.apply(&Input::Advance { ms: Millis(5_000) }).await;
        run
    });
    run.record
}

/// The words the transcription output received, each with when.
fn shown(record: &Recording) -> Vec<(u64, String)> {
    record
        .sink()
        .iter()
        .filter_map(|(at, sink)| match sink {
            Sink::Text { text, .. } => Some((at.at_ms, text.clone())),
            _ => None,
        })
        .collect()
}

/// A speaker that reports only its start and its finish (`tests/fake_io.FakeAudioOutput`, many
/// custom outputs) still shows the words as they are spoken, not all at once when it finishes.
#[test]
fn words_keep_pace_with_a_speaker_that_reports_no_progress() {
    for progress in [true, false] {
        let record = say_synchronised(TWELVE, progress);
        let when = |step: fn(&PlayoutStep) -> bool| {
            record.journal().iter().find_map(|(at, s)| match s {
                Step::Playout(s) if step(s) => Some(at.at_ms),
                _ => None,
            })
        };
        let started = when(|s| matches!(s, PlayoutStep::PlaybackStarted { .. }));
        let finished = when(|s| matches!(s, PlayoutStep::PlaybackFinished { .. }));
        let (Some(started), Some(finished)) = (started, finished) else {
            panic!("the scenario: the speech played (progress: {progress})")
        };
        assert!(
            finished >= started + 2_000,
            "the scenario: seconds of speech, {started}..{finished}"
        );
        let shown = shown(&record);
        let text: String = shown.iter().map(|(_, w)| w.as_str()).collect();
        assert_eq!(text.trim(), TWELVE, "progress: {progress}");
        let first = shown[0].0;
        assert!(
            first <= started + 500,
            "the first word {}ms in (progress: {progress})",
            first - started
        );
        // Over the playout, not in a burst at its end: at its middle, some words are out and
        // some are not.
        let middle = started + (finished - started) / 2;
        let by_middle: String =
            shown.iter().filter(|(at, _)| *at <= middle).map(|(_, w)| w.as_str()).collect();
        let count = by_middle.split_whitespace().count();
        assert!(
            (3..=9).contains(&count),
            "{count} of 12 words shown halfway through the playout (progress: {progress}): {shown:?}"
        );
        let (failures, _) = check_all(&record);
        assert!(failures.is_empty(), "progress: {progress}: {failures:#?}");
    }
}

/// The transcription output gets one flush per segment with text: none as the synchroniser
/// attaches it, and none as the session closes with nothing more said.
#[test]
fn a_synchronised_transcript_flushes_only_segments_with_text() {
    let record = say_synchronised("Hello there.", true);
    let calls: Vec<&str> = record
        .sink()
        .iter()
        .filter_map(|(_, sink)| match sink {
            Sink::Text { .. } => Some("text"),
            Sink::TextFlush { .. } => Some("flush"),
            _ => None,
        })
        .collect();
    let first_text = calls.iter().position(|c| *c == "text").expect("the scenario: words shown");
    assert!(!calls[..first_text].contains(&"flush"), "flushed before any text: {calls:?}");
    assert_eq!(calls.last(), Some(&"flush"), "the segment is closed: {calls:?}");
    assert_eq!(calls.iter().filter(|c| **c == "flush").count(), 1, "{calls:?}");
}

/// A reply that opens with an expression tag shows its words without the markup on the room's
/// agent transcript, whose header carries the expression as `lk.expression`: what the agents-ui
/// avatar reads (components-js `useAgentExpression`). The framework's `<expr/>` dialect and
/// Cartesia's `<emotion/>` alike, paced against the audio or not, and when the transcript is the
/// voice's own stamped words, which never carry the model's tag.
#[test]
fn a_leading_expression_reaches_the_room_transcript_as_lk_expression() {
    let replies: [&[&str]; 2] = [
        &["<expr type=\"expression\" ", "label=\"happy\"/>", " Great", " news!"],
        &["<emotion value=\"happy\"/>", "Great", " news!"],
    ];
    let cases = replies.into_iter().flat_map(|reply| {
        [(false, false), (true, false), (false, true), (true, true)]
            .map(|(sync, aligned)| (reply, sync, aligned))
    });
    for (reply, sync_transcription, aligned) in cases {
        let case = format!("{reply:?} sync={sync_transcription} aligned={aligned}");
        let setup = Setup {
            sync_transcription,
            room_legs: Some(RoomLegs { pre_connect: None }),
            tts_alignment: if aligned { TtsAlignment::Words } else { TtsAlignment::None },
            // only the voice's own tags are markup: `<emotion/>` is Cartesia's
            tts_markup: "cartesia",
            ..Setup::default()
        };
        let streams = simulate(async {
            let mut run = Run::start(&setup, 0, Default::default()).await;
            run.apply(&Input::Advance { ms: Millis(100) }).await;
            let reply_to =
                Input::GenerateReply { user_input: None, instructions: None, allow: None };
            run.apply(&reply_to).await;
            run.apply(&Input::Advance { ms: Millis(10) }).await;
            for piece in reply {
                run.apply(&Input::LlmText { chat: Ref::new(0), text: AgentText((*piece).into()) })
                    .await;
            }
            run.apply(&Input::LlmEnd { chat: Ref::new(0) }).await;
            run.apply(&Input::Advance { ms: Millis(5_000) }).await;
            run.room_text_streams()
        });

        let transcript = streams
            .iter()
            .find(|s| {
                s.opts.topic == lk_agents_abi::TOPIC_TRANSCRIPTION && s.text().contains("news")
            })
            .unwrap_or_else(|| panic!("the reply's transcript: {case}, {streams:#?}"));
        // The voice's stamps space the words their own way; no markup is shown either way.
        let shown: String = transcript.text().split_whitespace().collect();
        assert_eq!(shown, "Greatnews!", "{case}");
        let expression =
            transcript.opts.attributes.get(lk_agents_abi::ATTRIBUTE_TRANSCRIPTION_EXPRESSION);
        assert_eq!(
            expression.map(String::as_str),
            Some(r#"{"expression":"happy","mood":"happy"}"#),
            "{case}"
        );
    }
}
