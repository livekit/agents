//! The transcript forwarded for a speech is the text the case gave it, in order; a speech that
//! completed normally forwarded all of it.
//!
//! The text is the case's own: the `Say` payload, the accepted `FeedText` chunks of the stream a
//! `say` opened, the audio transcript, or the LLM text accepted on the speech's chat requests
//! (attributed through `Detail::Chat`). Forwarded text is the journal's `TranscriptForwarded`.
//! Whitespace is ignored on both sides because the TTS leg re-segments text at sentences.
//!
//! A speech whose transcript path the case closed forwards nothing from then on, which is the
//! documented behaviour and not a loss. The sink is re-read for every segment, so a detach
//! part-way through a speech suppresses the rest of it, and the completion clause therefore
//! applies only to speeches whose path was live for their whole life. The prefix clause always
//! applies: it holds vacuously when nothing was forwarded.
use std::collections::BTreeMap;

use lk_agents_voice::{
    agent_session::IoLeg,
    trace::{PlayoutStep::TranscriptForwarded, Step::Playout},
};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule},
    input::{AgentText, Input, SayInput},
    record::{Answer, Detail, Handle, HostView, InferenceView, InputsView, Kind, Recording, SpeechView},
};

pub struct TextConservation;

impl Rule for TextConservation {
    const CONTRACT: Contract = Contract {
        id: "text_conservation",
        guarantee: "Forwarded transcript is a prefix of the text the case gave the speech; a speech that completed with a transcript path live throughout forwarded all of it.",
        examples: &[all_text_forwarded, completed_with_no_transcript_path, detached_mid_speech_forwards_nothing,],
        counterexamples: &[forwarded_unsent_text, completed_but_lost_text],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let inputs = r.inputs_view();
        let (host, internal, inference) = (
            r.host_view(),
            r.internal_view(),
            r.inference_view(),
        );
        let spoken = said_text(&inputs, &host);
        let transcription = inputs.transcription_output();

        for speech in host.speeches().into_iter().filter(SpeechView::is_known) {
            let Some(given) = spoken
                .get(&speech.id)
                .cloned()
                .or_else(|| generated_text(&inputs, &inference, speech.id))
            else {
                continue;
            };
            let forwarded: String =
                internal.speech(speech.id).transcript_forwarded().into_iter().map(squash).collect();

            // Nothing reaches the host that the case did not give it, and nothing is reordered:
            // what was forwarded so far is a prefix of what was given.
            check.expect(&speech, given.starts_with(&forwarded), format!("forwarded {forwarded:?} is not a prefix of given {given:?}"));

            // A speech that ran to the end without interruption or error forwarded everything, as
            // long as there was a transcript path to forward it down for the whole of its life.
            let until = speech.done_at().unwrap_or(r.now());
            let live_path =
                speech.created_at().is_none_or(|created| transcription.on_throughout(created, until));
            if live_path && speech.completed_normally() && speech.is_scheduled() {
                check.expect(&speech, forwarded == given, format!("completed speech forwarded {forwarded:?}, was given {given:?}"));
            }
        }
    }
}

/// The text the case gave each speech a `say` created: the words of the call, or what it fed
/// down the stream the same call opened.
fn said_text(inputs: &InputsView, host: &HostView) -> BTreeMap<u64, String> {
    let mut given = BTreeMap::new();
    for call in host.api_calls() {
        let Input::Say { text: AgentText(text), input: say, .. } = call.input() else {
            continue;
        };
        let Some((_, Answer::Speech(speech))) = call.returned() else { continue };
        given.insert(
            *speech,
            match say {
                // Audio is spoken from its transcript, which the call carried like any words.
                SayInput::Words | SayInput::Audio => squash(text),
                // Words that arrive later come down the stream the same `say` opened.
                SayInput::TextStream => {
                    call.stream().map(|stream| fed_text(inputs, stream)).unwrap_or_default()
                }
            },
        );
    }
    given
}

/// What the case pushed down one text stream, in order. A refused chunk never arrived.
fn fed_text(inputs: &InputsView, stream: u32) -> String {
    inputs
        .accepted()
        .filter_map(|sent| match &sent.input {
            Input::FeedText { stream: fed, text: AgentText(text) } if fed.nth == stream => {
                Some(squash(text))
            }
            _ => None,
        })
        .collect()
}

/// The LLM text accepted on the requests this speech was planned with. `None` when the speech
/// has no request of its own yet, because then its text is not the case's to know.
fn generated_text(inputs: &InputsView, inference: &InferenceView, speech: u64) -> Option<String> {
    let requests = inference.requests_for(speech);
    if requests.is_empty() {
        return None;
    }
    Some(
        inputs
            .accepted()
            .filter_map(|sent| match &sent.input {
                Input::LlmText { chat, text: AgentText(text) } if requests.contains(&chat.nth) => {
                    Some(squash(text))
                }
                _ => None,
            })
            .collect(),
    )
}

fn squash(text: &str) -> String {
    text.split_whitespace().collect()
}

/// The text the case gave each speech a `say` created.
///
/// A handle no longer says where its speech came from, so the link is re-made from the stamp.
/// Applying an input takes one stamp, and the call that `say` opened carries the same one, so
/// the two are paired by an equality rather than by counting. The call then names the speech
/// the driver created for it.
/// What the case pushed down one text stream, in order. A refused chunk never arrived.
/// The LLM text accepted on the chat requests this speech was planned with. `None` when the
/// speech has no request of its own yet, because then its text is not the case's to know.
/// Whether the transcript path was live for the whole of a speech's life.
///
/// Only the case can close it, in two ways, and both outputs are attached before the first
/// input, so the path starts live. A detach before the speech existed and one part-way through
/// it are the same thing to the completion clause: some of the speech had nowhere to go.
fn say(text: &str) -> Input {
    Input::Say {
        text: AgentText(text.into()),
        input: SayInput::Words,
        allow: None,
        to_ctx: true,
    }
}

/// A `say` of two words, answered with the speech it created, as a real run records it.
fn say_two_words(handle: Handle) -> Authored {
    Authored::new()
        .sent_opening(say("one two"), &[(Kind::Answer, Detail::None)])
        .result(0, Answer::Speech(1))
        .speech(1, handle)
}

fn forwarded(text: &str) -> lk_agents_voice::trace::Step {
    use lk_agents_voice::agent_activity::scheduling::SpeechId;
    Playout(TranscriptForwarded { speech: SpeechId(1), text: text.into() })
}

/// Every word given was forwarded, in order, and the speech completed.
fn all_text_forwarded() -> Recording {
    say_two_words(Handle::finished()).journal(forwarded("one")).journal(forwarded("two")).done()
}

/// The case turned the transcription leg off, so a completed speech forwards nothing.
fn completed_with_no_transcript_path() -> Recording {
    use lk_agents_voice::agent_activity::scheduling::SpeechKind;
    Authored::new()
        .sent(Input::SetIoEnabled { leg: IoLeg::Transcription, enabled: false })
        .sent_opening(say("one two"), &[(Kind::Answer, Detail::None)])
        .result(0, Answer::Speech(1))
        .speech_created(1, SpeechKind::Say)
        .speech(1, Handle::finished())
        .done()
}

/// The sink was attached when the speech began and detached before it spoke, so nothing was
/// forwarded and nothing was lost.
fn detached_mid_speech_forwards_nothing() -> Recording {
    use lk_agents_voice::agent_activity::scheduling::SpeechKind;
    Authored::new()
        .sent_opening(say("one two"), &[(Kind::Answer, Detail::None)])
        .result(0, Answer::Speech(1))
        .speech_created(1, SpeechKind::Say)
        .sent(Input::SetTranscriptionOutput { attach: false })
        .speech(1, Handle::finished())
        .done()
}

/// The host received a word nobody gave it.
fn forwarded_unsent_text() -> Recording {
    say_two_words(Handle::live()).journal(forwarded("three")).done()
}

/// The speech completed normally with half its text forwarded.
fn completed_but_lost_text() -> Recording {
    say_two_words(Handle::finished()).journal(forwarded("one")).done()
}
