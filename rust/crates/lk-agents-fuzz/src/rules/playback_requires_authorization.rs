//! A speculative reply's audio does not reach the sink before that reply is authorized.
//!
//! This is the playback-enforcement half only. A sink push is attributed to a speech by the
//! journal's `SinkCalled { speech }`, stamped by the same counter as `Authorized`, so the order
//! is a real one. Whether the authorization itself was permitted is a separate contract that
//! needs the turn-policy predicate the strategy still leaves open. Restricted to speculative
//! speeches (the driver's `Preemptive` kind, started from a preflight transcript before the turn
//! ended); every other kind is authorized at creation and would pass trivially.
use lk_agents_voice::{
    agent_activity::scheduling::{SpeechId, SpeechKind},
    trace::{
        PlayoutStep::SinkCalled,
        SinkCall::Push,
        SpeechStep::Authorized,
        Step::{self, Playout, Speech},
    },
};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule},
    drive::{Case, Setup},
    input::{AgentText, Count, Input, Millis, Ref, TurnMode, UserText},
    record::{Recording, SpeechView},
};

pub struct PlaybackRequiresAuthorization;

impl Rule for PlaybackRequiresAuthorization {
    const CONTRACT: Contract = Contract {
        id: "playback_requires_authorization",
        guarantee: "No sink push attributed to a speculative speech precedes that speech's Authorized step.",
        examples: &[authorized_then_pushed],
        counterexamples: &[pushed_before_authorized, pushed_never_authorized],
        probe: Some(preemptive_guess),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let host = r.host_view();
        let internal = r.internal_view();
        for speech in host.speeches().into_iter().filter(SpeechView::is_speculative) {
            let trace = internal.speech(speech.id);
            let authorized = trace.authorized_at();

            // A speculative speech that has pushed nothing has nothing to enforce yet, and saying so
            // with an expectation that cannot fail would report this contract as exercised when it
            // was not. The expectation count is what tells a report the difference.
            let Some(push) = trace.first_sink_push_at() else { continue };

            // Its first audio reaches the sink only after the turn authorized this very speech.
            check.expect(
                &speech,
                authorized.is_some_and(|a| a < push),
                format!(
                    "speculative speech pushed audio at seq {} but was authorized at {authorized:?}",
                    push.seq
                ),
            );
        }
    }
}

/// The speculative speech was authorized, then its audio reached the sink.
fn authorized_then_pushed() -> Recording {
    speculative().journal(authorized()).journal(push()).done()
}

/// One speculative speech the host was told about.
fn speculative() -> Authored {
    Authored::new().speech_created(5, SpeechKind::Preemptive)
}

fn push() -> Step {
    Playout(SinkCalled { speech: Some(SpeechId(5)), call: Push { frames: 1 } })
}

fn authorized() -> Step {
    Speech(Authorized { speech: SpeechId(5) })
}

/// The speculative speech's audio reached the sink, then it was authorized.
fn pushed_before_authorized() -> Recording {
    speculative().journal(push()).journal(authorized()).done()
}

/// The speculative speech's audio reached the sink and it was never authorized.
fn pushed_never_authorized() -> Recording {
    speculative().journal(push()).done()
}

/// Prerequisite: a preflight transcript starts a guess whose audio is produced while the user
/// turn is still open, then the turn commits with a changed transcript.
///
/// The references count from the run's own history -- the first recognition stream, the utterance
/// the start of speech opens on it, the chat request the guess plans -- so the sequence resolves
/// the same way on replay and a reference that no longer exists is reported as drift.
/// Reach a speculative reply: recognise a preflight transcript, let the driver start a guess on
/// it, and only then end the turn.
///
/// The setup is what makes this reachable at all. Speculative generation is switched on when the
/// session is built, so no sequence of inputs can turn it on afterwards, and without it the
/// driver has no reason to speak before the turn ends.
fn preemptive_guess() -> Case {
    let stt = Ref::new(0);
    let utterance = Ref::new(0);
    let chat = Ref::new(0);
    let inputs = vec![
        Input::SttStart { stt },
        // The person says something: on a run with a microphone, a final transcript must follow
        // audio that reached the recogniser.
        Input::MicFrames { mic: Ref::new(0), chunks: Count(2) },
        Input::SttPreflight { utterance, text: UserText("hello".into()), speaker: None },
        Input::Advance { ms: Millis(10) },
        Input::LlmText { chat, text: AgentText("Sure, hello.".into()) },
        Input::LlmText { chat, text: AgentText(" How can I help?".into()) },
        Input::LlmEnd { chat },
        Input::Advance { ms: Millis(100) },
        Input::SttFinal { utterance, text: UserText("hello".into()), speaker: None },
        Input::SttEnd { utterance },
        // The guarantee is about audio reaching the sink, so the probe has to get there: let the
        // synthesis the guess asked for run and be pushed. Stopping at the guess would reach the
        // state the rule is about without ever testing what it guarantees.
        Input::Advance { ms: Millis(200) },
        Input::Advance { ms: Millis(500) },
    ];
    Case {
        setup: Setup { stt: true, turn: TurnMode::Stt, preemptive: true, ..Setup::default() },
        inputs,
        ..Case::default()
    }
}
