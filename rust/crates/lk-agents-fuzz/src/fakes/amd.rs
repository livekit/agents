//! The scripted AMD answerer's side of the reply gate's protocol (the AMD proposal, "DST":
//! "Scripted AMD"). Until the AMD machine exists, these are the answers a run's AMD sends: the
//! spec §10.1 rows a verdict stands for, and `Detach.pending` built as the machine will build it.

use std::time::Duration;

use lk_agents_voice::amd::{DetachCancel, Opening, ReplyDecision};

use crate::input::{AmdOpening, AmdVerdict};

/// `machine_silence_threshold`'s default (spec §4.1).
pub const SILENCE: Duration = Duration::from_millis(1500);

/// The stage instructions a staged reply carries: distinct words a rule can find in a request
/// and must never find in the record.
pub const STAGE: &str = "AMD stage: you reached a machine. Leave the callback number.";

/// What the opening says, or is generated from.
pub const OPENING_TEXT: &str = "Hi, this is the clinic calling about your appointment.";
pub const OPENING_INSTRUCTIONS: &str = "AMD opening: greet the callee and say who is calling.";

/// The decision a verdict sends for its turn.
pub fn decision(verdict: AmdVerdict) -> ReplyDecision {
    let staged = || ReplyDecision { instructions: Some(STAGE.into()), ..ReplyDecision::allow() };
    match verdict {
        AmdVerdict::Allow => ReplyDecision::allow(),
        AmdVerdict::Deny | AmdVerdict::Wait => ReplyDecision::deny(),
        AmdVerdict::Stage => staged(),
        AmdVerdict::Silence => ReplyDecision { silence: Some(SILENCE), ..staged() },
        AmdVerdict::Ivr => ReplyDecision { silence: Some(SILENCE), dtmf: true, ..staged() },
    }
}

/// `Detach.pending`, as the machine builds it (proposal "Detach", step 3): a `wait` category or
/// replies forbidden deny; anything else allows with no instructions, silence or DTMF.
pub fn pending(category: AmdVerdict, forbid: bool) -> ReplyDecision {
    if forbid || category == AmdVerdict::Wait {
        ReplyDecision::deny()
    } else {
        ReplyDecision::allow()
    }
}

pub fn cancel(forbid: bool) -> DetachCancel {
    if forbid { DetachCancel::ForbidReplies } else { DetachCancel::Keep }
}

pub fn opening(kind: AmdOpening) -> Opening {
    match kind {
        AmdOpening::Text => Opening::Text(OPENING_TEXT.into()),
        AmdOpening::Instructions => Opening::Instructions(OPENING_INSTRUCTIONS.into()),
    }
}

/// Production probes for the AMD rules: a real session with the scripted answerer's inputs, as
/// the authored stories in `tests/regressions/amd` drive it.
pub mod probe {
    use crate::drive::{Case, Setup};
    use crate::input::{AgentText, AmdOpening, AmdVerdict, Input, Millis, Ref, UserText};

    /// A detector and a recogniser on the call; the turn commits 500 ms after the detector's end
    /// of speech.
    pub fn call(inputs: Vec<Input>) -> Case {
        Case {
            setup: Setup { stt: true, vad: true, ..Setup::default() },
            inputs,
            ..Case::default()
        }
    }

    /// AMD attaches and starts listening, with this opening.
    pub fn attached(opening: Option<AmdOpening>) -> Vec<Input> {
        vec![Input::AmdAttach {}, Input::AmdActivate { opening }]
    }

    /// The person says `words` as the case's utterance `utterance`, for 300 ms, and the turn
    /// commits.
    pub fn says(utterance: u32, words: &str) -> Vec<Input> {
        let u = Ref::new(utterance);
        vec![
            Input::VadStart { vad: Ref::new(0), speech_ms: Millis(0) },
            Input::SttStart { stt: Ref::new(0) },
            Input::Advance { ms: Millis(300) },
            Input::SttFinal { utterance: u, text: UserText(words.into()), speaker: None },
            Input::VadEnd { vad: Ref::new(0), silence_ms: Millis(0) },
            Input::SttEnd { utterance: u },
            Input::Advance { ms: Millis(600) },
        ]
    }

    /// AMD's decision for the case's `turn`th gated turn.
    pub fn decide(turn: u32, verdict: AmdVerdict) -> Input {
        Input::AmdDecide { turn: Ref::new(turn), verdict }
    }

    /// The model answers each of the first `chats` requests with a line; a request already
    /// closed refuses it.
    pub fn answered(chats: u32) -> Vec<Input> {
        let mut inputs = vec![Input::Advance { ms: Millis(50) }];
        for chat in 0..chats {
            let text = AgentText("Hello, this is the clinic.".into());
            inputs.push(Input::LlmText { chat: Ref::new(chat), text });
            inputs.push(Input::LlmEnd { chat: Ref::new(chat) });
        }
        inputs
    }

    pub fn advance(ms: u64) -> Input {
        Input::Advance { ms: Millis(ms) }
    }

    /// One turn decided with `verdict`, answered, and the call carried past any hold.
    pub fn one_turn(words: &str, verdict: AmdVerdict) -> Case {
        let mut inputs = attached(None);
        inputs.extend(says(0, words));
        inputs.push(decide(0, verdict));
        inputs.extend(answered(3));
        inputs.push(advance(4_000));
        call(inputs)
    }
}
