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
