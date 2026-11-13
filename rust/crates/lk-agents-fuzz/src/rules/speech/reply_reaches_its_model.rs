use std::collections::BTreeMap;

use lk_agents_voice::agent_session::machine::activity::scheduling::{SpeechId, SpeechKind};
use lk_agents_voice::events::{SessionEvent, SpeechOutcome};
use lk_agents_voice::trace::{GenerationStep, SpeechStep, Step};

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    record::{Detail, Id, Kind, Recording, StorageOperation, StorageStep},
};

/// How long a reply may go without its model request, besides the slow answers the fake
/// database held its partition acquire on.
pub const BOUND_MS: u64 = 15_000;

/// What the case owes the driver before a reply can be planned: an answer to the turn hook, or
/// to a tool whose output the reply reads.
const OWED: [Kind; 2] = [Kind::Hook, Kind::Tool];

// The reference starts a reply's inference at once and waits for the floor only to play.

pub struct ReplyReachesItsModel;

impl Rule for ReplyReachesItsModel {
    const CONTRACT: Contract = Contract {
        id: "reply_reaches_its_model",
        guarantee: "A requested reply makes its model request or settles within a bound of its creation, whatever is playing; only an unanswered hook or tool, the close, or slow storage answers its acquire waited on hold it.",
        examples: &[requested_during_playout, waits_out_slow_storage_answers],
        counterexamples: &[waits_for_the_playing_speech, waits_after_its_storage_answered],
        probe: None,
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        // A realtime model takes the reply on its own socket: no chat request to look for.
        if r.opened().iter().any(|o| o.kind == Kind::Realtime) {
            return;
        }
        let end = r.now().at_ms;
        let closing = r.close_begun().map(|c| c.at_ms);
        // Replies the model writes: a `say` speaks the host's own text, and a preemptive reply
        // waits on the turn it guessed.
        let replies: std::collections::BTreeSet<u64> = r
            .opened()
            .iter()
            .filter_map(|o| match (&o.id, &o.detail) {
                (
                    Id::Speech(id),
                    Detail::Speech {
                        kind: SpeechKind::TurnReply | SpeechKind::GenerateReply, ..
                    },
                ) => Some(id.0),
                _ => None,
            })
            .collect();
        let mut created: BTreeMap<u64, u64> = BTreeMap::new();
        let mut requested: BTreeMap<u64, u64> = BTreeMap::new();
        let mut done: BTreeMap<u64, u64> = BTreeMap::new();
        for (at, step) in r.journal() {
            match step {
                Step::Speech(SpeechStep::ReplyCreated { speech }) => {
                    created.entry(speech.0).or_insert(at.at_ms);
                }
                Step::Generation(GenerationStep::ChatRequested { speech, .. }) => {
                    requested.entry(speech.0).or_insert(at.at_ms);
                }
                Step::Emitted(SessionEvent::SpeechDone { speech, .. }) => {
                    done.entry(speech.0).or_insert(at.at_ms);
                }
                _ => {}
            }
        }
        for (&speech, &at) in created.iter().filter(|(speech, _)| replies.contains(speech)) {
            let deadline = deadline(r, speech, at);
            if end <= deadline || closing.is_some_and(|c| c <= deadline) {
                continue;
            }
            let reached = requested.get(&speech).is_some_and(|t| *t <= deadline)
                || done.get(&speech).is_some_and(|t| *t <= deadline);
            let owed = OWED.iter().any(|&kind| {
                r.opened().iter().filter(|o| o.kind == kind && o.at.at_ms <= deadline).any(|o| {
                    !r.closed()
                        .iter()
                        .any(|(c, k, n)| *k == kind && *n == o.nth && c.at_ms <= deadline)
                })
            });
            check.expect(
                Subject::Speech(speech),
                reached || owed,
                format!("created at {at} ms, no model request and not done {BOUND_MS} ms later"),
            );
        }
    }
}

/// `BOUND_MS` after `at`, extended by the time the fake database held slow answers while the
/// reply's partition acquire waited on them.
fn deadline(r: &Recording, speech: u64, at: u64) -> u64 {
    let acquire = |step: &StorageStep, answered: bool| match step {
        StorageStep::Requested { speech: s, operation: StorageOperation::Acquire } => {
            !answered && *s == speech
        }
        StorageStep::Answered { speech: s, operation: StorageOperation::Acquire, .. } => {
            answered && *s == speech
        }
        _ => false,
    };
    let storage = r.host_view().storage();
    let mut deadline = at + BOUND_MS;
    let Some(requested) =
        storage.iter().find(|(o, step)| o.at_ms >= at && acquire(step, false)).map(|(o, _)| o.at_ms)
    else {
        return deadline;
    };
    let answered = storage
        .iter()
        .find(|(o, step)| o.at_ms >= requested && acquire(step, true))
        .map_or(u64::MAX, |(o, _)| o.at_ms);
    let mut covered = requested;
    for (o, step) in storage {
        let StorageStep::Slowed { ms } = step else { continue };
        if o.at_ms < requested || o.at_ms > answered.min(deadline) {
            continue;
        }
        let (start, end) = (o.at_ms.max(covered), (o.at_ms + ms).min(answered));
        if end > start {
            deadline += end - start;
            covered = end;
        }
    }
    deadline
}

fn reply_created(speech: u64) -> Step {
    Step::Speech(SpeechStep::ReplyCreated { speech: SpeechId(speech) })
}

fn chat_requested(speech: u64) -> Step {
    Step::Generation(GenerationStep::ChatRequested {
        speech: SpeechId(speech),
        step: 0,
        ctx_hash: 0,
        tools: Vec::new(),
        instructions_hash: 0,
        ctx_ids: Vec::new(),
        tool_choice: None,
    })
}

/// A reply asked for while another plays requests its model at once; the call goes on.
fn requested_during_playout() -> Recording {
    Authored::new()
        .journal(Step::Emitted(SessionEvent::FloorChanged { holder: Some(SpeechId(1)) }))
        .speech_created(2, SpeechKind::GenerateReply)
        .journal(reply_created(2))
        .journal(chat_requested(2))
        .later(20_000)
        .journal(Step::Emitted(SessionEvent::SpeechDone {
            speech: SpeechId(2),
            outcome: SpeechOutcome::Completed,
        }))
        .done()
}

/// The reply waits for the playing speech's item: no request for twenty seconds of call.
fn waits_for_the_playing_speech() -> Recording {
    Authored::new()
        .journal(Step::Emitted(SessionEvent::FloorChanged { holder: Some(SpeechId(1)) }))
        .speech_created(2, SpeechKind::GenerateReply)
        .journal(reply_created(2))
        .later(20_000)
        .done()
}

fn acquire(speech: u64, answered: bool) -> StorageStep {
    let operation = StorageOperation::Acquire;
    if answered {
        StorageStep::Answered { speech, operation, result: Ok(()), delivered: true }
    } else {
        StorageStep::Requested { speech, operation }
    }
}

/// The reply's acquire meets two slow lease takes back to back: its request comes 15.5 s after
/// its creation, 2.5 s of it its own.
fn waits_out_slow_storage_answers() -> Recording {
    Authored::new()
        .speech_created(2, SpeechKind::GenerateReply)
        .journal(reply_created(2))
        .storage(acquire(2, false))
        .storage(StorageStep::Slowed { ms: 6_300 })
        .later(6_300)
        .storage(StorageStep::Slowed { ms: 6_700 })
        .later(9_200)
        .storage(acquire(2, true))
        .journal(chat_requested(2))
        .later(20_000)
        .done()
}

/// The acquire's slow answer is excused, but not the twenty seconds after it.
fn waits_after_its_storage_answered() -> Recording {
    Authored::new()
        .speech_created(2, SpeechKind::GenerateReply)
        .journal(reply_created(2))
        .storage(acquire(2, false))
        .storage(StorageStep::Slowed { ms: 6_000 })
        .later(6_000)
        .storage(acquire(2, true))
        .later(20_000)
        .done()
}
