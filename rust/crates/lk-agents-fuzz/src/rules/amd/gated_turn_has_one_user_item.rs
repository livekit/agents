use lk_agents_voice::agent::Source;
use lk_agents_voice::amd::{AmdObservation, TurnEnd};
use lk_agents_voice::audio_recognition::TurnId;
use lk_agents_voice::events::SessionEvent;

use crate::{
    authored::Authored,
    check::{Check, Contract, Rule, Subject},
    input::{AmdVerdict, Input, Ref},
    record::{Recording, RtSent},
};

// Spec §7.2 and the proposal (Realtime): a gated realtime turn with words has exactly one user
// item. When the socket's model transcribes, the server's transcript of the committed audio is
// the item, so the session adds none of its own. Otherwise the session's STT words are the item
// when it has STT, and AMD's `InsertUserItem` when it does not. The server's own item is not
// counted here: its transcription runs on the server's clock and may fail on its own. Judged
// once the turn ended, from what AMD sent and the fake server's commits, not from the session's
// account of whether it needed the words. No turn has an empty user item.

pub struct GatedTurnHasOneUserItem;

impl Rule for GatedTurnHasOneUserItem {
    const CONTRACT: Contract = Contract {
        id: "gated_turn_has_one_user_item",
        guarantee: "A realtime turn committed under the AMD gate has exactly one user item: the \
                    server's transcript when the model transcribes, else the session's STT words, \
                    else AMD's transcript; and none is empty.",
        examples: &[transcribed_by_the_server, kept_from_session_stt, inserted_by_amd],
        counterexamples: &[
            stt_words_beside_the_server_transcript,
            no_words_from_amd,
            an_empty_item_before_amd_words,
        ],
        probe: Some(a_wait_then_a_person),
    };

    fn on_complete(r: &Recording, check: &mut Check) {
        let amd = r.amd_view();
        if !amd.realtime() {
            return;
        }
        let events = r.host_view().events();
        for turn in amd.gated_turns() {
            // `StopResponse` ignores the turn, and a turn dropped behind an uninterruptible speech
            // keeps no words, as in Python.
            if amd.stopped(&turn) || amd.dropped_unanswered(&turn) {
                continue;
            }
            let ended = amd.observed(turn.run).into_iter().any(|(_, o)| {
                matches!(o, AmdObservation::TurnEnded { turn: t, how }
                    if t.0 == turn.turn && *how != TurnEnd::Stopped)
            });
            let Some(socket) = amd.audio_committed(&turn).filter(|_| ended) else { continue };
            let items: Vec<(String, String)> = (events.iter())
                .filter_map(|(_, e)| match e {
                    SessionEvent::ConversationItemAdded {
                        item,
                        source: Source::UserTurn { turn: TurnId(t) },
                        ..
                    } if *t == turn.turn => {
                        let m = item.as_message()?;
                        Some((m.id.clone(), m.raw_text_content().unwrap_or_default()))
                    }
                    _ => None,
                })
                .collect();
            let mut ids: Vec<&String> = items.iter().map(|(id, _)| id).collect();
            ids.sort();
            ids.dedup();
            let empty = items.iter().any(|(_, text)| text.trim().is_empty());
            check.expect_with(Subject::Session, !empty, || {
                format!("turn {} has an empty user item: {items:?}", turn.turn)
            });
            let (expected, from) = if amd.rt_transcribes(socket) {
                (0, "the server transcribes it, so the session adds none")
            } else if !turn.words.trim().is_empty() {
                (1, "the session's STT words")
            } else if amd.amd_sent_words(&turn) {
                (1, "AMD's transcript")
            } else {
                (0, "nothing transcribed it")
            };
            check.expect_with(Subject::Session, ids.len() == expected, || {
                format!(
                    "turn {} ({:?}) has {} user items of its own, expected {expected}: {from} \
                     ({items:?})",
                    turn.turn,
                    turn.words,
                    ids.len()
                )
            });
        }
    }
}

const WORDS: &str = "please state your name";

/// A realtime socket whose model does or does not transcribe, and a turn committed under the
/// gate with `words` from the session's STT, its audio committed and its decision a denial.
fn turn(transcribes: bool, words: &str) -> Authored {
    Authored::new()
        .rt_sent(RtSent::Opened { transcribes })
        .amd_entered(1)
        .sent(Input::AmdAttach {})
        .user_turn_unsaved(Some(1), 1, words)
        .rt_sent(RtSent::CommitAudio)
        .sent(Input::AmdDecide { turn: Ref::new(0), verdict: AmdVerdict::Deny })
}

fn ended(a: Authored) -> Recording {
    a.amd(1, AmdObservation::TurnEnded { turn: TurnId(1), how: TurnEnd::Denied }).done()
}

fn insert(a: Authored) -> Authored {
    use lk_agents_voice::amd::{AmdAnswer, AmdRunId};
    a.amd_answer(AmdAnswer::InsertUserItem { run: AmdRunId(1), turn: TurnId(1), text: WORDS.into() })
}

fn transcribed_by_the_server() -> Recording {
    ended(turn(true, WORDS))
}

fn kept_from_session_stt() -> Recording {
    ended(turn(false, WORDS).user_item(1, WORDS))
}

fn inserted_by_amd() -> Recording {
    ended(insert(turn(false, "")).user_item(1, WORDS))
}

/// The superseded-turn bug: the session STT's words beside the server's transcript.
fn stt_words_beside_the_server_transcript() -> Recording {
    ended(turn(true, WORDS).user_item(1, WORDS))
}

/// AMD sent the words for a model that does not transcribe, and the session dropped them.
fn no_words_from_amd() -> Recording {
    ended(insert(turn(false, "")))
}

/// An empty item for the turn, which AMD's later insertion replaces under the same id.
fn an_empty_item_before_amd_words() -> Recording {
    ended(insert(turn(false, "").user_item(1, "")).user_item(1, WORDS))
}

/// A `wait` turn, then a person, on a model that transcribes: the session adds no item of its
/// own for either.
fn a_wait_then_a_person() -> crate::drive::Case {
    use crate::fakes::amd::probe::{advance, attached, decide, says};
    let mut inputs = attached(None);
    inputs.extend(says(0, "please hold"));
    inputs.push(decide(0, AmdVerdict::Wait));
    inputs.push(advance(1_000));
    inputs.extend(says(1, "hello who is this"));
    inputs.push(decide(1, AmdVerdict::Allow));
    inputs.push(advance(4_000));
    crate::fakes::amd::probe::realtime_call(inputs)
}
