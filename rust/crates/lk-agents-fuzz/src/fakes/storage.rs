//! The voice lane's storage at the port the voice executor consumes: every turn commits and every
//! abandon lands at once, until the lease is lost at the setup's `lost_at`-th of them. That one
//! reports the loss with its answer, every speech watching the lease hears it, and every later
//! acquire is refused, as a real store's lease stays lost. Between voice turns a text turn takes
//! the lane now and then, at a cadence the seed draws: an acquire after it reloads what that turn
//! left, the person's text and the agent's answer appended, an earlier answer of the agent's
//! edited, or the oldest exchange compacted into a summary.
//! Storage's own lease, renewal and fenced commit are its own tests' and the A2A DST's.
use std::sync::{Arc, Mutex};

use lk_agents_abi::storage::StorageError;
use lk_agents_abi::storage::voice::{
    StorageWork, VoiceStorage, VoiceStorageCommand, VoiceStorageEvent,
};
use lk_agents_abi::{ChatItem, ChatMessage, Role, StreamSink};
use std::time::UNIX_EPOCH;

type Events = Arc<dyn StreamSink<VoiceStorageEvent>>;

pub struct FakeVoiceStorage {
    lost_at: u32,
    /// Every this many acquires, a text turn moved the lane before it.
    text_turn_every: u32,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    rng: Option<fastrand::Rng>,
    answered: u32,
    acquired: u32,
    lost: bool,
    watching: Vec<(u64, Events)>,
}

impl FakeVoiceStorage {
    pub fn new(lost_at: u32, seed: u64) -> Arc<Self> {
        let mut rng = fastrand::Rng::with_seed(seed ^ 0x6c61_6e65);
        let text_turn_every = rng.u32(2..=6);
        let state = State { rng: Some(rng), ..State::default() };
        Arc::new(Self { lost_at, text_turn_every, state: Mutex::new(state) })
    }
}

impl VoiceStorage for FakeVoiceStorage {
    fn dispatch(
        self: Arc<Self>,
        command: VoiceStorageCommand,
        events: Events,
    ) -> Result<(), StorageError> {
        let mut state = self.state.lock().unwrap();
        let speech = match command {
            VoiceStorageCommand::ObserveLoss { speech } => {
                if state.lost {
                    drop(state);
                    let _ = events.push(VoiceStorageEvent::LeaseLost { speech });
                } else {
                    state.watching.push((speech, events));
                }
                return Ok(());
            }
            VoiceStorageCommand::Settle { speech, .. } | VoiceStorageCommand::Abort { speech } => {
                speech
            }
        };
        state.answered += 1;
        let losing = !state.lost && state.answered == self.lost_at;
        state.lost |= losing;
        let watching = if losing { std::mem::take(&mut state.watching) } else { Vec::new() };
        let lease_lost = state.lost;
        drop(state);
        let answer = match command {
            VoiceStorageCommand::Settle { .. } => {
                VoiceStorageEvent::Settled { speech, result: Ok(()), lease_lost }
            }
            _ => VoiceStorageEvent::Abandoned { speech, result: Ok(()) },
        };
        let _ = events.push(answer);
        for (speech, events) in watching {
            let _ = events.push(VoiceStorageEvent::LeaseLost { speech });
        }
        Ok(())
    }

    fn acquire(&self, _speech: u64, mut local: Vec<ChatItem>) -> StorageWork<'_, Vec<ChatItem>> {
        let mut state = self.state.lock().unwrap();
        let lost = state.lost;
        state.acquired += 1;
        let turn = state.acquired;
        let what = state.rng.as_mut().expect("a seeded store").u32(0..4);
        drop(state);
        Box::pin(async move {
            if lost {
                return Err(StorageError::LeaseLost("the voice lane lease was lost".into()));
            }
            if turn.is_multiple_of(self.text_turn_every) {
                text_turn(&mut local, turn, what);
            }
            Ok(local)
        })
    }

    fn lease_lost(&self) -> bool {
        self.state.lock().unwrap().lost
    }
}

/// What a text turn leaves of the lane: an earlier answer of the agent's edited (`what` 0), the
/// two oldest plain messages compacted into a summary (1), or an exchange appended (otherwise).
/// Tool calls and their outputs are never rewritten: a text turn answers in messages.
fn text_turn(lane: &mut Vec<ChatItem>, turn: u32, what: u32) {
    let plain = |item: &ChatItem| {
        item.as_message().is_some_and(|m| matches!(m.role, Role::User | Role::Assistant))
    };
    let said =
        |id: String, role, text: &str, at| ChatItem::Message(ChatMessage::text(id, role, text, at));
    match what {
        0 => {
            let answer = lane
                .iter_mut()
                .find(|item| item.as_message().is_some_and(|m| m.role == Role::Assistant));
            if let Some(answer) = answer {
                *answer = said(
                    answer.id().to_owned(),
                    Role::Assistant,
                    "Sure, edited.",
                    answer.created_at(),
                );
                return;
            }
        }
        1 => {
            let oldest: Vec<usize> = lane
                .iter()
                .enumerate()
                .filter(|(_, item)| plain(item))
                .map(|(i, _)| i)
                .take(2)
                .collect();
            if let [first, second] = oldest[..] {
                let at = lane[first].created_at();
                lane.remove(second);
                lane[first] = said(
                    format!("lane_summary_{turn}"),
                    Role::Assistant,
                    "Summary of the start.",
                    at,
                );
                return;
            }
        }
        _ => {}
    }
    // Written after everything the call had: the lane's last item's instant.
    let at = lane.iter().map(ChatItem::created_at).max().unwrap_or(UNIX_EPOCH);
    lane.push(said(
        format!("lane_text_{turn}_user"),
        Role::User,
        "can you also text me the address?",
        at,
    ));
    lane.push(said(format!("lane_text_{turn}_agent"), Role::Assistant, "Sure, I've sent it.", at));
}
