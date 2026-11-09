use std::sync::{Arc, Mutex};

use super::Registry;
use crate::record::{StorageOperation, StorageStep};
use lk_agents_abi::storage::StorageError;
use lk_agents_abi::storage::voice::{
    StorageWork, VoiceStorage, VoiceStorageCommand, VoiceStorageEvent,
};
use lk_agents_abi::{ChatItem, ChatMessage, Role, StreamInput};
use lk_agents_abi::{Clock, SpawnerExt};
use std::time::Duration;
use std::time::UNIX_EPOCH;

// Once the `lost_at`-th receipt loses the lease it stays lost, as a real store's does: watchers
// hear it and every later acquire is refused. The real lease, renewal and fenced commit are
// covered by storage and runtime tests.

type Events = Arc<dyn StreamInput<VoiceStorageEvent>>;

pub struct FakeVoiceStorage {
    lost_at: u32,
    /// Every this many acquires, a text turn moved the lane before it.
    text_turn_every: u32,
    state: Mutex<State>,
    registry: Arc<Registry>,
    clock: Arc<dyn Clock>,
}

struct State {
    rng: fastrand::Rng,
    answered: u32,
    acquired: u32,
    lost: bool,
    watching: Vec<(u64, Events)>,
}

impl FakeVoiceStorage {
    pub fn new(
        lost_at: u32,
        seed: u64,
        registry: Arc<Registry>,
        clock: Arc<dyn Clock>,
    ) -> Arc<Self> {
        let mut rng = fastrand::Rng::with_seed(seed ^ 0x6c61_6e65);
        let text_turn_every = rng.u32(2..=6);
        let state = State { rng, answered: 0, acquired: 0, lost: false, watching: Vec::new() };
        Arc::new(Self { lost_at, text_turn_every, state: Mutex::new(state), registry, clock })
    }

    fn draw(state: &mut State) -> (Result<(), StorageError>, Duration) {
        // Stress frequencies, not measured failure rates.
        let result = match state.rng.u32(0..128) {
            0 => Err(StorageError::Unavailable("seeded storage outage".into())),
            1 => Err(StorageError::VersionConflict {
                lane: "voice".into(),
                expected: state.answered.into(),
            }),
            _ => Ok(()),
        };
        let ms = if state.rng.u32(0..32) == 0 {
            state.rng.u64(250..=1500)
        } else {
            state.rng.u64(0..=5)
        };
        (result, Duration::from_millis(ms))
    }

    fn deliver(&self, events: &Events, event: VoiceStorageEvent) {
        let (speech, operation, result) = match &event {
            VoiceStorageEvent::Settled { speech, result, .. } => {
                (*speech, StorageOperation::Settle, result.clone())
            }
            VoiceStorageEvent::Abandoned { speech, result } => {
                (*speech, StorageOperation::Abandon, result.clone())
            }
            VoiceStorageEvent::LeaseLost { speech } => {
                (*speech, StorageOperation::ObserveLoss, Ok(()))
            }
        };
        let delivered = events.push(event).is_ok();
        self.registry.record_storage(StorageStep::Answered {
            speech,
            operation,
            result,
            delivered,
        });
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
                    self.deliver(&events, VoiceStorageEvent::LeaseLost { speech });
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
        let (result, latency) = Self::draw(&mut state);
        drop(state);
        let answer = match command {
            VoiceStorageCommand::Settle { .. } => {
                VoiceStorageEvent::Settled { speech, result, lease_lost }
            }
            _ => VoiceStorageEvent::Abandoned { speech, result },
        };
        let operation = match command {
            VoiceStorageCommand::Settle { .. } => StorageOperation::Settle,
            _ => StorageOperation::Abandon,
        };
        self.registry.record_storage(StorageStep::Requested { speech, operation });
        for (speech, events) in watching {
            self.deliver(&events, VoiceStorageEvent::LeaseLost { speech });
        }
        let until = self.clock.now().saturating_add(latency);
        self.registry.storage_due(until);
        // The storage request outlives a cancelled caller, just as an issued commit does.
        let world = self.registry.world.clone();
        world.spawn_detached(async move {
            self.clock.wait_until(until).await;
            self.deliver(&events, answer);
        });
        Ok(())
    }

    fn acquire(&self, speech: u64, mut local: Vec<ChatItem>) -> StorageWork<'_, Vec<ChatItem>> {
        let mut state = self.state.lock().unwrap();
        let lost = state.lost;
        state.acquired += 1;
        let turn = state.acquired;
        let what = state.rng.u32(0..4);
        let (result, latency) = Self::draw(&mut state);
        drop(state);
        self.registry.record_storage(StorageStep::Requested {
            speech,
            operation: StorageOperation::Acquire,
        });
        let until = self.clock.now().saturating_add(latency);
        self.registry.storage_due(until);
        Box::pin(async move {
            self.clock.wait_until(until).await;
            let result = if lost {
                Err(StorageError::LeaseLost("the voice lane lease was lost".into()))
            } else {
                result
            };
            self.registry.record_storage(StorageStep::Answered {
                speech,
                operation: StorageOperation::Acquire,
                result: result.clone(),
                delivered: true,
            });
            result?;
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

/// A text turn between voice turns: edits an agent answer (`what` 0), compacts the two oldest
/// plain messages into a summary (1), or appends an exchange. Tool items are never rewritten.
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
    let at = lane.iter().map(ChatItem::created_at).max().unwrap_or(UNIX_EPOCH);
    lane.push(said(
        format!("lane_text_{turn}_user"),
        Role::User,
        "can you also text me the address?",
        at,
    ));
    lane.push(said(format!("lane_text_{turn}_agent"), Role::Assistant, "Sure, I've sent it.", at));
}
