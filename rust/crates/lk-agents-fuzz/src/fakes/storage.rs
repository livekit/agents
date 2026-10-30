//! The voice lane's storage at the port the voice executor consumes: every turn commits and every
//! abandon lands at once, until the lease is lost at the setup's `lost_at`-th of them. That one
//! reports the loss with its answer, every speech watching the lease hears it, and every later
//! acquire is refused, as a real store's lease stays lost. Storage's own lease, renewal and
//! fenced commit are its own tests' and the A2A DST's.
use std::sync::{Arc, Mutex};

use lk_agents_abi::storage::StorageError;
use lk_agents_abi::storage::voice::{
    StorageWork, VoiceStorage, VoiceStorageCommand, VoiceStorageEvent,
};
use lk_agents_abi::{ChatItem, StreamSink};

type Events = Arc<dyn StreamSink<VoiceStorageEvent>>;

pub struct FakeVoiceStorage {
    lost_at: u32,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    answered: u32,
    lost: bool,
    watching: Vec<(u64, Events)>,
}

impl FakeVoiceStorage {
    pub fn new(lost_at: u32) -> Arc<Self> {
        Arc::new(Self { lost_at, state: Mutex::default() })
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

    fn acquire(&self, _speech: u64, local: Vec<ChatItem>) -> StorageWork<'_, Vec<ChatItem>> {
        let lost = self.lease_lost();
        Box::pin(async move {
            if lost {
                return Err(StorageError::LeaseLost("the voice lane lease was lost".into()));
            }
            Ok(local)
        })
    }

    fn settle(&self, _speech: u64, _history: Vec<ChatItem>) -> StorageWork<'_, ()> {
        Box::pin(async { Ok(()) })
    }

    fn abort(&self, _speech: u64) -> StorageWork<'_, ()> {
        Box::pin(async { Ok(()) })
    }

    fn wait_lost(&self) -> StorageWork<'_, bool> {
        Box::pin(std::future::pending())
    }

    fn lease_lost(&self) -> bool {
        self.state.lock().unwrap().lost
    }
}
