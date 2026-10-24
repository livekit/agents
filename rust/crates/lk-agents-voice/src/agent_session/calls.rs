//! The caller's side of awaited commands: a call gets an id for its command and a typed receiver,
//! settled from the session's answer stream.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use tokio::sync::oneshot;

use super::{Answer, CallId, Rx};

type Settle = Box<dyn FnOnce(Answer) + Send>;

/// One session's open calls, shared by everyone who sends it commands.
#[derive(Clone, Default)]
pub struct Calls(Arc<Mutex<Table>>);

#[derive(Default)]
struct Table {
    next: u64,
    open: BTreeMap<CallId, Settle>,
    closed: bool,
}

impl Calls {
    /// `pick` takes the answer's payload; an answer of another kind, or none before the session
    /// ends, closes the receiver.
    pub fn open<T: Send + 'static>(
        &self,
        pick: fn(Answer) -> Option<T>,
    ) -> (CallId, oneshot::Receiver<T>) {
        let (tx, rx) = oneshot::channel();
        let mut table = self.lock();
        let call = CallId(table.next);
        table.next += 1;
        if !table.closed {
            let settle = move |answer| {
                if let Some(value) = pick(answer) {
                    let _ = tx.send(value);
                }
            };
            table.open.insert(call, Box::new(settle));
        }
        (call, rx)
    }

    /// A late or unknown call's answer is dropped.
    pub fn settle(&self, call: CallId, answer: Answer) {
        let settle = self.lock().open.remove(&call);
        if let Some(settle) = settle {
            settle(answer);
        }
    }

    pub fn close(&self) {
        let open = {
            let mut table = self.lock();
            table.closed = true;
            std::mem::take(&mut table.open)
        };
        drop(open);
    }

    /// Settles every answer until the session's stream ends, then closes what is left.
    pub async fn pump(self, mut answers: Rx<(CallId, Answer)>) {
        while let Some((call, answer)) = answers.recv().await {
            self.settle(call, answer);
        }
        self.close();
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Table> {
        self.0.lock().unwrap_or_else(|p| p.into_inner())
    }
}
