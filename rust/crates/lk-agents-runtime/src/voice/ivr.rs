//! Runs the IVR navigator over the session's event feed and supplied clock.

use crate::voice::session::PeerHandle;
use lk_agents_net::task::TaskSet;
use lk_agents_voice::agent_session::Origin;
use lk_agents_voice::ivr::{IvrEffect, IvrSystem};
use std::sync::Arc;

/// The navigator runs among the session's tasks until it closes.
pub(super) fn start(peer: PeerHandle, enabled: bool, tasks: &Arc<TaskSet>) {
    let mut feed = peer.subscribe();
    let mut system = IvrSystem::new(peer.time.now(), enabled);
    let spawner = peer.spawner.clone();
    tasks.spawn(spawner.as_ref(), "ivr navigator", async move {
        perform(&peer, system.take_effects());
        while !system.closed() {
            let deadline = system.wake_at();
            let effects = if deadline.is_some_and(|at| at <= peer.time.now()) {
                system.handle(peer.time.now(), None)
            } else {
                tokio::select! {
                    biased;
                    () = peer.time.wait_until_opt(deadline) => system.handle(peer.time.now(), None),
                    item = feed.recv() => match item {
                        Some(item) => system.handle(peer.time.now(), Some(item)),
                        None => break,
                    },
                    () = peer.queue.closed() => break,
                }
            };
            perform(&peer, effects);
        }
    });
}

fn perform(peer: &PeerHandle, effects: Vec<IvrEffect>) {
    for effect in effects {
        match effect {
            IvrEffect::Journal(step) => peer.journal(step),
            IvrEffect::Reply(reply) => {
                let _ = peer.send(Origin::Ivr, reply);
            }
        }
    }
}
