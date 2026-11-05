//! Runs the IVR navigator over the session's event feed and supplied clock.

use crate::voice::session::PeerHandle;
use lk_agents_net::task::TaskSet;
use lk_agents_voice::agent_session::VoiceCommandOrigin;
use lk_agents_voice::ivr::IvrSystem;
use lk_agents_voice::ivr::effects::IvrEffect;
use std::sync::Arc;

/// The navigator runs among the session's tasks until it closes.
pub(super) fn start(peer: PeerHandle, enabled: bool, tasks: &Arc<TaskSet>) {
    let feed = peer.subscribe();
    let system = IvrSystem::new(peer.time.now(), enabled);
    let spawner = peer.spawner.clone();
    tasks.spawn(spawner.as_ref(), "ivr navigator", navigate(peer, feed, system));
}

/// Feeds the navigator the session's events and its own deadlines until it, the feed or the
/// session queue closes.
async fn navigate(
    peer: PeerHandle,
    mut feed: tokio::sync::mpsc::UnboundedReceiver<lk_agents_voice::agent_session::SystemTap>,
    mut system: IvrSystem,
) {
    perform(&peer, system.take_effects());
    while !system.closed() {
        let deadline = system.wake_at();
        if deadline.is_some_and(|at| at <= peer.time.now()) {
            perform(&peer, system.handle(peer.time.now(), None));
            continue;
        }
        let item = tokio::select! {
            biased;
            () = peer.time.wait_until_opt(deadline) => None,
            item = feed.recv() => Some(item),
            () = peer.queue.closed() => return,
        };
        let effects = match item {
            None => system.handle(peer.time.now(), None),
            Some(Some(item)) => system.handle(peer.time.now(), Some(item)),
            Some(None) => return,
        };
        perform(&peer, effects);
    }
}

fn perform(peer: &PeerHandle, effects: Vec<IvrEffect>) {
    for effect in effects {
        match effect {
            IvrEffect::Journal(step) => peer.journal(step),
            IvrEffect::Reply(reply) => {
                let _ = peer.send(VoiceCommandOrigin::Ivr, reply);
            }
        }
    }
}
