use lk_agents_abi::Ts;
use lk_agents_abi::storage::agentdb_error::ErrorCode;
use lk_agents_abi::storage::thread::Backoff;
use lk_agents_net::signal::Producer;

use super::{Conn, End, Socket, WsdbError};
use crate::telemetry::log_keys;

/// The task owning the live socket and the ready gate. However it ends (a give-up, a close, a
/// panic, its runtime dropped), it closes the connection, failing every request in flight, then
/// the gate ends, failing every request waiting on it.
pub(super) struct Supervisor {
    pub(super) conn: Conn,
    pub(super) ready: Producer<bool>,
}

impl Drop for Supervisor {
    fn drop(&mut self) {
        self.conn.close();
    }
}

impl Supervisor {
    /// Reads the live socket and, once it dies or retires, redials: for a database that moved,
    /// else while a request waits for it. A retiring socket finishes its reads meanwhile. Only a
    /// redial after a socket that lived its longest backoff step dials at once: one retired soon
    /// after its `Hello` waits the next step, so a server admitting sockets only to end them is
    /// not redialed in a loop.
    pub(super) async fn run(self, mut socket: Socket) {
        let (conn, clock) = (&self.conn, &*self.conn.0.clock);
        let (mut pace, mut redialed) = (conn.redial_backoff(), false);
        loop {
            let admitted = clock.now();
            let end = conn.read_loop(&mut socket, Some(&self.ready)).await;
            let lived = clock.now().saturating_duration_since(admitted);
            if lived >= conn.0.opts.reconnect.max_backoff {
                (redialed, pace) = (false, conn.redial_backoff());
            }
            if conn.is_closed() {
                return;
            }
            self.ready.set(false);
            let generation = socket.generation;
            let cause = match end {
                End::Retiring => {
                    let retiring = conn.unguarded().finish(socket);
                    let spawner = &*conn.0.spawner;
                    // Refused once the connection closed: its requests already failed.
                    conn.0.tasks.try_spawn(spawner, "agentdb retiring reader", retiring);
                    None
                }
                // Dropped here, its writer stops with it: a frame it never sent fails unsent
                // rather than wait out the redial. Only a retiring socket settles.
                End::Settled => {
                    drop(socket);
                    Some(WsdbError::Disconnected)
                }
                End::Dropped(cause) => {
                    drop(socket);
                    Some(cause)
                }
            };
            let now = match cause {
                None => true,
                Some(cause) => {
                    conn.ended(generation, Some(&cause));
                    cause.notice() == Some(ErrorCode::WRONG_SHARD_SERVER)
                }
            };
            match conn.redial(now, std::mem::replace(&mut redialed, true), &mut pace).await {
                Some(next) => socket = next,
                None => return,
            }
            self.ready.set(true);
        }
    }
}

impl Conn {
    /// The redials' backoff: the one retry policy over the reconnect bounds, jittered.
    fn redial_backoff(&self) -> Backoff {
        let r = &self.0.opts.reconnect;
        Backoff::jittered(r.min_backoff, r.max_backoff, rand::random())
    }

    /// Dials once a request waits for the socket (`now`: without waiting for one), each attempt
    /// bounded by the latest waiting deadline, after the next step of `pace` if the last socket
    /// was a redial's that ended soon (`pause`) and again after each failure; `None` once a
    /// refusal is final. Logged as a warning: no request hears why the socket is gone.
    async fn redial(&self, mut now: bool, mut pause: bool, pace: &mut Backoff) -> Option<Socket> {
        let clock = &*self.0.clock;
        let latest = |waiting: &mut Vec<Ts>| waiting.iter().max().copied();
        let (mut last, mut attempt) = (None::<WsdbError>, u32::from(pause));
        loop {
            if last.is_some() || std::mem::take(&mut pause) {
                let (delay, why) = (pace.delay(), last.take());
                let why: &dyn std::fmt::Display = match &why {
                    Some(e) => e,
                    None => &"the socket ended soon after its Hello",
                };
                let level = log::Level::Warn;
                lk_agents_abi::types::retry::retrying(level, "agentdb redial", attempt, delay, why);
                clock.wait_until(clock.now().saturating_add(delay)).await;
            }
            let deadline = match std::mem::take(&mut now) {
                true => self.0.waiting.with(latest),
                false => Some(self.0.waiting.wait_until(latest).await),
            };
            match self.connect(deadline).await {
                Ok(socket) => {
                    log::info!(target: lk_agents_abi::LOG_TARGET, (log_keys::AGENTDB_DATABASE_ID) = self.0.opts.database_id.as_str(); "agentdb reconnected");
                    return Some(socket);
                }
                Err(e) if e.is_final() => {
                    log::warn!(target: lk_agents_abi::LOG_TARGET, (log_keys::AGENTDB_DATABASE_ID) = self.0.opts.database_id.as_str(); "agentdb redial refused for good: {e}");
                    self.0.state.lock().fatal = Some(e);
                    return None;
                }
                Err(e) => (last, attempt) = (Some(e), attempt.saturating_add(1)),
            }
            if self.is_closed() {
                return None;
            }
        }
    }
}
