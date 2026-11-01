//! Connection attempts and retry deadlines. The runtime only executes their effects.

use lk_agents_abi::errors::{ApiError, TimeoutPhase};
use lk_agents_abi::{ApiConnectOptions, Ts};
use lk_agents_core::attempts::{AttemptFailure, AttemptStep, Attempts, RequestPolicy};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq)]
pub enum ConnectionEffect {
    Connect { generation: u64 },
    Close { generation: u64 },
    Retry { error: ApiError },
    Failed { error: ApiError },
    CleanupIncomplete { generation: u64 },
}

/// One session owns its total deadline; successful acquisitions restart the consecutive
/// connection-failure budget. Every socket launch has a new generation, including retries.
#[derive(Debug)]
pub struct RealtimeConnection {
    policy: RequestPolicy,
    attempts: Option<Attempts>,
    retired: Vec<RetiredAcquisition>,
    total_deadline: Option<Ts>,
    generation: u64,
    attempt_generations: BTreeMap<u32, u64>,
    current_attempt: Option<u32>,
    connected: bool,
    closed: bool,
}

impl RealtimeConnection {
    pub fn new(options: &ApiConnectOptions, now: Ts) -> Self {
        let mut policy = RequestPolicy::from_options(options);
        let total_deadline = policy.total_timeout.map(|limit| now.saturating_add(limit));
        // Total belongs to this session, not to any one acquisition/retry cycle.
        policy.total_timeout = None;
        Self {
            policy,
            attempts: None,
            retired: Vec::new(),
            total_deadline,
            generation: 0,
            attempt_generations: BTreeMap::new(),
            current_attempt: None,
            connected: false,
            closed: false,
        }
    }

    pub fn start(&mut self, now: Ts) -> Vec<ConnectionEffect> {
        if self.closed || self.attempts.is_some() {
            return Vec::new();
        }
        if self.total_deadline.is_some_and(|deadline| now >= deadline) {
            return self.expire(now);
        }
        let mut steps = Vec::new();
        self.attempts = Some(Attempts::start(self.policy, now, &mut steps));
        self.translate(steps)
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The acquisition future completed and handed its socket to the runtime session.
    pub fn connected(&mut self, generation: u64, now: Ts) -> bool {
        if self.closed || generation != self.generation || self.current_attempt.is_none() {
            return false;
        }
        let attempt = self.current_attempt.unwrap();
        if let Some(attempts) = &mut self.attempts {
            if !attempts.succeeded(attempt, now) {
                return false;
            }
            attempts.closed(attempt, now, &mut Vec::new());
        }
        self.connected = true;
        true
    }

    /// The socket ended. A healthy connection has fulfilled its acquisition, so the next
    /// acquisition gets a fresh retry allowance; failed handshakes keep their shared budget.
    pub fn failed(&mut self, generation: u64, now: Ts, error: ApiError) -> Vec<ConnectionEffect> {
        if self.closed || generation != self.generation || self.current_attempt.is_none() {
            return Vec::new();
        }
        if self.total_deadline.is_some_and(|deadline| now >= deadline) {
            return self.expire(now);
        }
        if !error.retryable || self.policy.max_retry == 0 {
            let mut effects = self.close(now);
            effects.push(ConnectionEffect::Failed { error });
            return effects;
        }
        if self.connected {
            self.connected = false;
            self.current_attempt = None;
            if let Some(attempts) = self.attempts.take()
                && !attempts.is_settled()
            {
                self.retired.push(RetiredAcquisition {
                    attempts,
                    generations: std::mem::take(&mut self.attempt_generations),
                });
            }
            if self.retired.len() >= 32 {
                self.closed = true;
                return vec![ConnectionEffect::Failed {
                    error: ApiError::new("realtime acquisition cleanup capacity exceeded")
                        .with_retryable(false),
                }];
            }
            let mut effects = vec![ConnectionEffect::Retry { error }];
            // The established socket completed the first attempt of this retry window.
            let mut policy = self.policy;
            policy.max_retry = policy.max_retry.saturating_sub(1);
            let mut steps = Vec::new();
            self.attempts = Some(Attempts::start(policy, now, &mut steps));
            effects.extend(self.translate(steps));
            return effects;
        }
        let attempt = self.current_attempt.take().unwrap();
        let mut steps = Vec::new();
        if let Some(attempts) = &mut self.attempts {
            attempts.failed(
                attempt,
                AttemptFailure::Connection(error.message.clone()),
                now,
                &mut steps,
            );
            attempts.closed(attempt, now, &mut steps);
        }
        self.translate(steps)
            .into_iter()
            .map(|effect| match effect {
                ConnectionEffect::Retry { .. } => ConnectionEffect::Retry { error: error.clone() },
                ConnectionEffect::Failed { .. } => {
                    ConnectionEffect::Failed { error: error.clone() }
                }
                other => other,
            })
            .collect()
    }

    /// The runtime settled a stopped acquisition future, including one from an old window.
    pub fn released(&mut self, generation: u64, now: Ts) -> Vec<ConnectionEffect> {
        for retired in &mut self.retired {
            if let Some(attempt) = retired
                .generations
                .iter()
                .find_map(|(attempt, known)| (*known == generation).then_some(*attempt))
            {
                retired.attempts.closed(attempt, now, &mut Vec::new());
                self.retired.retain(|retired| !retired.attempts.is_settled());
                return Vec::new();
            }
        }
        let Some(attempt) = self
            .attempt_generations
            .iter()
            .find_map(|(attempt, known)| (*known == generation).then_some(*attempt))
        else {
            return Vec::new();
        };
        let mut steps = Vec::new();
        if let Some(attempts) = &mut self.attempts {
            attempts.closed(attempt, now, &mut steps);
        }
        self.translate(steps)
    }

    pub fn next_deadline(&self) -> Option<Ts> {
        self.attempts
            .as_ref()
            .and_then(|a| a.wake_at(None))
            .into_iter()
            .chain(self.retired.iter().filter_map(|retired| retired.attempts.wake_at(None)))
            .chain(self.total_deadline.filter(|_| !self.closed))
            .min()
    }

    pub fn tick(&mut self, now: Ts) -> Vec<ConnectionEffect> {
        if !self.closed && self.total_deadline.is_some_and(|deadline| now >= deadline) {
            return self.expire(now);
        }
        let mut effects = Vec::new();
        for retired in &mut self.retired {
            let mut steps = Vec::new();
            retired.attempts.wake(now, None, &mut steps);
            for step in steps {
                if let AttemptStep::CleanupIncomplete(attempt) = step {
                    effects.push(ConnectionEffect::CleanupIncomplete {
                        generation: retired.generations[&attempt],
                    });
                }
            }
        }
        self.retired.retain(|retired| !retired.attempts.is_settled());
        let mut steps = Vec::new();
        // A successful acquisition is settled but may still own cleanup of older attempts.
        if let Some(attempts) = &mut self.attempts {
            attempts.wake(now, None, &mut steps);
        }
        effects.extend(self.translate(steps));
        effects
    }

    pub fn close(&mut self, now: Ts) -> Vec<ConnectionEffect> {
        if self.closed {
            return Vec::new();
        }
        self.closed = true;
        let mut steps = Vec::new();
        if let Some(attempts) = &mut self.attempts {
            attempts.cancel(now, &mut steps);
        }
        let mut effects = self.translate(steps);
        if self.connected {
            effects.push(ConnectionEffect::Close { generation: self.generation });
        }
        self.connected = false;
        effects
    }

    fn expire(&mut self, now: Ts) -> Vec<ConnectionEffect> {
        let mut effects = self.close(now);
        effects.push(ConnectionEffect::Failed {
            error: ApiError::timed_out(TimeoutPhase::Total).with_retryable(false),
        });
        effects
    }

    fn translate(&mut self, steps: Vec<AttemptStep>) -> Vec<ConnectionEffect> {
        let mut effects = Vec::new();
        for step in steps {
            match step {
                AttemptStep::Launch(attempt) => {
                    self.generation += 1;
                    self.current_attempt = Some(attempt);
                    self.attempt_generations.insert(attempt, self.generation);
                    effects.push(ConnectionEffect::Connect { generation: self.generation });
                }
                AttemptStep::Stop(attempt) => effects.push(ConnectionEffect::Close {
                    generation: self.attempt_generations[&attempt],
                }),
                AttemptStep::Retry { error, .. } => effects.push(ConnectionEffect::Retry { error }),
                AttemptStep::GiveUp(error) => {
                    self.closed = true;
                    effects.push(ConnectionEffect::Failed { error });
                }
                AttemptStep::CleanupIncomplete(attempt) => {
                    effects.push(ConnectionEffect::CleanupIncomplete {
                        generation: self.attempt_generations[&attempt],
                    })
                }
                AttemptStep::Unresolved => unreachable!("realtime acquisitions are never retired"),
            }
        }
        effects
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lk_agents_abi::errors::ApiErrorKind;

    fn at(ms: u64) -> Ts {
        Ts::from_nanos(ms * 1_000_000)
    }
    fn options() -> ApiConnectOptions {
        ApiConnectOptions {
            connect_timeout: Some(1.0),
            total_timeout: None,
            max_retry: 1,
            ..Default::default()
        }
    }

    #[test]
    fn realtime_connection_generation_and_connect_timeout_keep_their_identity() {
        let mut connection = RealtimeConnection::new(&options(), at(0));
        assert_eq!(connection.start(at(0)), [ConnectionEffect::Connect { generation: 1 }]);
        let effects = connection.tick(at(1000));
        assert!(effects.contains(&ConnectionEffect::Close { generation: 1 }));
        assert!(effects.iter().any(|effect| matches!(effect, ConnectionEffect::Retry { error }
            if error.kind == ApiErrorKind::Timeout && error.phase == Some(TimeoutPhase::Connect))));
        assert_eq!(connection.next_deadline(), Some(at(1100)));
        connection.released(1, at(1000));
        assert_eq!(connection.tick(at(1100)), [ConnectionEffect::Connect { generation: 2 }]);
        assert!(!connection.connected(1, at(1100)));
        assert!(connection.connected(2, at(1100)));
        assert!(connection.failed(1, at(1200), ApiError::connection(None)).is_empty());
    }

    #[test]
    fn realtime_cancel_retains_cleanup_deadlines_and_the_original_generation() {
        let mut connection = RealtimeConnection::new(&options(), at(0));
        connection.start(at(0));
        connection.tick(at(1000));
        connection.tick(at(1100));
        connection.close(at(1200));
        assert_eq!(connection.next_deadline(), Some(at(6000)));
        let effects = connection.tick(at(6000));
        assert!(effects.contains(&ConnectionEffect::CleanupIncomplete { generation: 1 }));
        assert!(!effects.contains(&ConnectionEffect::CleanupIncomplete { generation: 2 }));
        connection.released(2, at(6100));
        assert!(connection.tick(at(6200)).is_empty());
    }

    #[test]
    fn realtime_total_deadline_survives_a_healthy_reconnection() {
        let options = ApiConnectOptions { total_timeout: Some(2.0), ..options() };
        let mut connection = RealtimeConnection::new(&options, at(0));
        connection.start(at(0));
        connection.connected(1, at(100));
        let effects = connection.failed(1, at(500), ApiError::connection(None));
        assert!(effects.contains(&ConnectionEffect::Connect { generation: 2 }));
        connection.connected(2, at(600));
        assert_eq!(connection.next_deadline(), Some(at(2000)));
        assert!(connection.tick(at(2000)).iter().any(|effect| matches!(effect,
            ConnectionEffect::Failed { error } if error.phase == Some(TimeoutPhase::Total))));
        assert!(connection.start(at(2100)).is_empty());
    }

    #[test]
    fn realtime_old_acquisition_cleanup_progresses_after_a_new_socket_connects() {
        let mut connection = RealtimeConnection::new(&options(), at(0));
        connection.start(at(0));
        connection.tick(at(1000));
        connection.tick(at(1100));
        assert!(connection.connected(2, at(1200)));
        assert_eq!(connection.next_deadline(), Some(at(6000)));
        assert!(
            connection
                .tick(at(6000))
                .contains(&ConnectionEffect::CleanupIncomplete { generation: 1 })
        );
        assert!(connection.next_deadline().is_none());
    }

    #[test]
    fn realtime_old_acquisition_cleanup_survives_another_reconnection_window() {
        let mut connection = RealtimeConnection::new(&options(), at(0));
        connection.start(at(0));
        connection.tick(at(1000));
        connection.tick(at(1100));
        connection.connected(2, at(1200));
        connection.failed(2, at(1300), ApiError::connection(None));
        connection.connected(3, at(1400));
        assert!(
            connection
                .tick(at(6000))
                .contains(&ConnectionEffect::CleanupIncomplete { generation: 1 })
        );
    }
}

#[derive(Debug)]
struct RetiredAcquisition {
    attempts: Attempts,
    generations: BTreeMap<u32, u64>,
}
