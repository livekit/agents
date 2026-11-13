use crate::effects::RealtimeConnectionEffect;
use lk_agents_abi::errors::{ApiError, Timeout};
use lk_agents_abi::{ApiConnectOptions, Ts};
use lk_agents_core::request::{Policy, Request, Step};
use std::collections::BTreeMap;

#[derive(Debug)]
pub struct RealtimeConnection {
    policy: Policy,
    phase: Phase,
    retired: Vec<Acquisition>,
    total_deadline: Option<Ts>,
    generation: u64,
}

#[derive(Debug)]
enum Phase {
    NotStarted,
    Running { acquisition: Acquisition, socket: Socket },
    Closed { cleanup: Option<Acquisition> },
}

#[derive(Debug)]
enum Socket {
    Waiting,
    Connecting { attempt: u32, generation: u64 },
    Connected { generation: u64 },
}

#[derive(Debug)]
struct Acquisition {
    attempts: Request,
    generations: BTreeMap<u32, u64>,
}

impl Phase {
    fn close(&mut self) -> Option<Socket> {
        let phase = std::mem::replace(self, Self::Closed { cleanup: None });
        match phase {
            Self::Running { acquisition, socket } => {
                *self = Self::Closed { cleanup: Some(acquisition) };
                Some(socket)
            }
            Self::Closed { cleanup } => {
                *self = Self::Closed { cleanup };
                None
            }
            Self::NotStarted => None,
        }
    }

    fn acquisition(&self) -> Option<&Acquisition> {
        match self {
            Self::Running { acquisition, .. } => Some(acquisition),
            Self::Closed { cleanup } => cleanup.as_ref(),
            Self::NotStarted => None,
        }
    }
    fn acquisition_mut(&mut self) -> Option<&mut Acquisition> {
        match self {
            Self::Running { acquisition, .. } => Some(acquisition),
            Self::Closed { cleanup } => cleanup.as_mut(),
            Self::NotStarted => None,
        }
    }
}

impl RealtimeConnection {
    pub fn new(options: &ApiConnectOptions, now: Ts) -> Self {
        let mut policy = Policy::from_options(options);
        let total_deadline = policy.total_timeout.map(|limit| now.saturating_add(limit));
        policy.total_timeout = None;
        Self {
            policy,
            phase: Phase::NotStarted,
            retired: Vec::new(),
            total_deadline,
            generation: 0,
        }
    }

    pub fn start(&mut self, now: Ts) -> Vec<RealtimeConnectionEffect> {
        if !matches!(self.phase, Phase::NotStarted) {
            return Vec::new();
        }
        if self.total_deadline.is_some_and(|deadline| now >= deadline) {
            return self.expire(now);
        }
        self.acquire(self.policy, now)
    }

    fn acquire(&mut self, policy: Policy, now: Ts) -> Vec<RealtimeConnectionEffect> {
        let mut steps = Vec::new();
        let attempts = Request::start(policy, now, &mut steps);
        self.phase = Phase::Running {
            acquisition: Acquisition { attempts, generations: BTreeMap::new() },
            socket: Socket::Waiting,
        };
        self.translate(steps)
    }

    pub fn connected(&mut self, generation: u64, now: Ts) -> bool {
        let Phase::Running { acquisition, socket: socket @ Socket::Connecting { .. } } =
            &mut self.phase
        else {
            return false;
        };
        let Socket::Connecting { attempt, generation: current } = *socket else { return false };
        if generation != current || !acquisition.attempts.succeeded(attempt, now) {
            return false;
        }
        acquisition.attempts.closed(attempt, now, &mut Vec::new());
        *socket = Socket::Connected { generation };
        true
    }

    pub fn failed(
        &mut self,
        generation: u64,
        now: Ts,
        error: ApiError,
    ) -> Vec<RealtimeConnectionEffect> {
        let Phase::Running { socket, .. } = &self.phase else { return Vec::new() };
        let current = match socket {
            Socket::Waiting => return Vec::new(),
            Socket::Connecting { generation, .. } | Socket::Connected { generation } => *generation,
        };
        if generation != current {
            return Vec::new();
        }
        if self.total_deadline.is_some_and(|deadline| now >= deadline) {
            return self.expire(now);
        }
        if !error.retryable || self.policy.max_retry == 0 {
            let mut effects = self.close(now);
            effects.push(RealtimeConnectionEffect::Failed { error });
            return effects;
        }
        if matches!(socket, Socket::Connected { .. }) {
            if let Phase::Running { acquisition, .. } =
                std::mem::replace(&mut self.phase, Phase::NotStarted)
                && !acquisition.attempts.is_settled()
            {
                self.retired.push(acquisition);
            }
            if self.retired.len() >= 32 {
                self.phase = Phase::Closed { cleanup: None };
                return vec![RealtimeConnectionEffect::Failed {
                    error: ApiError::new("realtime acquisition cleanup capacity exceeded")
                        .with_retryable(false),
                }];
            }
            let mut effects = vec![RealtimeConnectionEffect::Reconnect { error }];
            let mut policy = self.policy;
            policy.max_retry = policy.max_retry.saturating_sub(1);
            effects.extend(self.acquire(policy, now));
            return effects;
        }
        let mut steps = Vec::new();
        if let Phase::Running { acquisition, socket: socket @ Socket::Connecting { .. } } =
            &mut self.phase
        {
            if let Socket::Connecting { attempt, .. } = *socket {
                acquisition.attempts.failed_error(attempt, error.clone(), now, &mut steps);
                acquisition.attempts.closed(attempt, now, &mut steps);
            }
            *socket = Socket::Waiting;
        }
        self.translate(steps)
            .into_iter()
            .map(|effect| match effect {
                RealtimeConnectionEffect::Retry { backoff, .. } => {
                    RealtimeConnectionEffect::Retry { error: error.clone(), backoff }
                }
                RealtimeConnectionEffect::Failed { .. } => {
                    RealtimeConnectionEffect::Failed { error: error.clone() }
                }
                other => other,
            })
            .collect()
    }

    pub fn released(&mut self, generation: u64, now: Ts) -> Vec<RealtimeConnectionEffect> {
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
        let Some(acquisition) = self.phase.acquisition_mut() else { return Vec::new() };
        let Some(attempt) = acquisition
            .generations
            .iter()
            .find_map(|(attempt, known)| (*known == generation).then_some(*attempt))
        else {
            return Vec::new();
        };
        let mut steps = Vec::new();
        acquisition.attempts.closed(attempt, now, &mut steps);
        self.translate(steps)
    }

    pub fn next_deadline(&self) -> Option<Ts> {
        self.phase
            .acquisition()
            .and_then(|a| a.attempts.wake_at(None))
            .into_iter()
            .chain(self.retired.iter().filter_map(|a| a.attempts.wake_at(None)))
            .chain(self.total_deadline.filter(|_| !matches!(self.phase, Phase::Closed { .. })))
            .min()
    }

    pub fn tick(&mut self, now: Ts) -> Vec<RealtimeConnectionEffect> {
        if !matches!(self.phase, Phase::Closed { .. })
            && self.total_deadline.is_some_and(|deadline| now >= deadline)
        {
            return self.expire(now);
        }
        let mut effects = Vec::new();
        for retired in &mut self.retired {
            let mut steps = Vec::new();
            retired.attempts.wake(now, None, &mut steps);
            for step in steps {
                if let Step::CleanupIncomplete(attempt) = step {
                    effects.push(RealtimeConnectionEffect::CleanupIncomplete {
                        generation: retired.generations[&attempt],
                    });
                }
            }
        }
        self.retired.retain(|retired| !retired.attempts.is_settled());
        let mut steps = Vec::new();
        if let Some(acquisition) = self.phase.acquisition_mut() {
            acquisition.attempts.wake(now, None, &mut steps);
        }
        effects.extend(self.translate(steps));
        effects
    }

    pub fn close(&mut self, now: Ts) -> Vec<RealtimeConnectionEffect> {
        if matches!(self.phase, Phase::Closed { .. }) {
            return Vec::new();
        }
        let socket = self.phase.close();
        let mut steps = Vec::new();
        if let Some(acquisition) = self.phase.acquisition_mut() {
            acquisition.attempts.cancel(now, &mut steps);
        }
        let mut effects = self.translate(steps);
        if let Some(Socket::Connected { generation }) = socket {
            effects.push(RealtimeConnectionEffect::Close { generation });
        }
        effects
    }

    fn expire(&mut self, now: Ts) -> Vec<RealtimeConnectionEffect> {
        let mut effects = self.close(now);
        effects.push(RealtimeConnectionEffect::Failed {
            error: ApiError::timed_out(Timeout::Total).with_retryable(false),
        });
        effects
    }

    fn translate(&mut self, steps: Vec<Step>) -> Vec<RealtimeConnectionEffect> {
        let mut effects = Vec::new();
        for step in steps {
            match step {
                Step::Launch(attempt) => {
                    let Phase::Running { acquisition, socket } = &mut self.phase else { continue };
                    self.generation += 1;
                    *socket = Socket::Connecting { attempt, generation: self.generation };
                    acquisition.generations.insert(attempt, self.generation);
                    effects.push(RealtimeConnectionEffect::Connect { generation: self.generation });
                }
                Step::Stop(attempt) => {
                    let Some(acquisition) = self.phase.acquisition() else { continue };
                    effects.push(RealtimeConnectionEffect::Close {
                        generation: acquisition.generations[&attempt],
                    });
                    if let Phase::Running { socket, .. } = &mut self.phase
                        && matches!(socket, Socket::Connecting { attempt: current, .. } if *current == attempt)
                    {
                        *socket = Socket::Waiting;
                    }
                }
                Step::Retry(backoff) => effects.push(RealtimeConnectionEffect::Retry {
                    error: backoff.error().clone(),
                    backoff,
                }),
                Step::GiveUp(error) => {
                    let _socket = self.phase.close();
                    effects.push(RealtimeConnectionEffect::Failed { error });
                }
                Step::CleanupIncomplete(attempt) => {
                    let Some(acquisition) = self.phase.acquisition() else { continue };
                    effects.push(RealtimeConnectionEffect::CleanupIncomplete {
                        generation: acquisition.generations[&attempt],
                    });
                }
                Step::Unresolved => unreachable!("realtime acquisitions are never retired"),
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
        assert_eq!(connection.start(at(0)), [RealtimeConnectionEffect::Connect { generation: 1 }]);
        let effects = connection.tick(at(1000));
        assert!(effects.contains(&RealtimeConnectionEffect::Close { generation: 1 }));
        assert!(
            effects
                .iter()
                .any(|effect| matches!(effect, RealtimeConnectionEffect::Retry { error, .. }
            if error.kind == ApiErrorKind::Timeout && error.timeout == Some(Timeout::Connect)))
        );
        assert_eq!(connection.next_deadline(), Some(at(1100)));
        connection.released(1, at(1000));
        assert_eq!(
            connection.tick(at(1100)),
            [RealtimeConnectionEffect::Connect { generation: 2 }]
        );
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
        assert!(effects.contains(&RealtimeConnectionEffect::CleanupIncomplete { generation: 1 }));
        assert!(!effects.contains(&RealtimeConnectionEffect::CleanupIncomplete { generation: 2 }));
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
        assert!(effects.contains(&RealtimeConnectionEffect::Connect { generation: 2 }));
        connection.connected(2, at(600));
        assert_eq!(connection.next_deadline(), Some(at(2000)));
        assert!(connection.tick(at(2000)).iter().any(|effect| matches!(effect,
            RealtimeConnectionEffect::Failed { error } if error.timeout == Some(Timeout::Total))));
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
                .contains(&RealtimeConnectionEffect::CleanupIncomplete { generation: 1 })
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
                .contains(&RealtimeConnectionEffect::CleanupIncomplete { generation: 1 })
        );
    }

    #[test]
    fn stopped_and_closed_sockets_cannot_reenter_the_connected_phase() {
        let mut connection = RealtimeConnection::new(&options(), at(0));
        assert!(!connection.connected(0, at(0)));
        connection.start(at(0));
        connection.tick(at(1000));
        assert!(!connection.connected(1, at(1000)));
        assert!(
            connection
                .failed(1, at(1000), ApiError::connection(None).with_retryable(false))
                .is_empty(),
            "a stopped acquisition's late error cannot end the retry window",
        );
        connection.released(1, at(1000));
        connection.tick(at(1100));
        assert!(connection.connected(2, at(1100)));
        assert!(!connection.connected(2, at(1100)), "an acquisition completes once");
        assert_eq!(connection.close(at(1200)), [RealtimeConnectionEffect::Close { generation: 2 }]);
        assert!(connection.close(at(1200)).is_empty());
        assert!(!connection.connected(2, at(1300)));
        assert!(connection.failed(2, at(1300), ApiError::connection(None)).is_empty());
        assert!(connection.start(at(1300)).is_empty());
    }
}
