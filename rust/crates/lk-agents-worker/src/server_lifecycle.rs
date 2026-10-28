//! Server readiness, drain and cleanup policy. The driver retains every live resource.

use std::collections::BTreeSet;
use std::time::Duration;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ServerPhase {
    Idle,
    Starting,
    Running,
    Draining,
    Closing,
    Closed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ServerResource {
    Setup,
    Host,
    Worker,
    Load,
    Health,
    Metrics,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ServerInput {
    Start,
    ResourceStarted(ServerResource),
    StartupFinished(Result<(), String>),
    Drain,
    DrainFinished(Result<(), String>),
    Close,
    ResourceStopped { resource: ServerResource, result: Result<(), String> },
    Failed(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ServerEffect {
    Start,
    Started,
    Drain,
    Close,
    DrainFinished(Result<(), String>),
    CloseFinished(Result<(), String>),
    IncompleteCleanup { resources: Vec<ServerResource> },
}

#[derive(Debug)]
pub(crate) struct ServerLifecycle {
    phase: ServerPhase,
    resources: BTreeSet<ServerResource>,
    drain_requested: bool,
    drain_result: Option<Result<(), String>>,
    failure: Option<String>,
    cleanup_timeout: Duration,
    cleanup_deadline: Option<Duration>,
    close_reported: bool,
}

impl ServerLifecycle {
    pub(crate) fn new(cleanup_timeout: Duration) -> Self {
        Self {
            phase: ServerPhase::Idle,
            resources: BTreeSet::new(),
            drain_requested: false,
            drain_result: None,
            failure: None,
            cleanup_timeout,
            cleanup_deadline: None,
            close_reported: false,
        }
    }

    pub(crate) fn phase(&self) -> ServerPhase {
        self.phase
    }

    pub(crate) fn wake_at(&self) -> Option<Duration> {
        self.cleanup_deadline
    }

    pub(crate) fn handle(&mut self, event: ServerInput, now: Duration) -> Vec<ServerEffect> {
        let mut effects = self.tick(now);
        match event {
            ServerInput::Start if self.phase == ServerPhase::Idle => {
                self.phase = ServerPhase::Starting;
                effects.push(ServerEffect::Start);
            }
            ServerInput::ResourceStarted(resource) => {
                self.resources.insert(resource);
            }
            ServerInput::StartupFinished(Ok(())) if self.phase == ServerPhase::Starting => {
                self.phase = ServerPhase::Running;
                effects.push(ServerEffect::Started);
                if self.drain_requested {
                    self.phase = ServerPhase::Draining;
                    effects.push(ServerEffect::Drain);
                }
            }
            ServerInput::StartupFinished(Err(error)) | ServerInput::Failed(error) => {
                self.failure.get_or_insert(error);
                self.close(now, &mut effects);
            }
            ServerInput::Drain => {
                self.drain_requested = true;
                if self.phase == ServerPhase::Running {
                    self.phase = ServerPhase::Draining;
                    effects.push(ServerEffect::Drain);
                }
            }
            ServerInput::DrainFinished(result) if self.drain_result.is_none() => {
                self.drain_result = Some(result.clone());
                effects.push(ServerEffect::DrainFinished(result));
            }
            ServerInput::Close => self.close(now, &mut effects),
            ServerInput::ResourceStopped { resource, result } => {
                if self.resources.remove(&resource) {
                    if let Err(error) = result {
                        self.failure.get_or_insert(format!("{resource:?}: {error}"));
                    }
                    if self.resources.is_empty() && self.phase == ServerPhase::Closing {
                        self.phase = ServerPhase::Closed;
                        self.cleanup_deadline = None;
                        if !self.close_reported {
                            self.close_reported = true;
                            effects.push(ServerEffect::CloseFinished(self.result()));
                        }
                    }
                }
            }
            _ => {}
        }
        effects
    }

    pub(crate) fn tick(&mut self, now: Duration) -> Vec<ServerEffect> {
        if self.cleanup_deadline.is_some_and(|deadline| now >= deadline) {
            self.cleanup_deadline = None;
            self.close_reported = true;
            return vec![
                ServerEffect::IncompleteCleanup {
                    resources: self.resources.iter().copied().collect(),
                },
                ServerEffect::CloseFinished(self.result()),
            ];
        }
        Vec::new()
    }

    fn result(&self) -> Result<(), String> {
        self.failure.clone().map_or(Ok(()), Err)
    }

    fn close(&mut self, now: Duration, effects: &mut Vec<ServerEffect>) {
        if matches!(self.phase, ServerPhase::Closing | ServerPhase::Closed) {
            return;
        }
        self.phase = ServerPhase::Closing;
        if self.drain_requested && self.drain_result.is_none() {
            let result = Err("server closed during drain".into());
            self.drain_result = Some(result.clone());
            effects.push(ServerEffect::DrainFinished(result));
        }
        effects.push(ServerEffect::Close);
        if self.resources.is_empty() {
            self.phase = ServerPhase::Closed;
            self.close_reported = true;
            effects.push(ServerEffect::CloseFinished(self.result()));
        } else {
            self.cleanup_deadline = Some(now.saturating_add(self.cleanup_timeout));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn server_startup_lifecycle_never_announces_incomplete_startup() {
        let mut state = ServerLifecycle::new(Duration::from_secs(1));
        state.handle(ServerInput::Start, Duration::ZERO);
        state.handle(ServerInput::ResourceStarted(ServerResource::Host), Duration::ZERO);
        let effects = state.handle(
            ServerInput::StartupFinished(Err("pool warmup timed out".into())),
            Duration::ZERO,
        );
        assert!(!effects.contains(&ServerEffect::Started));
        assert!(effects.contains(&ServerEffect::Close));
        let effects = state.handle(
            ServerInput::ResourceStopped { resource: ServerResource::Host, result: Ok(()) },
            Duration::ZERO,
        );
        assert!(
            effects.contains(&ServerEffect::CloseFinished(Err("pool warmup timed out".into())))
        );
    }

    #[test]
    fn server_drain_close_callers_join_and_close_interrupts_drain() {
        let mut state = ServerLifecycle::new(Duration::from_secs(1));
        state.handle(ServerInput::Start, Duration::ZERO);
        state.handle(ServerInput::ResourceStarted(ServerResource::Host), Duration::ZERO);
        state.handle(ServerInput::StartupFinished(Ok(())), Duration::ZERO);
        assert_eq!(state.handle(ServerInput::Drain, Duration::ZERO), vec![ServerEffect::Drain]);
        assert!(state.handle(ServerInput::Drain, Duration::ZERO).is_empty());
        assert_eq!(
            state.handle(ServerInput::Close, Duration::ZERO),
            vec![
                ServerEffect::DrainFinished(Err("server closed during drain".into())),
                ServerEffect::Close,
            ]
        );
        assert!(state.handle(ServerInput::Close, Duration::ZERO).is_empty());
    }

    #[test]
    fn server_shutdown_resources_retains_ownership_past_cleanup_deadline() {
        let mut state = ServerLifecycle::new(Duration::from_secs(1));
        state.handle(ServerInput::Start, Duration::ZERO);
        state.handle(ServerInput::ResourceStarted(ServerResource::Host), Duration::ZERO);
        state.handle(ServerInput::ResourceStarted(ServerResource::Health), Duration::ZERO);
        state.handle(ServerInput::Close, Duration::ZERO);
        state.handle(
            ServerInput::ResourceStopped { resource: ServerResource::Health, result: Ok(()) },
            Duration::ZERO,
        );
        assert_eq!(
            state.tick(Duration::from_secs(1)),
            vec![
                ServerEffect::IncompleteCleanup { resources: vec![ServerResource::Host] },
                ServerEffect::CloseFinished(Ok(())),
            ]
        );
        assert_eq!(state.phase(), ServerPhase::Closing);
        assert!(state.tick(Duration::from_secs(2)).is_empty());
        state.handle(
            ServerInput::ResourceStopped { resource: ServerResource::Host, result: Ok(()) },
            Duration::from_secs(3),
        );
        assert_eq!(state.phase(), ServerPhase::Closed);
    }
}
