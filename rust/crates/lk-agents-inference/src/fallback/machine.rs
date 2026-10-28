//! Request-local fallback decisions and the single plugin-instance health authority.
//! Adapters execute the effects and report observations; neither owns retry policy.

use std::collections::BTreeMap;
use std::time::Duration;

use lk_agents_abi::Ts;
use lk_agents_abi::errors::{ApiError, ApiErrorKind, ProviderError, TimeoutPhase};
use lk_agents_abi::resources::ResourceId;

use super::policy::ChainPolicy;
use crate::attempts::{AttemptStep, Attempts, RequestPolicy};

#[derive(Debug, Default)]
pub struct FallbackHealth {
    entries: BTreeMap<ResourceId, Health>,
}

#[derive(Debug)]
struct Health {
    available: bool,
    generation: u64,
    selected: u64,
}

impl FallbackHealth {
    pub fn is_available(&self, backend: ResourceId) -> bool {
        self.entries.get(&backend).is_none_or(|entry| entry.available)
    }

    /// Every observation supersedes probes authorized before it, even a repeated failure.
    pub fn observe(&mut self, backend: ResourceId, available: bool) -> bool {
        let entry = self.entries.entry(backend).or_insert(Health {
            available: true,
            generation: 0,
            selected: 0,
        });
        entry.generation += 1;
        let changed = entry.available != available;
        entry.available = available;
        changed
    }

    pub fn select(&mut self, backend: ResourceId, selection: u64) {
        self.entries
            .entry(backend)
            .or_insert(Health { available: true, generation: 0, selected: 0 })
            .selected = selection;
    }

    pub fn order(&self, backends: &[ResourceId], sticky: bool) -> Vec<usize> {
        let mut order: Vec<_> = (0..backends.len()).collect();
        if sticky && !order.is_empty() {
            let selected = |index: usize| {
                self.entries
                    .get(&backends[index])
                    .filter(|entry| entry.available)
                    .map_or(0, |entry| entry.selected)
            };
            let first = order
                .iter()
                .copied()
                .fold(0, |best, index| if selected(index) > selected(best) { index } else { best });
            order.remove(first);
            order.insert(0, first);
        }
        order
    }

    pub fn probe_generation(&self, backend: ResourceId) -> u64 {
        self.entries.get(&backend).map_or(0, |entry| entry.generation)
    }

    pub fn recovered(&mut self, backend: ResourceId, generation: u64) -> bool {
        if self.probe_generation(backend) != generation {
            return false;
        }
        self.observe(backend, true)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Modality {
    Llm,
    Stt,
    Tts,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainEffect {
    LlmRequestStart { attempt: u64, index: usize, llm: ResourceId },
    SttStreamOpen { attempt: u64, index: usize, stt: ResourceId },
    TtsSynthesisStart { attempt: u64, index: usize, tts: ResourceId },
    Cancel { attempt: u64 },
    WakeAt(Ts),
    Finished(Result<(), ProviderError>),
}

#[derive(Debug)]
pub enum ChainEvent {
    Wake,
    Succeeded { attempt: u64 },
    Failed { attempt: u64, error: ProviderError, produced_output: bool },
    Cancel,
}

#[derive(Debug)]
enum Phase {
    Ready,
    Active(u64),
    Waiting,
    Finished,
}

#[derive(Debug)]
pub struct ChainRequest {
    modality: Modality,
    entries: Vec<(usize, ResourceId, String)>,
    cursor: usize,
    attempts: Option<Attempts>,
    next_attempt: u64,
    phase: Phase,
    policy: ChainPolicy,
    deadline: Option<Ts>,
    /// Every failure is retained, including same-host diagnostic ownership.
    failures: Vec<ProviderError>,
}

impl ChainRequest {
    pub fn new(
        modality: Modality,
        mut entries: Vec<(usize, ResourceId, String)>,
        policy: ChainPolicy,
        deadline: Option<Ts>,
        health: &FallbackHealth,
    ) -> Self {
        let all_unavailable = entries.iter().all(|(_, id, _)| !health.is_available(*id));
        if !all_unavailable {
            entries.retain(|(_, id, _)| health.is_available(*id));
        }
        Self {
            modality,
            entries,
            cursor: 0,
            attempts: None,
            next_attempt: 0,
            phase: Phase::Ready,
            policy,
            deadline,
            failures: Vec::new(),
        }
    }

    pub fn deadline(&self) -> Option<Ts> {
        self.deadline
    }

    pub fn handle(&mut self, event: ChainEvent, now: Ts) -> Vec<ChainEffect> {
        if matches!(self.phase, Phase::Finished) {
            return Vec::new();
        }
        if matches!(event, ChainEvent::Cancel) {
            let mut effects = self.cancel_active(now);
            self.phase = Phase::Finished;
            effects.push(ChainEffect::Finished(Err(ApiError::status(
                "fallback cancelled",
                499,
                None,
                None,
            )
            .into())));
            return effects;
        }
        if self.deadline.is_some_and(|deadline| now >= deadline) {
            let mut effects = self.cancel_active(now);
            self.phase = Phase::Finished;
            effects.push(ChainEffect::Finished(Err(ApiError::timed_out(TimeoutPhase::Total)
                .with_retryable(false)
                .into())));
            return effects;
        }
        match event {
            ChainEvent::Wake => match self.phase {
                Phase::Ready => self.launch(now),
                Phase::Waiting => self.retry(now),
                _ => Vec::new(),
            },
            ChainEvent::Succeeded { attempt } if matches!(self.phase, Phase::Active(id) if id == attempt) =>
            {
                if let Some(attempts) = &mut self.attempts {
                    if let Some(id) = attempts.current() {
                        attempts.succeeded(id, now);
                        attempts.closed(id, now, &mut Vec::new());
                    }
                }
                self.phase = Phase::Finished;
                vec![ChainEffect::Finished(Ok(()))]
            }
            ChainEvent::Failed { attempt, error, produced_output } if matches!(self.phase, Phase::Active(id) if id == attempt) =>
            {
                let cancelled = error.api_kind() == Some(ApiErrorKind::Cancelled);
                self.failures.push(error.clone());
                if produced_output || cancelled {
                    self.phase = Phase::Finished;
                    return vec![ChainEffect::Finished(Err(error.into_final()))];
                }
                let attempts = self.attempts.as_mut().expect("active backend attempts");
                let id = attempts.current().expect("active attempt");
                let mut steps = Vec::new();
                let api_error = error.attempt_error();
                attempts.failed_error(id, api_error, now, &mut steps);
                attempts.closed(id, now, &mut steps);
                if steps.iter().any(|step| matches!(step, AttemptStep::Retry { .. })) {
                    let at = attempts.wake_at(None).expect("retry wake");
                    self.phase = Phase::Waiting;
                    return vec![ChainEffect::WakeAt(at)];
                }
                self.cursor += 1;
                self.attempts = None;
                if self.cursor >= self.entries.len() {
                    self.phase = Phase::Finished;
                    let detail = self
                        .failures
                        .iter()
                        .enumerate()
                        .map(|(i, e)| format!("attempt {}: {}", i + 1, e.message()))
                        .collect::<Vec<_>>()
                        .join("; ");
                    return vec![ChainEffect::Finished(Err(ApiError::connection(Some(format!(
                        "all backends failed ({}): {detail}",
                        self.entries
                            .iter()
                            .map(|(_, _, label)| label.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )))
                    .with_retryable(false)
                    .into()))];
                }
                self.phase = Phase::Ready;
                vec![ChainEffect::WakeAt(now)]
            }
            _ => Vec::new(),
        }
    }

    fn cancel_active(&mut self, now: Ts) -> Vec<ChainEffect> {
        if let Some(attempts) = &mut self.attempts {
            attempts.cancel(now, &mut Vec::new());
        }
        match self.phase {
            Phase::Active(attempt) => vec![ChainEffect::Cancel { attempt }],
            _ => Vec::new(),
        }
    }

    fn retry(&mut self, now: Ts) -> Vec<ChainEffect> {
        let attempts = self.attempts.as_mut().expect("waiting backend attempts");
        let mut steps = Vec::new();
        attempts.wake(now, None, &mut steps);
        if steps.iter().any(|step| matches!(step, AttemptStep::Launch(_))) {
            self.launch_effect()
        } else {
            Vec::new()
        }
    }

    fn launch(&mut self, now: Ts) -> Vec<ChainEffect> {
        let policy = RequestPolicy {
            connect_timeout: None,
            write_timeout: None,
            read_timeout: None,
            total_timeout: None,
            max_retry: self.policy.max_retry_per_backend,
            retry_interval: Duration::from_secs_f64(self.policy.retry_interval.unwrap_or(0.0)),
        };
        let mut attempts = Attempts::start(policy, now, &mut Vec::new());
        attempts.constrain_deadline(self.deadline);
        self.attempts = Some(attempts);
        self.launch_effect()
    }

    fn launch_effect(&mut self) -> Vec<ChainEffect> {
        let Some((index, backend, _)) = self.entries.get(self.cursor).cloned() else {
            self.phase = Phase::Finished;
            return vec![ChainEffect::Finished(Err(ApiError::connection(Some(
                "all backends failed".into(),
            ))
            .into()))];
        };
        self.next_attempt += 1;
        let attempt = self.next_attempt;
        self.phase = Phase::Active(attempt);
        vec![match self.modality {
            Modality::Llm => ChainEffect::LlmRequestStart { attempt, index, llm: backend },
            Modality::Stt => ChainEffect::SttStreamOpen { attempt, index, stt: backend },
            Modality::Tts => ChainEffect::TtsSynthesisStart { attempt, index, tts: backend },
        }]
    }
}

#[cfg(test)]
mod tests;
