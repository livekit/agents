//! Request-local fallback decisions and the single plugin-instance health authority.
//! Adapters execute the effects and report observations; neither owns retry policy.

use std::collections::BTreeMap;
use std::time::Duration;

use lk_agents_abi::Ts;
use lk_agents_abi::errors::{ApiError, ApiErrorKind, ProviderError, TimeoutPhase};
use lk_agents_abi::resources::ResourceId;

use lk_agents_abi::ApiConnectOptions;
pub use lk_agents_abi::llm::fallback::{AvailabilityChange, ErrorPlan};
use lk_agents_net::attempts::{AttemptStep, Attempts, RequestPolicy};

/// The `conn_options` fields the chain rewrites before handing a request to a child.
/// Visible output always forbids replay. `max_retry` defaults per modality (LLM 0, STT 1, TTS 2); the hard 0s
/// are [`caller_default`] and [`conn_options_for_attempt`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChainPolicy {
    pub max_retry_per_backend: u32,
    /// Caps each phase of an attempt; `None` keeps the caller's limits, as TTS does (`tts/fallback_adapter.py:193`).
    pub attempt_timeout: Option<f64>,
    /// Replaces `conn_options.retry_interval`; `None` keeps the caller's.
    pub retry_interval: Option<f64>,
    /// LLM only: keep the entry that last served until it fails, even once a preferred one
    /// recovers; on failure the rest are tried in order (`llm/fallback_adapter.py:96-105`).
    pub sticky: bool,
}

impl ChainPolicy {
    /// `llm/fallback_adapter.py:45-49`: `attempt_timeout=5.0`, `max_retry_per_llm=0`,
    /// `retry_interval=0.5`.
    pub fn llm_defaults() -> Self {
        ChainPolicy {
            max_retry_per_backend: 0,
            attempt_timeout: Some(5.0),
            retry_interval: Some(0.5),
            sticky: false,
        }
    }

    /// `stt/fallback_adapter.py:57-59`: `attempt_timeout=10.0`, `max_retry_per_stt=1`,
    /// `retry_interval=5`. Visible transcripts forbid replay.
    pub fn stt_defaults() -> Self {
        ChainPolicy {
            max_retry_per_backend: 1,
            attempt_timeout: Some(10.0),
            retry_interval: Some(5.0),
            sticky: false,
        }
    }

    /// `tts/fallback_adapter.py:64`: `max_retry_per_tts=2` and nothing else -- the caller's
    /// timeout and retry interval are passed through untouched.
    pub fn tts_defaults() -> Self {
        ChainPolicy {
            max_retry_per_backend: 2,
            attempt_timeout: None,
            retry_interval: None,
            sticky: false,
        }
    }
}

/// Availability, ordering and recovery authorization shared by every provider family.
/// Streaming adapters own this per plugin; realtime owns it per session.
#[derive(Debug, Default)]
pub struct FallbackPolicy {
    entries: BTreeMap<ResourceId, Health>,
    active_index: usize,
    cooldown: Option<Duration>,
}

#[derive(Debug)]
struct Health {
    available: bool,
    generation: u64,
    selected: u64,
    cooldown_until: Option<Ts>,
}

impl Default for Health {
    fn default() -> Self {
        Self { available: true, generation: 0, selected: 0, cooldown_until: None }
    }
}

impl FallbackPolicy {
    pub fn realtime(model_count: usize, cooldown: Duration) -> Self {
        Self {
            entries: (0..model_count).map(|i| (ResourceId(i as u64), Health::default())).collect(),
            cooldown: Some(cooldown),
            ..Self::default()
        }
    }

    pub fn is_available(&self, backend: ResourceId) -> bool {
        self.entries.get(&backend).is_none_or(|entry| entry.available)
    }

    /// Every observation supersedes probes authorized before it, even a repeated failure.
    pub fn observe(&mut self, backend: ResourceId, available: bool) -> bool {
        let entry = self.entries.entry(backend).or_default();
        entry.generation += 1;
        let changed = entry.available != available;
        entry.available = available;
        if available {
            entry.cooldown_until = None;
        }
        changed
    }

    pub fn select(&mut self, backend: ResourceId, selection: u64) {
        self.entries.entry(backend).or_default().selected = selection;
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

    pub fn active_index(&self) -> usize {
        self.active_index
    }

    pub fn set_active_index(&mut self, index: usize) {
        self.active_index = index;
    }

    /// Repeated failure does not re-arm the realtime cooldown or announce another transition.
    pub fn set_available(
        &mut self,
        index: usize,
        available: bool,
        now: Ts,
    ) -> Option<AvailabilityChange> {
        let backend = ResourceId(index as u64);
        let entry = self.entries.get(&backend)?;
        if entry.available == available {
            return None;
        }
        self.observe(backend, available);
        if !available {
            self.entries.get_mut(&backend).expect("known backend").cooldown_until =
                self.cooldown.map(|cooldown| now.saturating_add(cooldown));
        }
        Some(AvailabilityChange { index, available })
    }

    fn sweep(&mut self, now: Ts) -> Vec<AvailabilityChange> {
        let expired: Vec<_> = self
            .entries
            .iter()
            .filter(|(_, entry)| entry.cooldown_until.is_some_and(|at| at <= now))
            .map(|(id, _)| id.0 as usize)
            .collect();
        expired.into_iter().filter_map(|index| self.set_available(index, true, now)).collect()
    }

    fn next_available(&self, exclude_current: bool) -> Option<usize> {
        self.entries
            .iter()
            .find(|(id, entry)| {
                entry.available && !(exclude_current && id.0 as usize == self.active_index)
            })
            .map(|(id, _)| id.0 as usize)
    }

    pub fn next_available_index(
        &mut self,
        now: Ts,
        exclude_current: bool,
    ) -> (Option<usize>, Vec<AvailabilityChange>) {
        let changes = self.sweep(now);
        (self.next_available(exclude_current), changes)
    }

    /// Child reconnects retain ownership of recoverable errors. A terminal failure starts one
    /// fallback pass; even zero cooldown cannot resurrect that child in the same pass.
    pub fn on_child_error(&mut self, recoverable: bool, swap_busy: bool, now: Ts) -> ErrorPlan {
        if recoverable {
            return ErrorPlan { changes: vec![], rewrite_recoverable: false, swap_to: None };
        }
        let mut changes = self.sweep(now);
        changes.extend(self.set_available(self.active_index, false, now));
        let target = self.next_available(false);
        ErrorPlan {
            changes,
            rewrite_recoverable: target.is_some(),
            swap_to: target.filter(|_| !swap_busy),
        }
    }

    pub fn restart_target(
        &mut self,
        switch_model: bool,
        now: Ts,
    ) -> (usize, Vec<AvailabilityChange>) {
        if !switch_model {
            return (self.active_index, vec![]);
        }
        let changes = self.sweep(now);
        (self.next_available(true).unwrap_or(self.active_index), changes)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Modality {
    Llm,
    Stt,
    Tts,
    Realtime,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChainEffect {
    LlmRequestStart { attempt: u64, index: usize, llm: ResourceId },
    SttStreamOpen { attempt: u64, index: usize, stt: ResourceId },
    TtsSynthesisStart { attempt: u64, index: usize, tts: ResourceId },
    RealtimeSessionOpen { attempt: u64, index: usize, model: ResourceId },
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
        health: &FallbackPolicy,
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
            Modality::Realtime => {
                ChainEffect::RealtimeSessionOpen { attempt, index, model: backend }
            }
        }]
    }
}

#[cfg(test)]
mod tests;
