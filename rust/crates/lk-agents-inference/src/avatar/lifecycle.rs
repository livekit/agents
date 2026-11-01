//! One avatar instance's provider sessions as a synchronous machine on supplied time. A create is
//! reserved before it launches, retried under one operation identity (its request id, which the
//! runtime turns into one idempotency key), delivered to its caller and adopted by the instance,
//! or retired: every session the gateway reports is either held by the instance or terminated.
//!
//! One creation at a time: a second caller is refused while one is in flight or once a session
//! was adopted, and the first creation is left untouched. A create the gateway may have received
//! is never abandoned: when its caller stops waiting, the instance closes, or it fails without an
//! answer after it was sent, it is retired: asked again under its identity until the gateway
//! answers, and a session it made is terminated rather than leaked. A retirement still unanswered
//! at its deadline and a missed cleanup deadline are each reported once.

use std::collections::BTreeMap;
use std::time::Duration;

use lk_agents_abi::Ts;
use lk_agents_abi::errors::ApiError;
use lk_agents_abi::resources::ResourceId;

use lk_agents_core::attempts::{AttemptFailure, AttemptStep, Attempts, RequestPolicy};

/// How long a create no caller takes keeps asking the gateway for the session it may have made.
pub const RETIREMENT_TIMEOUT: Duration = Duration::from_secs(30);

/// A gateway request of one avatar instance: a create (its operation identity) or a terminate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct AvatarRequestId(pub u64);

/// A provider session the gateway created.
#[derive(Clone, PartialEq)]
pub struct ProviderSession {
    /// The gateway's id.
    pub session_id: Option<String>,
    /// The provider's own id; terminate names it.
    pub provider_session_id: Option<String>,
    /// The HMAC issued with `provider_session_id`: terminate proves this project owns it.
    pub terminate_token: Option<String>,
    /// The audio rate the avatar takes.
    pub sample_rate: u32,
}

impl ProviderSession {
    /// Without both ids it bills until the provider's idle timeout.
    pub fn terminable(&self) -> bool {
        self.provider_session_id.is_some() && self.terminate_token.is_some()
    }

    /// The gateway replays one operation's answer under its idempotency key.
    fn same_as(&self, other: &ProviderSession) -> bool {
        self.session_id == other.session_id && self.provider_session_id == other.provider_session_id
    }
}

impl std::fmt::Debug for ProviderSession {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProviderSession")
            .field("session_id", &self.session_id)
            .field("provider_session_id", &self.provider_session_id)
            .field("terminate_token", &self.terminate_token.as_ref().map(|_| "<redacted>"))
            .field("sample_rate", &self.sample_rate)
            .finish()
    }
}

#[derive(Debug, Clone)]
pub enum AvatarSessionEvent {
    /// A deadline from [`AvatarSessionLifecycle::wake_at`] is due.
    Wake,
    /// A caller asks for a session.
    Create,
    /// The create caller stopped waiting.
    Cancel,
    /// Terminate every held session and stop creating (`aclose()`); a repeat retries what a
    /// failed terminate left held.
    Close,
    /// The caller received the delivered session: the instance holds it.
    Adopted,
    /// The delivered session found no caller to receive it.
    DeliveryRefused,
    Connected {
        request: AvatarRequestId,
        attempt: u32,
    },
    RequestWritten {
        request: AvatarRequestId,
        attempt: u32,
    },
    Created {
        request: AvatarRequestId,
        attempt: u32,
        session: ProviderSession,
    },
    Terminated {
        request: AvatarRequestId,
        attempt: u32,
    },
    Failed {
        request: AvatarRequestId,
        attempt: u32,
        failure: AttemptFailure,
    },
    /// The attempt's transport resources are released.
    AttemptClosed {
        request: AvatarRequestId,
        attempt: u32,
    },
}

/// Why a create caller got no session.
#[derive(Debug, Clone, PartialEq)]
pub enum CreateFailure {
    Api(ApiError),
    /// The instance was closed while creating.
    Closed,
}

/// Why a session is terminated without being held: the journalled disposition of a result no
/// caller adopted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Retirement {
    /// Its caller stopped waiting.
    CallerGone,
    /// It was delivered and nobody received it.
    Refused,
    /// The instance was closed.
    Closed,
    /// A replayed answer the operation had settled without.
    Late,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AvatarSessionEffect {
    /// The caller is refused: a creation is in flight, a session was adopted or the instance
    /// is closed.
    CreateRefused,
    AvatarSessionCreate {
        request: AvatarRequestId,
        attempt: u32,
        avatar: ResourceId,
    },
    AvatarSessionTerminate {
        request: AvatarRequestId,
        attempt: u32,
        avatar: ResourceId,
        session: ProviderSession,
    },
    AvatarRequestCancel {
        request: AvatarRequestId,
        attempt: u32,
    },
    /// A failed attempt that will be retried.
    AttemptFailed {
        request: AvatarRequestId,
        attempt: u32,
        error: ApiError,
    },
    /// The create caller's single outcome. A caller that stopped waiting has none.
    CreateFinished(Result<ProviderSession, CreateFailure>),
    Retiring {
        session: ProviderSession,
        reason: Retirement,
    },
    /// A session that cannot be terminated (no provider id or token) is released to its idle
    /// timeout.
    Unterminable(ProviderSession),
    SessionTerminated(ProviderSession),
    /// The instance still holds `session`; the next close retries.
    TerminateFailed {
        session: ProviderSession,
        error: ApiError,
    },
    /// An attempt missed its cleanup deadline; the runtime still owns it.
    CleanupIncomplete {
        request: AvatarRequestId,
        attempt: u32,
    },
    /// A retiring create was still unanswered at its deadline: a session it may have made bills
    /// until the provider's idle timeout; a later answer is still terminated.
    RetirementIncomplete {
        request: AvatarRequestId,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Create,
    Terminate,
}

#[derive(Debug)]
struct Request {
    kind: Kind,
    attempts: Attempts,
}

/// A create in flight; `caller` is owed its outcome until it stops waiting.
#[derive(Debug)]
struct Creating {
    request: AvatarRequestId,
    caller: bool,
}

/// An exchange's response is expected whenever it is being read: the caller always waits.
const EXPECTING: Option<Ts> = Some(Ts::ZERO);

#[derive(Debug)]
pub struct AvatarSessionLifecycle {
    avatar: ResourceId,
    policy: RequestPolicy,
    next_request: u64,
    /// Every request whose outcome is undecided or whose attempts still owe their release.
    requests: BTreeMap<AvatarRequestId, Request>,
    creating: Option<Creating>,
    delivering: Option<ProviderSession>,
    /// Sessions the instance owns with no terminate running.
    held: Vec<ProviderSession>,
    terminating: BTreeMap<AvatarRequestId, ProviderSession>,
    /// A session was adopted: one instance is one avatar.
    adopted: bool,
    closing: bool,
}

impl AvatarSessionLifecycle {
    pub fn new(avatar: ResourceId, policy: RequestPolicy) -> Self {
        AvatarSessionLifecycle {
            avatar,
            policy,
            next_request: 1,
            requests: BTreeMap::new(),
            creating: None,
            delivering: None,
            held: Vec::new(),
            terminating: BTreeMap::new(),
            adopted: false,
            closing: false,
        }
    }

    /// The earliest instant a [`AvatarSessionEvent::Wake`] has work to do.
    pub fn wake_at(&self) -> Option<Ts> {
        self.requests.values().filter_map(|r| r.attempts.wake_at(EXPECTING)).min()
    }

    /// Nothing is in flight: no request, no attempt owing its release, no pending delivery.
    pub fn is_settled(&self) -> bool {
        self.requests.is_empty() && self.creating.is_none() && self.delivering.is_none()
    }

    /// Whether `request` is undecided or still owes an attempt's release.
    pub fn tracks(&self, request: AvatarRequestId) -> bool {
        self.requests.contains_key(&request)
    }

    pub fn handle(&mut self, event: AvatarSessionEvent, now: Ts) -> Vec<AvatarSessionEffect> {
        let mut effects = Vec::new();
        match event {
            AvatarSessionEvent::Wake => {
                let ids: Vec<AvatarRequestId> = self.requests.keys().copied().collect();
                for id in ids {
                    self.step(id, now, &mut effects, |attempts, steps| {
                        attempts.wake(now, EXPECTING, steps)
                    });
                }
            }
            AvatarSessionEvent::Create => self.create(now, &mut effects),
            AvatarSessionEvent::Cancel => self.stop_creating(now, &mut effects),
            AvatarSessionEvent::Close => {
                self.closing = true;
                if self.creating.as_ref().is_some_and(|c| c.caller) {
                    effects.push(AvatarSessionEffect::CreateFinished(Err(CreateFailure::Closed)));
                }
                self.stop_creating(now, &mut effects);
                for session in std::mem::take(&mut self.held) {
                    self.terminate(session, None, now, &mut effects);
                }
            }
            AvatarSessionEvent::Adopted => {
                if let Some(session) = self.delivering.take() {
                    self.adopted = true;
                    if self.closing {
                        self.terminate(session, None, now, &mut effects);
                    } else {
                        self.held.push(session);
                    }
                }
            }
            AvatarSessionEvent::DeliveryRefused => {
                if let Some(session) = self.delivering.take() {
                    self.terminate(session, Some(Retirement::Refused), now, &mut effects);
                }
            }
            AvatarSessionEvent::Connected { request, attempt } => {
                if let Some(r) = self.requests.get_mut(&request) {
                    r.attempts.connected(attempt, now);
                }
            }
            AvatarSessionEvent::RequestWritten { request, attempt } => {
                if let Some(r) = self.requests.get_mut(&request) {
                    r.attempts.written(attempt, now);
                }
            }
            AvatarSessionEvent::Created { request, attempt, session } => {
                self.created(request, attempt, session, now, &mut effects);
            }
            AvatarSessionEvent::Terminated { request, attempt } => {
                if self.answered(request, attempt, now, &mut effects)
                    && let Some(session) = self.terminating.remove(&request)
                {
                    effects.push(AvatarSessionEffect::SessionTerminated(session));
                }
                self.forget_settled(request);
            }
            AvatarSessionEvent::Failed { request, attempt, failure } => {
                self.step(request, now, &mut effects, |attempts, steps| {
                    attempts.failed(attempt, failure, now, steps)
                });
            }
            AvatarSessionEvent::AttemptClosed { request, attempt } => {
                self.step(request, now, &mut effects, |attempts, steps| {
                    attempts.closed(attempt, now, steps)
                });
            }
        }
        effects
    }

    /// Reserves the creation before its first attempt launches.
    fn create(&mut self, now: Ts, effects: &mut Vec<AvatarSessionEffect>) {
        if self.closing || self.adopted || self.creating.is_some() || self.delivering.is_some() {
            effects.push(AvatarSessionEffect::CreateRefused);
            return;
        }
        let request = self.reserve();
        self.creating = Some(Creating { request, caller: true });
        let mut steps = Vec::new();
        let attempts = Attempts::start(self.policy, now, &mut steps);
        self.requests.insert(request, Request { kind: Kind::Create, attempts });
        self.apply(request, steps, now, effects);
    }

    /// No caller is owed the creation any more: it is retired.
    fn stop_creating(&mut self, now: Ts, effects: &mut Vec<AvatarSessionEffect>) {
        if let Some(creating) = &mut self.creating {
            creating.caller = false;
            let request = creating.request;
            self.retire(request, now, effects);
            self.forget_settled(request);
        }
    }

    /// `request` keeps asking under its identity until the gateway answers it, or stops now when
    /// the gateway cannot have received it.
    fn retire(
        &mut self,
        request: AvatarRequestId,
        now: Ts,
        effects: &mut Vec<AvatarSessionEffect>,
    ) {
        let Some(r) = self.requests.get_mut(&request) else { return };
        let mut steps = Vec::new();
        r.attempts.retire(now, now.saturating_add(RETIREMENT_TIMEOUT), &mut steps);
        self.apply(request, steps, now, effects);
    }

    fn created(
        &mut self,
        request: AvatarRequestId,
        attempt: u32,
        session: ProviderSession,
        now: Ts,
        effects: &mut Vec<AvatarSessionEffect>,
    ) {
        let answered = self.answered(request, attempt, now, effects);
        let creating = self.creating.as_ref().filter(|c| c.request == request);
        match creating {
            Some(c) if answered && c.caller && !self.closing => {
                self.creating = None;
                self.delivering = Some(session.clone());
                effects.push(AvatarSessionEffect::CreateFinished(Ok(session)));
            }
            Some(_) if answered => {
                self.creating = None;
                let reason = if self.closing { Retirement::Closed } else { Retirement::CallerGone };
                self.terminate(session, Some(reason), now, effects);
            }
            _ if self.knows(&session) => {}
            _ => self.terminate(session, Some(Retirement::Late), now, effects),
        }
        self.forget_settled(request);
    }

    /// Whether this answer decides `request`. Any attempt of an undecided request answers it: the
    /// operation's identity makes every attempt's answer the same, so a still-running attempt is
    /// stopped.
    fn answered(
        &mut self,
        request: AvatarRequestId,
        attempt: u32,
        now: Ts,
        effects: &mut Vec<AvatarSessionEffect>,
    ) -> bool {
        let Some(r) = self.requests.get_mut(&request) else { return false };
        if r.attempts.is_decided() {
            return false;
        }
        if !r.attempts.succeeded(attempt, now) {
            let mut steps = Vec::new();
            r.attempts.cancel(now, &mut steps);
            self.apply(request, steps, now, effects);
        }
        true
    }

    /// Whether `session` is already held, delivered or being terminated.
    fn knows(&self, session: &ProviderSession) -> bool {
        self.delivering
            .iter()
            .chain(&self.held)
            .chain(self.terminating.values())
            .any(|s| s.same_as(session))
    }

    /// Terminates `session`; `reason` journals a session no caller holds.
    fn terminate(
        &mut self,
        session: ProviderSession,
        reason: Option<Retirement>,
        now: Ts,
        effects: &mut Vec<AvatarSessionEffect>,
    ) {
        if let Some(reason) = reason {
            effects.push(AvatarSessionEffect::Retiring { session: session.clone(), reason });
        }
        if !session.terminable() {
            effects.push(AvatarSessionEffect::Unterminable(session));
            return;
        }
        let request = self.reserve();
        self.terminating.insert(request, session);
        let mut steps = Vec::new();
        let attempts = Attempts::start(self.policy, now, &mut steps);
        self.requests.insert(request, Request { kind: Kind::Terminate, attempts });
        self.apply(request, steps, now, effects);
    }

    fn reserve(&mut self) -> AvatarRequestId {
        let id = AvatarRequestId(self.next_request);
        self.next_request += 1;
        id
    }

    fn step(
        &mut self,
        request: AvatarRequestId,
        now: Ts,
        effects: &mut Vec<AvatarSessionEffect>,
        f: impl FnOnce(&mut Attempts, &mut Vec<AttemptStep>),
    ) {
        let Some(r) = self.requests.get_mut(&request) else { return };
        let mut steps = Vec::new();
        f(&mut r.attempts, &mut steps);
        self.apply(request, steps, now, effects);
        self.forget_settled(request);
    }

    fn apply(
        &mut self,
        request: AvatarRequestId,
        steps: Vec<AttemptStep>,
        now: Ts,
        effects: &mut Vec<AvatarSessionEffect>,
    ) {
        let Some(kind) = self.requests.get(&request).map(|r| r.kind) else { return };
        for step in steps {
            match step {
                AttemptStep::Launch(attempt) => effects.push(match kind {
                    Kind::Create => AvatarSessionEffect::AvatarSessionCreate {
                        request,
                        attempt,
                        avatar: self.avatar,
                    },
                    Kind::Terminate => AvatarSessionEffect::AvatarSessionTerminate {
                        request,
                        attempt,
                        avatar: self.avatar,
                        session: self.terminating[&request].clone(),
                    },
                }),
                AttemptStep::Stop(attempt) => {
                    effects.push(AvatarSessionEffect::AvatarRequestCancel { request, attempt })
                }
                AttemptStep::Retry { attempt, error } => {
                    effects.push(AvatarSessionEffect::AttemptFailed { request, attempt, error });
                }
                AttemptStep::CleanupIncomplete(attempt) => {
                    effects.push(AvatarSessionEffect::CleanupIncomplete { request, attempt });
                }
                AttemptStep::Unresolved => {
                    effects.push(AvatarSessionEffect::RetirementIncomplete { request })
                }
                AttemptStep::GiveUp(error) => match kind {
                    Kind::Create => {
                        if let Some(creating) = self.creating.take_if(|c| c.request == request)
                            && creating.caller
                        {
                            effects.push(AvatarSessionEffect::CreateFinished(Err(
                                CreateFailure::Api(error),
                            )));
                        }
                        // The caller has its error; what the gateway may have made is retired.
                        self.retire(request, now, effects);
                    }
                    Kind::Terminate => {
                        if let Some(session) = self.terminating.remove(&request) {
                            self.held.push(session.clone());
                            effects.push(AvatarSessionEffect::TerminateFailed { session, error });
                        }
                    }
                },
            }
        }
    }

    /// Drops `request` once nothing is owed for it; a decided create no longer reserves.
    fn forget_settled(&mut self, request: AvatarRequestId) {
        let decided = self.requests.get(&request).is_none_or(|r| r.attempts.is_decided());
        if decided {
            self.creating.take_if(|c| c.request == request);
        }
        if self.requests.get(&request).is_some_and(|r| r.attempts.is_settled()) {
            self.requests.remove(&request);
        }
    }
}

#[cfg(test)]
#[path = "lifecycle_tests.rs"]
mod tests;
