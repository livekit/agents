//! `FallbackChain` as a state machine: a ports recorder, a [`VirtualClock`], and a scripted
//! [`Backend`]. Probes are real tasks, gated by semaphores and observed through [`until`].

use std::collections::VecDeque;
#[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
use std::sync::atomic::{AtomicUsize, Ordering};
#[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use lk_agents_abi::ApiConnectOptions;
use lk_agents_abi::StreamSource;
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::{BoxFuture, Spawner, TokioSpawner};
use lk_agents_abi::{Clock, VirtualClock};
use tokio::sync::Semaphore;

use crate::fallback::ChainPolicy;
use crate::fallback::chain::{
    Attempt, AvailabilityChanged, Backend, ChainObserver, ChainOutcome, FallbackChain, Recovery,
    RecoverySlot,
};

const LIMIT: Duration = Duration::from_secs(5);

#[allow(clippy::disallowed_methods, reason = "S0: existing test harness live resources")]
async fn within<F: std::future::Future>(what: &str, fut: F) -> F::Output {
    match tokio::time::timeout(LIMIT, fut).await {
        Ok(v) => v,
        Err(_) => panic!("timed out after {LIMIT:?} waiting for {what}"),
    }
}

#[allow(clippy::disallowed_methods, reason = "S0: existing test harness live resources")]
async fn until(what: &str, mut cond: impl FnMut() -> bool) {
    within(what, async {
        while !cond() {
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    })
    .await;
}

// ---------------------------------------------------------------------------------------------
// Ports recorder
// ---------------------------------------------------------------------------------------------

#[derive(Default)]
struct Recorder {
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    events: Mutex<Vec<AvailabilityChanged>>,
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    all_unavailable: AtomicUsize,
}

impl Recorder {
    fn events(&self) -> Vec<AvailabilityChanged> {
        self.events.lock().expect("recorder").clone()
    }
    fn transitions(&self) -> Vec<(usize, bool)> {
        self.events().into_iter().map(|e| (e.index, e.available)).collect()
    }
}

impl ChainObserver for Recorder {
    fn availability_changed(&self, ev: AvailabilityChanged) {
        self.events.lock().expect("recorder").push(ev);
    }
    fn all_unavailable(&self) {
        self.all_unavailable.fetch_add(1, Ordering::SeqCst);
    }
}

// ---------------------------------------------------------------------------------------------
// A scripted backend
// ---------------------------------------------------------------------------------------------

/// What the next `open`/drain pair should do; the last entry repeats forever.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Step {
    /// `open` raises -- the plugin's `chat()` failing during provider-format conversion.
    OpenErr,
    Serve,
    /// The session opened and then died with nothing emitted.
    FailCold,
    /// The session emitted at least one item and then died. Feeds [`AfterOutput`].
    FailHot,
}

struct Fake {
    label: String,
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    script: Mutex<VecDeque<Step>>,
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    opened: Mutex<Vec<ApiConnectOptions>>,
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    probes: Mutex<Vec<(RecoverySlot, ApiConnectOptions)>>,
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    probe_verdict: Mutex<VecDeque<Recovery>>,
    /// A probe on slot `i` waits here. Starts saturated, so probes complete immediately unless a
    /// test closes the gate.
    gate: [Arc<Semaphore>; 2],
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    probes_entered: AtomicUsize,
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    /// Incremented as the last statement of `recover`, so a test can `until` on "the probe
    /// finished" without racing the availability event.
    probes_done: AtomicUsize,
}

impl Fake {
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    fn new(label: &str, script: &[Step]) -> Arc<Self> {
        Arc::new(Fake {
            label: label.into(),
            script: Mutex::new(script.iter().copied().collect()),
            opened: Mutex::new(Vec::new()),
            probes: Mutex::new(Vec::new()),
            probe_verdict: Mutex::new(VecDeque::new()),
            gate: [
                Arc::new(Semaphore::new(usize::MAX >> 4)),
                Arc::new(Semaphore::new(usize::MAX >> 4)),
            ],
            probes_entered: AtomicUsize::new(0),
            probes_done: AtomicUsize::new(0),
        })
    }

    fn recovers(self: &Arc<Self>, verdicts: &[Recovery]) -> Arc<Self> {
        *self.probe_verdict.lock().expect("verdicts") = verdicts.iter().copied().collect();
        Arc::clone(self)
    }

    fn shut(self: &Arc<Self>, slot: RecoverySlot) -> Arc<Self> {
        // Drain the permits so the next probe on this slot parks.
        self.gate[slot as usize].forget_permits(usize::MAX >> 4);
        Arc::clone(self)
    }

    fn next_step(&self) -> Step {
        let mut s = self.script.lock().expect("script");
        if s.len() > 1 {
            s.pop_front().expect("non-empty")
        } else {
            *s.front().expect("script must not be empty")
        }
    }

    fn probe_count(&self, slot: RecoverySlot) -> usize {
        self.probes.lock().expect("probes").iter().filter(|(s, _)| *s == slot).count()
    }
}

fn boom() -> ProviderError {
    ProviderError::Local(ApiError::new("boom"))
}

#[async_trait]
impl Backend for Fake {
    type Req = String;
    type Session = Step;

    fn resource_id(&self) -> lk_agents_abi::resources::ResourceId {
        super::backend_resource(self)
    }

    fn label(&self) -> String {
        self.label.clone()
    }

    async fn open(
        &self,
        _req: &Self::Req,
        conn: &ApiConnectOptions,
    ) -> Result<Self::Session, ProviderError> {
        self.opened.lock().expect("opened").push(*conn);
        match self.next_step() {
            Step::OpenErr => Err(boom()),
            other => Ok(other),
        }
    }

    async fn recover(
        &self,
        _req: &Self::Req,
        conn: &ApiConnectOptions,
        slot: RecoverySlot,
        _stop: tokio::sync::watch::Receiver<bool>,
    ) -> Result<Recovery, ProviderError> {
        self.probes.lock().expect("probes").push((slot, *conn));
        self.probes_entered.fetch_add(1, Ordering::SeqCst);
        let _permit = self.gate[slot as usize].acquire().await.expect("gate is never closed");
        let verdict = match self.probe_verdict.lock().expect("verdicts").pop_front() {
            Some(v) => Ok(v),
            None => Err(boom()),
        };
        self.probes_done.fetch_add(1, Ordering::SeqCst);
        verdict
    }
}

// ---------------------------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------------------------

struct Rig {
    chain: Arc<FallbackChain<Fake>>,
    rec: Arc<Recorder>,
    clock: Arc<VirtualClock>,
}

fn rig(backends: Vec<Arc<Fake>>, mut policy: ChainPolicy) -> Rig {
    policy.max_retry_per_backend = 0;
    policy.retry_interval = Some(0.0);
    let rec = Arc::new(Recorder::default());
    let clock = Arc::new(VirtualClock::new());
    let spawner: Arc<dyn Spawner> = Arc::new(TokioSpawner::current());
    let chain = FallbackChain::new(
        backends.clone(),
        policy,
        Arc::clone(&clock) as Arc<dyn Clock>,
        spawner,
        Arc::clone(&rec) as Arc<dyn ChainObserver>,
    )
    .expect("at least one backend");
    let _ = &backends;
    Rig { chain: Arc::new(chain), rec, clock }
}

fn caller() -> ApiConnectOptions {
    ApiConnectOptions::default()
}

/// Runs one request, recording which entries drained; the drain calls `mark_serving` on first
/// output, as a real adapter does.
async fn run_once(
    rig: &Rig,
    slot: RecoverySlot,
) -> (Result<ChainOutcome<usize>, ProviderError>, Vec<usize>) {
    let mut drained: Vec<usize> = Vec::new();
    let chain = Arc::clone(&rig.chain);
    let out = rig
        .chain
        .run(
            &"req".to_string(),
            &caller(),
            slot,
            &mut drained,
            move |index, _b, session, drained| {
                let chain = Arc::clone(&chain);
                Box::pin(async move {
                    drained.push(index);
                    match session {
                        Step::Serve => {
                            chain.mark_serving(index);
                            Attempt::Served(index)
                        }
                        Step::FailHot => {
                            chain.mark_serving(index);
                            Attempt::Failed { error: boom(), produced_output: true }
                        }
                        _ => Attempt::Failed { error: boom(), produced_output: false },
                    }
                }) as BoxFuture<'_, Attempt<usize>>
            },
        )
        .await;
    (out, drained)
}

// ---------------------------------------------------------------------------------------------
// The behaviours worth pinning
// ---------------------------------------------------------------------------------------------

// Catches: a cursor. `llm/fallback_adapter.py:274` restarts at index 0 on every request.
#[tokio::test]
async fn recovery_returns_traffic_to_an_earlier_entry() {
    let primary = Fake::new("primary", &[Step::FailCold, Step::Serve])
        .recovers(&[Recovery::Recovered])
        .shut(RecoverySlot::OneShot);
    let backup = Fake::new("backup", &[Step::Serve]);
    let rig = rig(vec![Arc::clone(&primary), backup], ChainPolicy::llm_defaults());

    let (out, drained) = run_once(&rig, RecoverySlot::OneShot).await;
    assert_eq!(out.expect("the backup serves"), ChainOutcome::Served(1));
    assert_eq!(drained, vec![0, 1], "the primary is tried first, then the backup");
    assert!(!rig.chain.is_available(0));

    primary.gate[RecoverySlot::OneShot as usize].add_permits(1);
    until("the primary's probe to succeed", || rig.chain.is_available(0)).await;

    let (out, drained) = run_once(&rig, RecoverySlot::OneShot).await;
    assert_eq!(out.expect("the primary serves again"), ChainOutcome::Served(0));
    assert_eq!(drained, vec![0], "the backup must not be reached once the primary is back");
    assert_eq!(rig.chain.active_index(), 0);
    assert_eq!(rig.rec.transitions(), vec![(0, false), (0, true)]);
}

// Catches: a caller's `conn_options` leaking to the wrapped provider; also pins the probe's hard 0.
#[tokio::test]
async fn the_rewritten_conn_options_reach_the_provider_on_both_paths() {
    let primary = Fake::new("primary", &[Step::FailCold]).recovers(&[Recovery::NotYet]);
    let rig = rig(vec![Arc::clone(&primary)], ChainPolicy::tts_defaults());

    let _ = run_once(&rig, RecoverySlot::Streaming).await;
    let opened = primary.opened.lock().expect("opened").clone();
    assert_eq!(opened.len(), 1);
    let limits =
        ApiConnectOptions { max_retry: 0, retry_interval: opened[0].retry_interval, ..caller() };
    assert_eq!(opened[0], limits, "the chain owns every retry; TTS keeps the caller's limits");

    until("the probe to run", || primary.probe_count(RecoverySlot::Streaming) == 1).await;
    let probes = primary.probes.lock().expect("probes").clone();
    assert_eq!(probes[0].1.max_retry, 0, "a probe must never retry internally");
}

// Catches: the two bugs of a shared recovery slot (`tts/fallback_adapter.py:39-43`).
#[tokio::test]
async fn the_two_recovery_slots_do_not_suppress_or_clobber_each_other() {
    let primary = Fake::new("primary", &[Step::FailCold])
        .recovers(&[Recovery::NotYet, Recovery::NotYet])
        .shut(RecoverySlot::OneShot)
        .shut(RecoverySlot::Streaming);
    let rig = rig(vec![Arc::clone(&primary)], ChainPolicy::stt_defaults());

    let _ = run_once(&rig, RecoverySlot::OneShot).await;
    until("the batch probe to park", || primary.probe_count(RecoverySlot::OneShot) == 1).await;

    // A batch probe is parked. The streaming path must still be probeable.
    let _ = run_once(&rig, RecoverySlot::Streaming).await;
    until("the streaming probe to park", || primary.probe_count(RecoverySlot::Streaming) == 1)
        .await;

    // A second request on the batch path must not start a second batch probe: one is live.
    let _ = run_once(&rig, RecoverySlot::OneShot).await;
    tokio::task::yield_now().await;
    assert_eq!(primary.probe_count(RecoverySlot::OneShot), 1, "one live probe per slot");

    // Both handles must still be reachable, i.e. neither slot overwrote the other.
    rig.chain.aclose();
    until("both parked probes to be aborted", || primary.gate[0].available_permits() == 0).await;
    assert_eq!(primary.probes_entered.load(Ordering::SeqCst), 2);
}

// Catches: failing fast when every entry is down instead of the `all_failed` pass.
#[tokio::test]
async fn when_everything_is_unavailable_every_entry_is_tried_anyway() {
    let a = Fake::new("a", &[Step::FailCold, Step::Serve]).recovers(&[Recovery::NotYet]);
    let b = Fake::new("b", &[Step::FailCold]).recovers(&[Recovery::NotYet]);
    let rig = rig(vec![Arc::clone(&a), b], ChainPolicy::llm_defaults());

    let (out, drained) = run_once(&rig, RecoverySlot::OneShot).await;
    assert!(out.is_err());
    assert_eq!(drained, vec![0, 1]);
    assert_eq!(rig.rec.all_unavailable.load(Ordering::SeqCst), 0, "not yet: they were up on entry");

    let (out, drained) = run_once(&rig, RecoverySlot::OneShot).await;
    assert_eq!(out.expect("a recovered on the retry-everything pass"), ChainOutcome::Served(0));
    assert_eq!(drained, vec![0], "a serves, so b is never reached");
    assert_eq!(rig.rec.all_unavailable.load(Ordering::SeqCst), 1);
}

// Catches: probing only the entry that just failed, never a skipped unavailable one.
#[tokio::test]
async fn a_skipped_entry_is_still_probed() {
    let a = Fake::new("a", &[Step::FailCold]).recovers(&[Recovery::NotYet, Recovery::NotYet]);
    let b = Fake::new("b", &[Step::Serve]);
    let rig = rig(vec![Arc::clone(&a), b], ChainPolicy::llm_defaults());

    let _ = run_once(&rig, RecoverySlot::OneShot).await;
    until("the first probe", || a.probe_count(RecoverySlot::OneShot) == 1).await;

    // `a` is now unavailable and will be skipped, not opened -- but must still be probed.
    let opens_before = a.opened.lock().expect("opened").len();
    let _ = run_once(&rig, RecoverySlot::OneShot).await;
    until("the second probe", || a.probe_count(RecoverySlot::OneShot) == 2).await;
    assert_eq!(a.opened.lock().expect("opened").len(), opens_before, "skipped means not opened");
}

// Catches: a probe started after `aclose` (`tts:155-159`).
#[tokio::test]
async fn aclose_refuses_later_probes() {
    let a = Fake::new("a", &[Step::FailCold]).recovers(&[Recovery::NotYet]);
    let rig = rig(vec![Arc::clone(&a)], ChainPolicy::llm_defaults());

    rig.chain.aclose();
    let _ = run_once(&rig, RecoverySlot::OneShot).await;
    tokio::task::yield_now().await;
    assert_eq!(a.probe_count(RecoverySlot::OneShot), 0);
}

// Catches: reading wall time for the exhaustion error's elapsed figure.
#[tokio::test]
async fn the_exhaustion_error_measures_the_injected_clock() {
    let a = Fake::new("a", &[Step::OpenErr]).recovers(&[Recovery::NotYet]);
    let rig = rig(vec![Arc::clone(&a)], ChainPolicy::llm_defaults());

    let mut drained: Vec<usize> = Vec::new();
    let clock = Arc::clone(&rig.clock);
    let out = within(
        "the chain to exhaust its entries",
        rig.chain.run(
            &"req".to_string(),
            &caller(),
            RecoverySlot::OneShot,
            &mut drained,
            move |i, _b, s, d| {
                clock.advance(Duration::from_millis(1500));
                Box::pin(async move {
                    d.push(i);
                    let _ = s;
                    Attempt::Failed { error: boom(), produced_output: false }
                }) as BoxFuture<'_, Attempt<usize>>
            },
        ),
    )
    .await;

    let err = out.expect_err("every entry failed");
    assert!(err.message().contains("all backends failed"), "{}", err.message());
    assert!(
        err.message().contains("boom"),
        "open() failed, so the drain never advanced time: {}",
        err.message()
    );
}

// Catches: dropping the `if st.available != available` guard (repeated events under `all_failed`).
#[tokio::test]
async fn a_provider_that_fails_again_while_already_down_emits_nothing() {
    let verdicts = [Recovery::NotYet; 6];
    let a = Fake::new("a", &[Step::FailCold]).recovers(&verdicts);
    let b = Fake::new("b", &[Step::FailCold]).recovers(&verdicts);
    let rig = rig(vec![a, b], ChainPolicy::llm_defaults());

    let (out, drained) = run_once(&rig, RecoverySlot::OneShot).await;
    assert!(out.is_err());
    assert_eq!(drained, vec![0, 1]);

    // Everything is down now, so this pass re-opens both and both fail a second time.
    let (out, drained) = run_once(&rig, RecoverySlot::OneShot).await;
    assert!(out.is_err());
    assert_eq!(drained, vec![0, 1], "all_failed tries everything again");
    assert_eq!(rig.rec.all_unavailable.load(Ordering::SeqCst), 1);

    assert_eq!(
        rig.rec.transitions(),
        vec![(0, false), (1, false)],
        "one event per provider, on the transition -- not one per failure"
    );
}

// Catches: dropping the `if !st.available` guard on recovery (the second slot re-announcing).
#[tokio::test]
async fn a_probe_that_finds_the_provider_already_back_announces_nothing() {
    let primary = Fake::new("primary", &[Step::FailCold])
        .recovers(&[Recovery::Recovered, Recovery::Recovered])
        .shut(RecoverySlot::OneShot);
    let backup = Fake::new("backup", &[Step::Serve]);
    let rig = rig(vec![Arc::clone(&primary), backup], ChainPolicy::stt_defaults());

    // The batch path fails, so a batch probe is spawned -- and parks on its closed gate.
    let _ = run_once(&rig, RecoverySlot::OneShot).await;
    until("the batch probe to park", || primary.probe_count(RecoverySlot::OneShot) == 1).await;
    assert!(!rig.chain.is_available(0));

    // The streaming path probes the same provider and recovers it: that is the one transition.
    let _ = run_once(&rig, RecoverySlot::Streaming).await;
    until("the streaming probe to bring the primary back", || rig.chain.is_available(0)).await;

    // Now release the parked batch probe. It also succeeds, against a flag already true.
    primary.gate[RecoverySlot::OneShot as usize].add_permits(1);
    until("the batch probe to finish", || primary.probes_done.load(Ordering::SeqCst) == 2).await;

    assert_eq!(rig.rec.transitions(), vec![(0, false), (0, true)]);
}

// Catches: closing the consumer's half without stopping the chain task and its member; an ignored
// stop is aborted and reported `TimedOut`.
#[allow(clippy::disallowed_methods, reason = "S0: existing test harness live resources")]
#[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
#[tokio::test]
async fn closing_a_channel_source_stops_the_task_and_waits_for_it() {
    use std::sync::Arc;
    #[allow(clippy::disallowed_types, reason = "S0: existing test harness live resources")]
    use std::sync::atomic::{AtomicBool, Ordering};
    let clock = Arc::new(lk_agents_abi::VirtualClock::new());
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<u8, ProviderError>>(4);
    let (mut stop, done, chain) = crate::fallback::channel::ChainTask::pair(clock.clone());
    let member_closed = Arc::new(AtomicBool::new(false));
    let closed_flag = Arc::clone(&member_closed);
    let filling = tokio::spawn(async move {
        let _tx = tx;
        // The task's loop: serve until the consumer stops it, then close the member.
        let _ = stop.changed().await;
        closed_flag.store(true, Ordering::SeqCst);
        let _ = done.send(());
    });
    let source: Box<dyn StreamSource<u8>> = Box::new(crate::fallback::channel::ChannelSource {
        rx,
        task: Some(filling.abort_handle().into()),
        chain,
    });
    let closed = within(
        "the close",
        source.close(clock.now().saturating_add(std::time::Duration::from_secs(1))),
    )
    .await;
    assert_eq!(closed, lk_agents_abi::Closed::Clean);
    assert!(
        member_closed.load(Ordering::SeqCst),
        "the task closed its member before reporting done"
    );
}

#[allow(clippy::disallowed_methods, reason = "S0: existing test harness live resources")]
#[tokio::test]
async fn a_task_that_ignores_the_stop_is_aborted_and_reported_timed_out() {
    use std::sync::Arc;
    let clock = Arc::new(lk_agents_abi::VirtualClock::new());
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<u8, ProviderError>>(4);
    let (_stop, _done, chain) = crate::fallback::channel::ChainTask::pair(clock.clone());
    let filling = tokio::spawn(async move {
        let _tx = tx;
        std::future::pending::<()>().await;
    });
    let handle = filling.abort_handle();
    let source: Box<dyn StreamSource<u8>> =
        Box::new(crate::fallback::channel::ChannelSource { rx, task: Some(handle.into()), chain });
    // A deadline already past gives the task one poll and no more.
    let closed = within("the close", source.close(lk_agents_abi::Ts::ZERO)).await;
    assert_eq!(closed, lk_agents_abi::Closed::TimedOut);
    assert!(filling.await.unwrap_err().is_cancelled(), "the wedged task was aborted");
}

#[tokio::test]
async fn cancellation_is_not_a_backend_health_penalty() {
    let primary = Fake::new("primary", &[Step::Serve]);
    let rig = rig(vec![primary.clone()], ChainPolicy::llm_defaults());
    let out: Result<ChainOutcome<()>, _> = rig
        .chain
        .run(&"req".to_string(), &caller(), RecoverySlot::OneShot, &mut (), |_, _, _, _| {
            Box::pin(async {
                Attempt::Failed {
                    error: ProviderError::Local(ApiError::status("cancelled", 499, None, None)),
                    produced_output: false,
                }
            })
        })
        .await;
    assert!(out.is_err());
    assert!(rig.chain.is_available(0));
    assert!(rig.rec.transitions().is_empty());
    tokio::task::yield_now().await;
    assert_eq!(
        primary.probes_entered.load(Ordering::SeqCst),
        0,
        "provider cancellation must not launch recovery work"
    );
}

#[tokio::test]
async fn every_modality_fails_after_partial_output_without_replay() {
    for policy in
        [ChainPolicy::llm_defaults(), ChainPolicy::tts_defaults(), ChainPolicy::stt_defaults()]
    {
        let backup = Fake::new("backup", &[Step::Serve]);
        let rig = rig(vec![Fake::new("primary", &[Step::FailHot]), backup.clone()], policy);
        let (out, drained) = run_once(&rig, RecoverySlot::Streaming).await;
        assert!(out.is_err(), "partial output must retain its failure");
        assert_eq!(drained, vec![0]);
        assert!(backup.opened.lock().expect("opened").is_empty());
    }
}

#[tokio::test]
async fn recovery_probes_inherit_the_remaining_operation_deadline() {
    let primary = Fake::new("primary", &[Step::FailCold]);
    let backup = Fake::new("backup", &[Step::Serve]);
    let rig = rig(vec![primary.clone(), backup.clone()], ChainPolicy::tts_defaults());
    let clock = rig.clock.clone();
    let caller = ApiConnectOptions { total_timeout: Some(9.0), ..caller() };
    let out = rig
        .chain
        .run(
            &"request".to_owned(),
            &caller,
            RecoverySlot::OneShot,
            &mut (),
            move |index, _, _, _| {
                let clock = clock.clone();
                Box::pin(async move {
                    if index == 0 {
                        clock.advance(Duration::from_secs(6));
                        Attempt::Failed { error: boom(), produced_output: false }
                    } else {
                        Attempt::Served(())
                    }
                }) as BoxFuture<'_, Attempt<()>>
            },
        )
        .await;
    assert!(out.is_ok());
    until("the probe to open", || primary.probes_entered.load(Ordering::SeqCst) > 0).await;
    assert_eq!(backup.opened.lock().unwrap()[0].total_timeout, Some(3.0));
    assert_eq!(primary.probes.lock().unwrap()[0].1.total_timeout, Some(3.0));
    rig.chain.close().await;
}

#[tokio::test]
async fn a_recovery_result_after_its_request_deadline_cannot_restore_health() {
    let primary = Fake::new("primary", &[Step::FailCold])
        .recovers(&[Recovery::Recovered])
        .shut(RecoverySlot::OneShot);
    let backup = Fake::new("backup", &[Step::Serve]);
    let rig = rig(vec![primary.clone(), backup], ChainPolicy::tts_defaults());
    let caller = ApiConnectOptions { total_timeout: Some(1.0), ..caller() };
    rig.chain
        .run(&"request".to_owned(), &caller, RecoverySlot::OneShot, &mut (), |index, _, _, _| {
            Box::pin(async move {
                if index == 0 {
                    Attempt::Failed { error: boom(), produced_output: false }
                } else {
                    Attempt::Served(())
                }
            }) as BoxFuture<'_, Attempt<()>>
        })
        .await
        .unwrap();
    until("the probe to wait", || primary.probes_entered.load(Ordering::SeqCst) > 0).await;
    rig.clock.advance(Duration::from_secs(2));
    primary.gate[0].add_permits(1);
    until("the late probe to finish", || primary.probes_done.load(Ordering::SeqCst) > 0).await;
    assert!(!rig.chain.is_available(0), "an expired operation cannot change provider health");
    rig.chain.close().await;
}

struct HeldSpawner {
    runtime: TokioSpawner,
    queued: tokio::sync::mpsc::Sender<()>,
    release: Arc<Semaphore>,
}

impl Spawner for HeldSpawner {
    fn spawn(&self, future: BoxFuture<'static, ()>) -> lk_agents_abi::AbortHandle {
        self.queued.try_send(()).expect("one queued provider attempt");
        let release = self.release.clone();
        self.runtime.spawn(Box::pin(async move {
            let _permit = release.acquire().await.expect("held task released");
            future.await;
        }))
    }
}

#[tokio::test]
async fn a_queued_provider_attempt_does_not_open_after_its_operation_expires_or_is_cancelled() {
    use crate::fallback::chain::ChainCall;
    use std::task::Poll;

    for cancel_before_execution in [false, true] {
        let backend = Fake::new("held", &[Step::Serve]);
        let clock = Arc::new(VirtualClock::new());
        let release = Arc::new(Semaphore::new(0));
        let (queued, mut scheduled) = tokio::sync::mpsc::channel(1);
        let spawner = Arc::new(HeldSpawner {
            runtime: TokioSpawner::current(),
            queued,
            release: release.clone(),
        });
        let chain = FallbackChain::new(
            vec![backend.clone()],
            ChainPolicy::llm_defaults(),
            clock.clone(),
            spawner,
            Arc::new(Recorder::default()),
        )
        .unwrap();
        let (stop, stopped) = tokio::sync::watch::channel(false);
        let request = "request".to_owned();
        let mut state = ();
        let mut run = Box::pin(chain.run_cancellable(
            &request,
            ChainCall {
                conn_options: ApiConnectOptions { total_timeout: Some(1.0), ..caller() },
                accepted_at: clock.now(),
                slot: RecoverySlot::OneShot,
            },
            &mut state,
            stopped,
            |_, _, _, _, _| Box::pin(async { Attempt::Served(()) }),
        ));
        // Accept and authorize the attempt, then hold its actual provider task in the executor.
        std::future::poll_fn(|cx| {
            assert!(run.as_mut().poll(cx).is_pending());
            Poll::Ready(())
        })
        .await;
        scheduled.recv().await.expect("the inner attempt was scheduled");
        if cancel_before_execution {
            stop.send(true).unwrap();
        } else {
            clock.advance(Duration::from_secs(2));
        }
        release.add_permits(1);
        let error = run.await.unwrap_err();
        assert!(
            backend.opened.lock().unwrap().is_empty(),
            "queued task performed provider I/O after revocation"
        );
        assert_eq!(
            error.api_kind(),
            Some(if cancel_before_execution {
                lk_agents_abi::errors::ApiErrorKind::Cancelled
            } else {
                lk_agents_abi::errors::ApiErrorKind::Timeout
            }),
        );
        chain.close().await;
    }
}

#[tokio::test]
async fn repeated_backend_entries_share_one_recovery_probe() {
    let primary = Fake::new("primary", &[Step::FailCold]).shut(RecoverySlot::OneShot);
    let backup = Fake::new("backup", &[Step::Serve]);
    let rig = rig(vec![primary.clone(), primary.clone(), backup], ChainPolicy::llm_defaults());
    let _ = run_once(&rig, RecoverySlot::OneShot).await;
    until("the primary probe to park", || primary.probe_count(RecoverySlot::OneShot) == 1).await;
    let _ = run_once(&rig, RecoverySlot::OneShot).await;
    tokio::task::yield_now().await;
    assert_eq!(
        primary.probe_count(RecoverySlot::OneShot),
        1,
        "a backend has one probe per recovery slot, independent of list occurrences"
    );
    primary.gate[RecoverySlot::OneShot as usize].add_permits(1);
    rig.chain.close().await;
}
