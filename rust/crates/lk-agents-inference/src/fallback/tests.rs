//! `FallbackChain` as a state machine: a ports recorder, a [`VirtualClock`], and a scripted
//! [`Backend`]. Probes are real tasks, gated by semaphores and observed through [`until`].

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use lk_agents_abi::StreamSource;
use lk_agents_abi::{BoxFuture, Spawner, TokioSpawner};
use lk_agents_abi::{Clock, VirtualClock};
use lk_agents_abi::errors::{ApiError, ProviderError};
use lk_agents_abi::ApiConnectOptions;
use tokio::sync::Semaphore;

use crate::fallback::chain::{
    Attempt, AvailabilityChanged, Backend, ChainObserver, ChainOutcome, FallbackChain, Recovery,
    RecoverySlot,
};
use crate::fallback::policy::ChainPolicy;

const LIMIT: Duration = Duration::from_secs(5);

async fn within<F: std::future::Future>(what: &str, fut: F) -> F::Output {
    match tokio::time::timeout(LIMIT, fut).await {
        Ok(v) => v,
        Err(_) => panic!("timed out after {LIMIT:?} waiting for {what}"),
    }
}

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
    events: Mutex<Vec<AvailabilityChanged>>,
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
    script: Mutex<VecDeque<Step>>,
    opened: Mutex<Vec<ApiConnectOptions>>,
    probes: Mutex<Vec<(RecoverySlot, ApiConnectOptions)>>,
    probe_verdict: Mutex<VecDeque<Recovery>>,
    /// A probe on slot `i` waits here. Starts saturated, so probes complete immediately unless a
    /// test closes the gate.
    gate: [Arc<Semaphore>; 2],
    probes_entered: AtomicUsize,
    /// Incremented as the last statement of `recover`, so a test can `until` on "the probe
    /// finished" without racing the availability event.
    probes_done: AtomicUsize,
}

impl Fake {
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

fn rig(backends: Vec<Arc<Fake>>, policy: ChainPolicy) -> Rig {
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
    ApiConnectOptions { max_retry: 3, retry_interval: 2.0, timeout: 10.0 }
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
    let primary =
        Fake::new("primary", &[Step::FailCold, Step::Serve]).recovers(&[Recovery::Recovered]);
    let backup = Fake::new("backup", &[Step::Serve]);
    let rig = rig(vec![Arc::clone(&primary), backup], ChainPolicy::llm_defaults());

    let (out, drained) = run_once(&rig, RecoverySlot::OneShot).await;
    assert_eq!(out.expect("the backup serves"), ChainOutcome::Served(1));
    assert_eq!(drained, vec![0, 1], "the primary is tried first, then the backup");
    assert!(!rig.chain.is_available(0));

    until("the primary's probe to succeed", || rig.chain.is_available(0)).await;

    let (out, drained) = run_once(&rig, RecoverySlot::OneShot).await;
    assert_eq!(out.expect("the primary serves again"), ChainOutcome::Served(0));
    assert_eq!(drained, vec![0], "the backup must not be reached once the primary is back");
    assert_eq!(rig.chain.active_index(), 0);
    assert_eq!(rig.rec.transitions(), vec![(0, false), (0, true)]);
}

// Catches, per policy, a failure after output: retrying an LLM after it emitted text (`llm:300-306`
// re-raises); `retry_on_chunk_sent=True` doing nothing; TTS's progress guard modelled as `Abort`
// (`tts:293-297` returns success); STT guarding on progress at all.
#[tokio::test]
async fn a_failure_after_output_follows_the_policys_after_output() {
    let llm = ChainPolicy::llm_defaults();
    for (policy, slot, want, want_drained) in [
        (llm, RecoverySlot::OneShot, None, vec![0]),
        (llm.with_retry_on_chunk_sent(true), RecoverySlot::OneShot, Some(ChainOutcome::Served(1)), vec![0, 1]),
        (ChainPolicy::tts_defaults(), RecoverySlot::Streaming, Some(ChainOutcome::Truncated), vec![0]),
        (ChainPolicy::stt_defaults(), RecoverySlot::Streaming, Some(ChainOutcome::Served(1)), vec![0, 1]),
    ] {
        let backup = Fake::new("backup", &[Step::Serve]);
        let rig = rig(vec![Fake::new("primary", &[Step::FailHot]), Arc::clone(&backup)], policy);
        let (out, drained) = run_once(&rig, slot).await;
        assert_eq!(out.ok(), want, "an error is the provider's failure, not a silent fallback");
        assert_eq!(drained, want_drained);
        assert_eq!(backup.opened.lock().expect("opened").len(), want_drained.len() - 1);
        if want.is_none() {
            assert_eq!(rig.chain.active_index(), 0, "metrics still belong to the provider that emitted");
        }
    }
}

// Catches: a caller's `conn_options` leaking to the wrapped provider; also pins the probe's hard 0.
#[tokio::test]
async fn the_rewritten_conn_options_reach_the_provider_on_both_paths() {
    let primary = Fake::new("primary", &[Step::FailCold]).recovers(&[Recovery::NotYet]);
    let rig = rig(vec![Arc::clone(&primary)], ChainPolicy::tts_defaults());

    let _ = run_once(&rig, RecoverySlot::Streaming).await;
    let opened = primary.opened.lock().expect("opened").clone();
    assert_eq!(opened.len(), 1);
    assert_eq!(opened[0].max_retry, 2, "TTS allows 2 internal retries, not the caller's 3");
    assert_eq!(opened[0].timeout, caller().timeout, "TTS keeps the caller's timeout");

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
            |i, _b, s, d| {
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
    assert!(err.message().contains("all backends failed (a)"), "{}", err.message());
    assert!(
        err.message().contains("0.000 seconds"),
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
#[tokio::test]
async fn closing_a_channel_source_stops_the_task_and_waits_for_it() {
    use std::sync::Arc;
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
    let source: Box<dyn StreamSource<u8>> =
        Box::new(crate::fallback::channel::ChannelSource { rx, task: Some(filling.abort_handle().into()), chain });
    let closed = within("the close", source.close(clock.now().saturating_add(std::time::Duration::from_secs(1)))).await;
    assert_eq!(closed, lk_agents_abi::Closed::Clean);
    assert!(member_closed.load(Ordering::SeqCst), "the task closed its member before reporting done");
}

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
