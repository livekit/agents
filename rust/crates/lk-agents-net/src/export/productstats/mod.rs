use std::sync::Arc;
use std::time::{Duration, SystemTime};

use lk_agents_abi::errors::ApiError;
use lk_agents_abi::productstats::runner::{Runner, RunnerCmd};
use lk_agents_abi::productstats::{ProductStat, ProductStatsResource, RuntimeName, ThreadStorage};
use lk_agents_abi::telemetry::decl::{self, Attrs as _, Label, MetricDecl, Number};
use lk_agents_abi::telemetry::record::KeyValues;
use lk_agents_abi::{Clock, OwnedTask, Spawner, SpawnerExt as _, Ts, before, timeout};
use parking_lot::{Mutex, RwLock};

use super::otlp::cloud::{self, PostError};
use super::otlp::config::Temporality;
use super::otlp::encode;
use super::otlp::metrics::Aggregator;
use super::otlp::proto::metrics::v1::Metric;
use crate::signal;

// LiveKit's product statistics: metrics only, aggregated with delta temporality and sent once a
// minute to LiveKit Cloud's collector. Independent of the customer's observability export: its
// own aggregation, timer, destination and failures, so neither refuses or disables the other.

/// How often a window is exported.
pub const INTERVAL: Duration = Duration::from_secs(60);
const REQUEST_TIMEOUT: Duration = Duration::from_secs(10);
/// The collector's route, under the Cloud host.
pub const PATH: &str = "/observability/productstats/agents/metrics/otlp/v0";

/// LiveKit Cloud's collector and the project key the export is signed with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Destination {
    base_url: String,
    api_key: String,
    api_secret: String,
}

impl Destination {
    /// The collector of the Cloud project at `livekit_url`; `None` off LiveKit Cloud or without a
    /// key, where product statistics go nowhere.
    pub fn cloud(livekit_url: &str, api_key: &str, api_secret: &str) -> Option<Self> {
        if api_key.is_empty() || api_secret.is_empty() {
            return None;
        }
        let base_url = crate::strings::observability_url(livekit_url, None)?;
        Some(Destination { base_url, api_key: api_key.into(), api_secret: api_secret.into() })
    }

    /// The export's bearer: the same `write` grant as the customer's observability export.
    fn bearer(&self) -> Result<String, ApiError> {
        crate::token::observability_bearer(&self.api_key, &self.api_secret)
    }

    async fn post(&self, body: encode::Body) -> Result<(), PostError> {
        let client = cloud::client().await.map_err(|error| PostError::Setup(error.to_string()))?;
        let bearer = self.bearer().map_err(|error| PostError::Setup(error.to_string()))?;
        let url = format!("{}{PATH}", self.base_url.trim_end_matches('/'));
        let response = body.attach(client.post(url).bearer_auth(bearer)).send().await?;
        cloud::answer(response, false).await
    }
}

/// Where the next window goes.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Route {
    /// No credential yet: the window waits for one.
    Waiting,
    Cloud(Destination),
    /// Off LiveKit Cloud: nothing is sent.
    Nowhere,
}

struct State {
    aggregator: Aggregator,
    resource: ProductStatsResource,
    route: Route,
    /// The last window collected while waiting for a credential: one, replaced by the next.
    held: Option<Vec<Metric>>,
}

impl State {
    /// The window to send now, and where: the held window first.
    fn take(&mut self, now: u64) -> Option<(Destination, Vec<Metric>, KeyValues)> {
        let mut window = self.aggregator.collect(now, &KeyValues::new());
        match &self.route {
            Route::Nowhere => None,
            Route::Waiting => {
                if !window.is_empty() {
                    self.held = Some(window);
                }
                None
            }
            Route::Cloud(destination) => {
                if let Some(mut held) = self.held.take() {
                    held.append(&mut window);
                    window = held;
                }
                if window.is_empty() {
                    return None;
                }
                let mut resource = KeyValues::new();
                self.resource.write(&mut resource);
                Some((destination.clone(), window, resource))
            }
        }
    }
}

struct Shared {
    state: Mutex<State>,
    clock: Arc<dyn Clock>,
    /// Exports in progress.
    in_flight: signal::Producer<usize>,
}

/// The process's product statistics.
pub struct ProductStats {
    shared: Arc<Shared>,
    task: Mutex<Option<OwnedTask<()>>>,
}

impl ProductStats {
    /// Starts the one-minute export on `spawner`, timed by `clock`; nothing is sent before
    /// [`ProductStats::configure`].
    pub fn start(
        spawner: &dyn Spawner,
        clock: Arc<dyn Clock>,
        resource: ProductStatsResource,
    ) -> Self {
        let now = wall_nanos(&*clock);
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                aggregator: Aggregator::new(Temporality::Delta, now),
                resource,
                route: Route::Waiting,
                held: None,
            }),
            clock,
            in_flight: signal::signal(0).0,
        });
        let task = spawner.spawn(run(shared.clone()));
        ProductStats { shared, task: Mutex::new(Some(task)) }
    }

    /// Where later windows go: the most recent configuration wins; `None` sends nothing.
    pub fn configure(&self, destination: Option<Destination>) {
        let route = destination.map_or(Route::Nowhere, Route::Cloud);
        let mut state = self.shared.state.lock();
        if route == Route::Nowhere {
            state.held = None;
        }
        state.route = route;
    }

    /// Sends the current window. False when `limit` passed first.
    pub async fn flush(&self, limit: Duration) -> bool {
        let shared = &*self.shared;
        let deadline = shared.clock.now().saturating_add(limit);
        shared.export(Some(deadline)).await && shared.settled(deadline).await
    }

    /// The process-exit flush within `limit`; the periodic export stops.
    pub async fn shutdown(&self, limit: Duration) {
        self.flush(limit).await;
        drop(self.task.lock().take());
    }
}

impl ProductStats {
    /// Adds one measurement to the current window; one the aggregation refuses (its series
    /// limit) is dropped.
    pub fn record(&self, stat: ProductStat) {
        self.shared.state.lock().aggregator.record(&stat.into_measurement());
    }
}

async fn run(shared: Arc<Shared>) {
    let clock = shared.clock.clone();
    loop {
        clock.wait_until(clock.now().saturating_add(INTERVAL)).await;
        shared.export(None).await;
    }
}

fn wall_nanos(clock: &dyn Clock) -> u64 {
    let since = clock.wall().duration_since(SystemTime::UNIX_EPOCH);
    since.map_or(0, |d| d.as_nanos().min(u64::MAX as u128) as u64)
}

/// Counts an export in progress until dropped, cancellation included.
struct InFlight<'a>(&'a Shared);

impl Drop for InFlight<'_> {
    fn drop(&mut self) {
        self.0.in_flight.modify(|n| *n -= 1);
    }
}

impl Shared {
    /// Sends the window, if any. False when `deadline` cut it short.
    async fn export(&self, deadline: Option<Ts>) -> bool {
        let now = wall_nanos(&*self.clock);
        let Some((destination, metrics, resource)) = self.state.lock().take(now) else {
            return true;
        };
        self.in_flight.modify(|n| *n += 1);
        let _in_flight = InFlight(self);
        let limit = match deadline {
            None => REQUEST_TIMEOUT,
            Some(at) => at.saturating_duration_since(self.clock.now()).min(REQUEST_TIMEOUT),
        };
        let request = encode::metrics_request(encode::resource(&resource), metrics);
        let body = encode::body(&request, true);
        match timeout(&*self.clock, limit, destination.post(body)).await {
            Ok(Ok(())) => true,
            Ok(Err(error)) => {
                log::debug!(target: lk_agents_abi::LOG_TARGET, "product statistics export failed: {error}");
                true
            }
            Err(_) => deadline.is_none_or(|at| self.clock.now() < at),
        }
    }

    async fn settled(&self, deadline: Ts) -> bool {
        let in_flight = self.in_flight.cell();
        before(&*self.clock, Some(deadline), in_flight.wait_for(|n| *n == 0)).await.is_ok()
    }
}

/// The process's product statistics exporter, which every `measure` and `record` writes to.
/// The host's telemetry installation installs it once ([`install`]), unless they are opted out of.
static EXPORTER: RwLock<Option<Arc<ProductStats>>> = RwLock::new(None);

/// Installs the exporter `start` builds as the process's, unless one is installed or product
/// statistics are opted out of ([`crate::env::product_stats_enabled`]).
pub fn install(start: impl FnOnce() -> ProductStats) -> Option<Arc<ProductStats>> {
    if !crate::env::product_stats_enabled() {
        return None;
    }
    let mut slot = EXPORTER.write();
    if let Some(exporter) = &*slot {
        return Some(exporter.clone());
    }
    let exporter = Arc::new(start());
    *slot = Some(exporter.clone());
    Some(exporter)
}

/// Removes the process's exporter and returns it for its last flush: measurements are dropped
/// until the next install.
pub fn uninstall() -> Option<Arc<ProductStats>> {
    EXPORTER.write().take()
}

/// A forked child abandons its parent's exporter without dropping it: its task belongs to the
/// parent's runtime.
pub fn forget() {
    std::mem::forget(EXPORTER.write().take());
}

/// Signs later exports with the key of the server or job at `livekit_url`; off LiveKit Cloud
/// nothing is sent.
pub fn configure_destination(livekit_url: &str, api_key: &str, api_secret: &str) {
    if let Some(exporter) = EXPORTER.read().clone() {
        exporter.configure(Destination::cloud(livekit_url, api_key, api_secret));
    }
}

/// Records one measurement of `M` under `labels`. Only a product statistic records here:
///
/// ```
/// use lk_agents_abi::productstats::testkit::TestDuration;
/// lk_agents_net::export::productstats::measure(&TestDuration, 1.0);
/// ```
///
/// A customer instrument does not compile:
///
/// ```compile_fail,E0271
/// use lk_agents_abi::telemetry::instruments::WorkerProcessReadyDuration;
/// lk_agents_net::export::productstats::measure(&WorkerProcessReadyDuration, 1.0);
/// ```
pub fn measure<M: MetricDecl<Audience = decl::ProductStats>>(labels: &M, value: impl Into<Number>) {
    record(ProductStat::of(labels, value));
}

/// Records a measurement made elsewhere (a sans-I/O machine's, several of one usage): dropped
/// when no exporter is installed.
pub fn record(stat: ProductStat) {
    #[cfg(any(test, feature = "testkit"))]
    let Some(stat) = testkit::capture(stat) else { return };
    if let Some(exporter) = EXPORTER.read().clone() {
        exporter.record(stat);
    }
}

/// The process opened a thread storage of this kind: later exports name it. A process that opens
/// several names the last one.
pub fn storage_opened(storage: ThreadStorage) {
    if let Some(exporter) = EXPORTER.read().clone() {
        exporter.shared.state.lock().resource.storage = storage;
    }
}

/// The command that started this process, which a worker hands its job processes; none when no
/// exporter is installed, where nothing reads it.
pub fn runner_cmd() -> RunnerCmd {
    let exporter = EXPORTER.read().clone();
    exporter
        .map_or(RunnerCmd::NoCommand, |exporter| exporter.shared.state.lock().resource.runner.cmd)
}

/// The process's resource: the framework, the host's runtime, the platform and the runner, which
/// is derived from the command that started the process, its platform and the `CI` and
/// `LK_HOSTED_AGENTS` variables (no host name, user name or path is read). No storage is opened
/// yet.
pub fn resource(
    sdk_version: &str,
    runtime_name: RuntimeName,
    runtime_version: &str,
    runner_cmd: RunnerCmd,
) -> ProductStatsResource {
    let var = |name: &str| std::env::var(name).ok();
    ProductStatsResource {
        service: Label(lk_agents_abi::telemetry::record::FRAMEWORK.to_owned()),
        framework_version: Label(sdk_version.to_owned()),
        runtime_name,
        runtime_version: Label(runtime_version.to_owned()),
        os: Label(std::env::consts::OS.to_owned()),
        arch: Label(std::env::consts::ARCH.to_owned()),
        runner: Runner::detect(var, std::env::consts::OS, runner_cmd),
        storage: ThreadStorage::None,
    }
}

/// A test's own product statistics: what its thread records, instead of the process's exporter.
/// Tests of one binary run in parallel on one process, so an installed exporter would mix theirs.
#[cfg(any(test, feature = "testkit"))]
pub mod testkit {
    use std::cell::RefCell;

    use lk_agents_abi::productstats::ProductStat;
    use lk_agents_abi::productstats::testkit::{Stat, stat};

    thread_local! {
        static CAPTURED: RefCell<Option<Vec<ProductStat>>> = const { RefCell::new(None) };
    }

    /// Captures what this thread records until dropped: a test on a current-thread runtime sees
    /// its own statistics and no other test's.
    pub struct Capture(());

    impl Capture {
        pub fn start() -> Self {
            CAPTURED.with(|c| *c.borrow_mut() = Some(Vec::new()));
            Capture(())
        }

        /// What was recorded since the last call.
        pub fn take(&self) -> Vec<Stat> {
            let taken = CAPTURED.with(|c| c.borrow_mut().as_mut().map(std::mem::take));
            taken.unwrap_or_default().into_iter().map(stat).collect()
        }
    }

    impl Drop for Capture {
        fn drop(&mut self) {
            CAPTURED.with(|c| *c.borrow_mut() = None);
        }
    }

    /// `None` when the thread's capture took the measurement.
    pub(super) fn capture(stat: ProductStat) -> Option<ProductStat> {
        CAPTURED.with(|c| match c.borrow_mut().as_mut() {
            Some(captured) => {
                captured.push(stat);
                None
            }
            None => Some(stat),
        })
    }
}

#[cfg(test)]
#[path = "productstats_tests.rs"]
mod tests;
