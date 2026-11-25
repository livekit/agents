use std::sync::Arc;
use std::time::{Duration, SystemTime};

use lk_agents_abi::errors::ApiError;
use lk_agents_abi::productstats::ProductStatsResource;
use lk_agents_abi::telemetry::decl::{Attrs as _, Label};
use lk_agents_abi::telemetry::record::{KeyValues, Measurement};
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

impl lk_agents_telemetry::productstats::Sink for ProductStats {
    /// A measurement the aggregation refuses (its series limit) is dropped.
    fn record(&self, measurement: Measurement) {
        self.shared.state.lock().aggregator.record(&measurement);
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

/// The process's product statistics once the host installed them.
static PROCESS: RwLock<Option<Arc<ProductStats>>> = RwLock::new(None);

/// Installs the process's product statistics `start` builds, unless installed or opted out of
/// ([`crate::env::product_stats_enabled`]): every measurement point records to them from then on.
pub fn install(start: impl FnOnce() -> ProductStats) -> Option<Arc<ProductStats>> {
    if !crate::env::product_stats_enabled() {
        return None;
    }
    let mut slot = PROCESS.write();
    if let Some(stats) = &*slot {
        return Some(stats.clone());
    }
    let stats = Arc::new(start());
    lk_agents_telemetry::productstats::install(stats.clone());
    *slot = Some(stats.clone());
    Some(stats)
}

/// The installed product statistics.
pub fn get() -> Option<Arc<ProductStats>> {
    PROCESS.read().clone()
}

/// Removes the installed product statistics, so a later install starts new ones.
pub fn take() -> Option<Arc<ProductStats>> {
    lk_agents_telemetry::productstats::uninstall();
    PROCESS.write().take()
}

/// A forked child abandons its parent's product statistics without dropping them: their task
/// belongs to the parent's runtime.
pub fn forget() {
    lk_agents_telemetry::productstats::forget();
    std::mem::forget(PROCESS.write().take());
}

/// The server or job at `livekit_url` signs later exports with its key.
pub fn configure(livekit_url: &str, api_key: &str, api_secret: &str) {
    if let Some(stats) = get() {
        stats.configure(Destination::cloud(livekit_url, api_key, api_secret));
    }
}

/// The process's resource: the framework, the host's runtime, the platform and the runner.
pub fn resource(sdk_version: &str, python_version: Option<&str>) -> ProductStatsResource {
    ProductStatsResource {
        service: Label(lk_agents_abi::telemetry::record::FRAMEWORK.to_owned()),
        framework_version: Label(sdk_version.to_owned()),
        python_version: python_version.map(|v| Label(v.to_owned())),
        os: Label(std::env::consts::OS.to_owned()),
        arch: Label(std::env::consts::ARCH.to_owned()),
        runner: crate::env::runner(),
    }
}

#[cfg(test)]
#[path = "productstats_tests.rs"]
mod tests;
