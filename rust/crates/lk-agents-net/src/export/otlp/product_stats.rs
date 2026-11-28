use std::time::Duration;

use lk_agents_abi::errors::ApiError;
use lk_agents_abi::productstats::runner::{Runner, RunnerCmd};
use lk_agents_abi::productstats::{ProductStat, ProductStatsResource, RuntimeName, ThreadStorage};
use lk_agents_abi::telemetry::decl::{Attrs as _, BoundedLabel};
use lk_agents_abi::telemetry::record::KeyValues;

use super::cloud::{self, PostError};
use super::config::Temporality;
use super::encode;
use super::metrics::Aggregator;
use super::proto::metrics::v1::Metric;

// LiveKit's product statistics, the exporter's third destination: metrics only, aggregated with
// delta temporality and sent once a minute to LiveKit Cloud's collector under its own resource.
// It takes `ProductStat`s and nothing else, belongs to no job, and is gated by no project's
// recording decision.

/// How often a window is exported.
pub const INTERVAL: Duration = Duration::from_secs(60);
/// The collector's route, under the Cloud host.
pub const PATH: &str = "/observability/productstats/agents/metrics/otlp/v0";

/// LiveKit Cloud's collector and the project key the export is signed with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Collector {
    pub(super) base_url: String,
    pub(super) api_key: String,
    pub(super) api_secret: String,
}

impl Collector {
    /// The collector of the Cloud project at `livekit_url`; `None` off LiveKit Cloud or without a
    /// key, where product statistics go nowhere.
    pub fn cloud(livekit_url: &str, api_key: &str, api_secret: &str) -> Option<Self> {
        if api_key.is_empty() || api_secret.is_empty() {
            return None;
        }
        let base_url = crate::strings::observability_url(livekit_url, None)?;
        Some(Collector { base_url, api_key: api_key.into(), api_secret: api_secret.into() })
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
    Cloud(Collector),
    /// Off LiveKit Cloud: nothing is sent.
    Nowhere,
}

/// One window on its way to the collector.
pub(crate) struct Window {
    collector: Collector,
    resource: KeyValues,
    metrics: Vec<Metric>,
}

impl Window {
    /// A failed export is dropped, and is no failure of the customer's.
    pub async fn export(self) {
        let request = encode::metrics_request(encode::resource(&self.resource), self.metrics);
        if let Err(error) = self.collector.post(encode::body(&request, true)).await {
            log::debug!(target: lk_agents_abi::LOG_TARGET, "product statistics export failed: {error}");
        }
    }
}

pub(crate) struct ProductStatsDestination {
    aggregator: Aggregator,
    pub resource: ProductStatsResource,
    route: Route,
    /// The last window collected while waiting for a credential: one, replaced by the next.
    held: Option<Vec<Metric>>,
}

impl ProductStatsDestination {
    pub fn new(resource: ProductStatsResource, now: u64) -> Self {
        ProductStatsDestination {
            aggregator: Aggregator::new(Temporality::Delta, now),
            resource,
            route: Route::Waiting,
            held: None,
        }
    }

    /// A measurement the aggregation refuses (its series limit) is dropped.
    pub fn record(&mut self, stat: ProductStat) {
        self.aggregator.record(&stat.into_measurement());
    }

    /// Where later windows go: the most recent configuration wins; `None` sends nothing.
    pub fn configure(&mut self, collector: Option<Collector>) {
        self.route = collector.map_or(Route::Nowhere, Route::Cloud);
        if self.route == Route::Nowhere {
            self.held = None;
        }
    }

    /// The window to send now: the held window first. An empty window is no request.
    pub fn take(&mut self, now: u64) -> Option<Window> {
        let mut metrics = self.aggregator.collect(now, &KeyValues::new());
        match &self.route {
            Route::Nowhere => None,
            Route::Waiting => {
                if !metrics.is_empty() {
                    self.held = Some(metrics);
                }
                None
            }
            Route::Cloud(collector) => {
                if let Some(mut held) = self.held.take() {
                    held.append(&mut metrics);
                    metrics = held;
                }
                if metrics.is_empty() {
                    return None;
                }
                let mut resource = KeyValues::new();
                self.resource.write(&mut resource);
                Some(Window { collector: collector.clone(), resource, metrics })
            }
        }
    }
}

/// The destination's own resource, never the customer's: the framework, the host's runtime, the
/// platform and the runner, which is derived from the command that started the process, its
/// platform and the `CI` and `LK_HOSTED_AGENTS` variables (no host name, user name or path is
/// read). No storage is opened yet.
pub fn resource(
    sdk_version: &str,
    runtime_name: RuntimeName,
    runtime_version: &str,
    runner_cmd: RunnerCmd,
) -> ProductStatsResource {
    let var = |name: &str| std::env::var(name).ok();
    ProductStatsResource {
        service: BoundedLabel::new(lk_agents_abi::telemetry::record::FRAMEWORK),
        framework_version: BoundedLabel::new(sdk_version),
        runtime_name,
        runtime_version: BoundedLabel::new(runtime_version),
        os: BoundedLabel::new(std::env::consts::OS),
        arch: BoundedLabel::new(std::env::consts::ARCH),
        runner: Runner::detect(var, std::env::consts::OS, runner_cmd),
        storage: ThreadStorage::None,
    }
}

#[cfg(test)]
#[path = "product_stats_tests.rs"]
mod tests;
