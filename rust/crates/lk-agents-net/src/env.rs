use std::sync::Arc;

use lk_agents_abi::telemetry::runner::{Launch, Runner};
use lk_agents_abi::{Clock, Spawner};

use crate::http::GatewayHttp;

/// Why the process cannot lend a runtime-bound handle. Nothing retries it: a client that cannot
/// get one fails for good.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ProcessError {
    #[error("the native runtime failed to start: {0}")]
    Start(String),
    #[error(
        "livekit.agents._core was inherited across a fork. The Rust core cannot be used in a \
         forked child without re-initialising it: call _core.init(role) in the child."
    )]
    Inherited,
    /// A Rust caller with no runtime on its thread.
    #[error("no runtime to run on: {0}")]
    NoRuntime(String),
}

/// A client that cannot get its process handles fails for good, as a non-retryable `APIError`.
impl From<ProcessError> for lk_agents_abi::errors::ApiError {
    fn from(error: ProcessError) -> Self {
        Self::new(lk_agents_abi::errors::ErrorKind::InvalidRequest, error.to_string())
    }
}

/// Immutable provenance for native capabilities which may be reached through a detached-safe
/// view. A new runtime cannot make inherited storage or command handles current again.
#[derive(Clone, Copy)]
pub struct ProcessStamp(u32);

impl ProcessStamp {
    pub fn current() -> Self {
        Self(std::process::id())
    }

    pub fn check(self) -> Result<(), ProcessError> {
        if self.0 == std::process::id() { Ok(()) } else { Err(ProcessError::Inherited) }
    }
}

/// What the hosting process lends a declared class: the generated bindings pass
/// it to each declared constructor that takes it, and Rust callers and tests build their own.
/// Handles bound to the process runtime are read on use, so constructing a client starts
/// nothing and a fork-inherited runtime is refused rather than used.
#[derive(Clone)]
pub struct ProcessEnv {
    /// The runtime this crate's tasks and sockets run on.
    pub runtime: fn() -> Result<tokio::runtime::Handle, ProcessError>,
    /// The executor the host awaits a chain's futures on.
    pub spawner: fn() -> Result<Arc<dyn Spawner>, ProcessError>,
    /// The host's clock (virtual in a virtual-time test).
    pub clock: Arc<dyn Clock>,
    /// The process's shared connection pool.
    pub http: fn() -> Result<Arc<GatewayHttp>, ProcessError>,
    /// The process's span recording, timed by `clock`.
    pub tracer: lk_agents_telemetry::Tracer,
    /// The process's own tasks: work no session owns (a node the host calls outside a session,
    /// a handler span the SDK cancelled), each bounded and ending on its own.
    pub tasks: Arc<crate::task::TaskSet>,
    /// The runtime's executor for an explicitly bound state-only control operation. The
    /// callback depends only on ABI ports, so storage never depends on runtime decisions.
    pub remote_control: Option<RemoteControlExecutor>,
}

pub type RemoteControlExecutor = fn(
    ProcessEnv,
    Arc<dyn lk_agents_abi::session::control::BoundSessionControl>,
    lk_agents_abi::session::control::RemoteControlRef,
    Option<lk_agents_abi::delegation::remote::RemoteCancellationEvidence>,
)
    -> Arc<dyn lk_agents_abi::session::control::RemoteCancellation>;

impl ProcessEnv {
    /// Where work no session owns runs: the process's tasks, on its executor and clock.
    pub fn background(&self) -> Result<crate::owned::Background, ProcessError> {
        let spawner = (self.spawner)()?;
        Ok(crate::owned::Background::new(Arc::clone(&self.tasks), spawner, Arc::clone(&self.clock)))
    }

    /// `<hostname>-<pid>`: this process's default storage lease holder. Read on each call, so a
    /// job process names itself, not the parent it was forked from.
    pub fn lease_holder(&self) -> String {
        let host = sysinfo::System::host_name().unwrap_or_default();
        format!("{host}-{}", std::process::id())
    }

    /// The runtime the caller is on, else the process runtime.
    pub fn caller_runtime(&self) -> Result<tokio::runtime::Handle, ProcessError> {
        tokio::runtime::Handle::try_current().or_else(|_| (self.runtime)())
    }

    /// A Rust caller with no host: the caller's Tokio runtime, the wall clock, no span recording,
    /// a connection pool per client (a pool's connections belong to the runtime that opened them, and the
    /// caller's runtime may not outlive the client).
    pub fn current_runtime() -> Self {
        fn runtime() -> Result<tokio::runtime::Handle, ProcessError> {
            tokio::runtime::Handle::try_current()
                .map_err(|error| ProcessError::NoRuntime(error.to_string()))
        }
        let clock: Arc<dyn Clock> = Arc::new(crate::clock::RealClock::new());
        ProcessEnv {
            runtime,
            spawner: || Ok(Arc::new(crate::task::TokioSpawner::new(runtime()?))),
            tracer: lk_agents_telemetry::Tracer::disabled(clock.clone()),
            clock,
            tasks: Arc::default(),
            remote_control: None,
            http: || {
                GatewayHttp::new().map(Arc::new).map_err(|e| ProcessError::Start(e.to_string()))
            },
        }
    }
}

/// The HTTP proxy the worker handed this process's jobs: provider sockets tunnel through it.
static HTTP_PROXY: parking_lot::RwLock<Option<String>> = parking_lot::RwLock::new(None);

/// Registers the HTTP proxy this process's jobs connect through.
pub fn serve_http_proxy(proxy: Option<String>) {
    *HTTP_PROXY.write() = proxy;
}

impl ProcessEnv {
    /// The HTTP proxy this process's jobs connect through, as the worker handed it over.
    pub fn http_proxy(&self) -> Option<String> {
        HTTP_PROXY.read().clone()
    }
}

/// The server's settings (`LIVEKIT_URL`, `LIVEKIT_API_KEY`, ...) as the jobs this process runs
/// read them: what `AgentServer` was given, else its environment. A job process inherits them as
/// its environment; a job thread reads the ones its server registered here.
static SERVER_SETTINGS: parking_lot::RwLock<Vec<(&'static str, String)>> =
    parking_lot::RwLock::new(Vec::new());

/// The registration of a server's settings for its job threads, withdrawn when dropped.
pub struct ServerSettings(());

impl Drop for ServerSettings {
    fn drop(&mut self) {
        SERVER_SETTINGS.write().clear();
    }
}

/// Registers the running server's settings until the returned value drops.
pub fn serve_settings(settings: Vec<(&'static str, String)>) -> ServerSettings {
    *SERVER_SETTINGS.write() = settings;
    ServerSettings(())
}

/// The server setting `name`: the running server's, else the environment variable.
pub fn server_setting(name: &str) -> Option<String> {
    let registered =
        SERVER_SETTINGS.read().iter().find(|(n, _)| *n == name).map(|(_, v)| v.clone());
    registered.or_else(|| std::env::var(name).ok())
}

/// The command that started this process: the launcher's, or its worker's for a job process.
static LAUNCH: parking_lot::RwLock<Launch> = parking_lot::RwLock::new(Launch::None);

/// Registers the command that started this process, before its telemetry is installed.
pub fn serve_launch(launch: Launch) {
    *LAUNCH.write() = launch;
}

/// The command that started this process; `none` when no `lk agent` command did.
pub fn launch() -> Launch {
    *LAUNCH.read()
}

/// The process's runner, derived once from the command that started it, its platform and the
/// `CI` and `LK_HOSTED_AGENTS` variables. No hostname, user name or path is read.
pub fn runner() -> Runner {
    static RUNNER: std::sync::OnceLock<Runner> = std::sync::OnceLock::new();
    *RUNNER.get_or_init(|| {
        Runner::detect(|name| std::env::var(name).ok(), std::env::consts::OS, launch())
    })
}

/// Whether LiveKit's product statistics are on, read once: `LIVEKIT_PRODUCT_STATS=0` or
/// `DO_NOT_TRACK=1` turns them off for the process.
pub fn product_stats_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| product_stats_allowed(|name| std::env::var(name).ok()))
}

fn product_stats_allowed(var: impl Fn(&str) -> Option<String>) -> bool {
    let value = |name| var(name).map(|v| v.trim().to_ascii_lowercase());
    let off =
        matches!(value("LIVEKIT_PRODUCT_STATS").as_deref(), Some("0" | "false" | "off" | "no"));
    let do_not_track =
        value("DO_NOT_TRACK").is_some_and(|v| !matches!(v.as_str(), "" | "0" | "false"));
    !off && !do_not_track
}

#[cfg(test)]
mod product_stats_tests {
    // Catches: an opt-out variable ignored, or a value that keeps them on read as an opt-out.
    #[test]
    fn either_variable_opts_out() {
        for (vars, allowed) in [
            (&[][..], true),
            (&[("LIVEKIT_PRODUCT_STATS", "0")][..], false),
            (&[("LIVEKIT_PRODUCT_STATS", "false")][..], false),
            (&[("LIVEKIT_PRODUCT_STATS", "1")][..], true),
            (&[("DO_NOT_TRACK", "1")][..], false),
            (&[("DO_NOT_TRACK", "0")][..], true),
            (&[("LIVEKIT_PRODUCT_STATS", "1"), ("DO_NOT_TRACK", "1")][..], false),
        ] {
            let var =
                |name: &str| vars.iter().find(|(k, _)| *k == name).map(|(_, v)| (*v).to_owned());
            assert_eq!(super::product_stats_allowed(var), allowed, "{vars:?}");
        }
    }
}

#[cfg(test)]
mod stamp_tests {
    use super::*;

    #[test]
    fn a_capability_stamp_accepts_only_its_creator_process() {
        ProcessStamp::current().check().unwrap();
        let inherited = ProcessStamp(std::process::id() ^ u32::MAX);
        assert_eq!(inherited.check(), Err(ProcessError::Inherited));
        assert_eq!(inherited.check(), Err(ProcessError::Inherited), "the stamp is immutable");
    }
}
