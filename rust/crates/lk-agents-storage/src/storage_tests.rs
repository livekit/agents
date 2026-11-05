//! The crate's one integration binary: threads, contexts and backends through the public API.

#[path = "backends/agentdb/tasks_tests.rs"]
mod cloud_tasks;
#[path = "context_tests.rs"]
mod context;
#[path = "http/http_tests.rs"]
mod http_api;
#[path = "thread/lease_tests.rs"]
mod lease;
#[path = "backends/lifecycle_tests.rs"]
mod lifecycle;
#[path = "backends/process_tests.rs"]
mod process;
#[path = "thread/runs_tests.rs"]
mod runs;
#[path = "thread/schema_tests.rs"]
mod schema;
#[path = "voice_tests.rs"]
mod voice;
