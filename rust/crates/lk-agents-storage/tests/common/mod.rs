//! The staging project the local tests run against: `LIVEKIT_URL`, `LIVEKIT_API_KEY` and
//! `LIVEKIT_API_SECRET`. `None`, and the test skips, when they are not set.

use lk_agents_storage::{StorageOptions, TokenMinter};

pub fn project() -> Option<(StorageOptions, TokenMinter)> {
    let env = |name: &str| std::env::var(name).ok().filter(|v| !v.is_empty());
    match (env("LIVEKIT_URL"), env("LIVEKIT_API_KEY"), env("LIVEKIT_API_SECRET")) {
        (Some(url), Some(key), Some(secret)) => {
            Some((StorageOptions::new(&url), TokenMinter::new(key, secret).unwrap()))
        }
        _ => {
            eprintln!("skipped: no agentstore project in the environment");
            None
        }
    }
}

/// A database for one test, an hour long.
pub async fn create_database(
    storage: &lk_agents_storage::Storage,
) -> Result<livekit_protocol::agent_db::CreateResponse, lk_agents_storage::StorageError> {
    let ttl = Some(std::time::Duration::from_secs(3600).into());
    let req = livekit_protocol::agent_db::CreateRequest { ttl };
    storage.databases().create(req).await.map_err(lk_agents_storage::service_error)
}
