//! The server API calls of a warm transfer in token mode.
//!
//! These tests run in their own test binary. They register server settings with a worker token
//! and no key or secret, and those settings apply to every test in the binary.
#![allow(clippy::unwrap_used, reason = "integration test")]
#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "tests spawn and bind directly"
)]

use std::collections::BTreeMap;
use std::sync::Arc;

use lk_agents_abi::errors::ApiError;
use lk_agents_abi::telemetry::record::JobKey;
use lk_agents_net::job_token::{self, Issue, Issued, JobTokens};
use lk_agents_runtime::voice::workflows::warm_transfer::{ProjectApi, TransferApi};
use parking_lot::Mutex;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// A fake token server for one job. It issues `<job>-token-<n>` for each request.
struct Server(&'static str, Mutex<Vec<String>>);

impl Server {
    fn of(job: &'static str) -> Arc<Self> {
        Arc::new(Server(job, Mutex::default()))
    }
}

#[async_trait::async_trait]
impl Issue for Server {
    async fn issue(&self, claims: &str) -> Result<Issued, ApiError> {
        let mut asked = self.1.lock();
        asked.push(claims.to_owned());
        let expires_at = lk_agents_net::clock::unix_now() as i64 + 600;
        Ok(Issued { token: format!("{}-token-{}", self.0, asked.len()), expires_at })
    }
}

/// A fake LiveKit server API that answers `count` calls. Returns the path and the bearer of each
/// call.
async fn livekit_api(count: usize) -> (String, tokio::task::JoinHandle<Vec<(String, String)>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let served = tokio::spawn(async move {
        let mut calls = Vec::new();
        for _ in 0..count {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut chunk = [0u8; 4096];
            while !request.windows(4).any(|w| w == b"\r\n\r\n") {
                let n = socket.read(&mut chunk).await.unwrap();
                request.extend_from_slice(&chunk[..n]);
            }
            let request = String::from_utf8_lossy(&request).to_string();
            let path = request.split_whitespace().nth(1).unwrap().to_owned();
            let bearer = request
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(": ")?;
                    let bearer = value.strip_prefix("Bearer ")?;
                    name.eq_ignore_ascii_case("authorization").then(|| bearer.to_owned())
                })
                .unwrap_or_default();
            calls.push((path, bearer));
            let response = "HTTP/1.1 200 OK\r\ncontent-type: application/protobuf\r\n\
                            content-length: 0\r\nconnection: close\r\n\r\n";
            socket.write_all(response.as_bytes()).await.unwrap();
        }
        calls
    });
    (url, served)
}

// Two jobs are registered, and the first job makes warm transfer calls. Catches these bugs:
// - a call signs its own token with no key, instead of using one job token per call;
// - a call uses the second job's token;
// - a job token request asks for grants other than the call's, for example the wrong room;
// - in key mode, a call does not sign its token with the key and secret.
#[tokio::test]
async fn token_mode_transfer_calls_carry_a_job_token_with_their_grants() {
    let _settings = lk_agents_net::env::serve_settings(vec![
        ("LIVEKIT_TOKEN", "the-worker-token".into()),
        ("LIVEKIT_API_KEY", String::new()),
        ("LIVEKIT_API_SECRET", String::new()),
    ]);
    let (server, other) = (Server::of("transfer"), Server::of("other"));
    let clock = Arc::new(lk_agents_net::clock::RealClock::new());
    let (job, newer) = (JobKey("AJ_transfer".into()), JobKey("AJ_other".into()));
    job_token::open(job.clone(), Arc::new(JobTokens::new(server.clone(), clock.clone())));
    job_token::open(newer.clone(), Arc::new(JobTokens::new(other.clone(), clock)));

    let (url, served) = livekit_api(2).await;
    let env = BTreeMap::from([("LIVEKIT_URL".to_owned(), url)]);
    let api = ProjectApi::of_server(&env, job.clone());
    api.move_participant("call".into(), "caller".into(), "human-room".into()).await.unwrap();
    api.delete_room("human-room".into()).await.unwrap();
    let calls = served.await.unwrap();
    let gone = ProjectApi::of_server(&env, JobKey("AJ_ended".into()));
    let error = gone.delete_room("human-room".into()).await.unwrap_err();
    assert!(error.message.contains("needs a running job"), "{error}");
    job_token::close(&job);
    job_token::close(&newer);

    assert!(other.1.lock().is_empty(), "another job's issuer was asked");
    let asked: Vec<serde_json::Value> =
        server.1.lock().iter().map(|c| serde_json::from_str(c).unwrap()).collect();
    assert_eq!(
        asked,
        [
            serde_json::json!({"video": {"roomAdmin": true, "room": "call", "destinationRoom": "human-room"}}),
            serde_json::json!({"video": {"roomCreate": true}}),
        ]
    );
    assert_eq!(
        calls,
        [
            (
                "/twirp/livekit.RoomService/MoveParticipant".to_owned(),
                "transfer-token-1".to_owned()
            ),
            ("/twirp/livekit.RoomService/DeleteRoom".to_owned(), "transfer-token-2".to_owned()),
        ]
    );

    let (url, served) = livekit_api(1).await;
    let keyed = BTreeMap::from([
        ("LIVEKIT_URL".to_owned(), url),
        ("LIVEKIT_API_KEY".to_owned(), "key".to_owned()),
        ("LIVEKIT_API_SECRET".to_owned(), "secret-secret-secret-secret-secret".to_owned()),
    ]);
    ProjectApi::of_server(&keyed, job).delete_room("human-room".into()).await.unwrap();
    let [(_, bearer)] = <[_; 1]>::try_from(served.await.unwrap()).unwrap();
    let claims =
        livekit_token::TokenVerifier::with_api_key("key", "secret-secret-secret-secret-secret")
            .verify(&bearer)
            .expect("signed with the key pair");
    assert!(claims.video.room_create);
    assert_eq!(server.1.lock().len(), 2, "no job token in key mode");
}
