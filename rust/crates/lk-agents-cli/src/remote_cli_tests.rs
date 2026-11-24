use super::*;

struct Server;

impl ServerJobs for Server {
    fn running_jobs(&self) -> Vec<RunningJobInfo> {
        Vec::new()
    }

    fn reload_jobs(
        &self,
        _: Vec<RunningJobInfo>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<(), lk_agents_abi::worker::ipc_host::WorkerError>>
    {
        Box::pin(async { Ok(()) })
    }

    fn update_token(&self, _: String) {}
}

#[tokio::test]
async fn invalid_or_truncated_frames_fail_and_oversized_frames_need_no_payload() {
    for (bytes, kind) in [
        (u32::MAX.to_be_bytes().to_vec(), std::io::ErrorKind::InvalidData),
        (vec![0, 0, 0, 1, 0xff], std::io::ErrorKind::InvalidData),
        (vec![0, 0, 0, 2, 0], std::io::ErrorKind::UnexpectedEof),
        (vec![0, 0], std::io::ErrorKind::UnexpectedEof),
    ] {
        assert_eq!(recv(&mut bytes.as_slice()).await.unwrap_err().kind(), kind);
    }
}

#[tokio::test]
async fn receive_errors_escape_both_the_initial_reply_and_the_request_loop() {
    for initial in [true, false] {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let peer = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let message = |m: Option<AgentDevMessage>| m.unwrap().message;
            assert!(matches!(
                message(recv(&mut socket).await.unwrap()),
                Some(Message::ServerInfo(info)) if info.supports_token_mode
            ));
            assert!(matches!(
                message(recv(&mut socket).await.unwrap()),
                Some(Message::GetRunningJobsRequest(_))
            ));
            if !initial {
                let reply = AgentDevMessage {
                    message: Some(Message::GetRunningJobsResponse(
                        livekit_protocol::agent::GetRunningAgentJobsResponse { jobs: Vec::new() },
                    )),
                }
                .encode_to_vec();
                socket.write_all(&(reply.len() as u32).to_be_bytes()).await.unwrap();
                socket.write_all(&reply).await.unwrap();
            }
            socket.write_all(&[0, 0, 0, 1, 0xff]).await.unwrap();
        });
        let error = serve_dev_channel(
            addr,
            "test".into(),
            "ws://localhost".into(),
            String::new(),
            String::new(),
            Arc::new(Server),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("dev channel:"));
        peer.await.unwrap();
    }
}

#[tokio::test]
async fn a_close_between_frames_ends_the_channel_quietly() {
    assert!(recv(&mut [].as_slice()).await.unwrap().is_none());
    for initial in [true, false] {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap().to_string();
        let peer = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            recv(&mut socket).await.unwrap().unwrap();
            recv(&mut socket).await.unwrap().unwrap();
            if !initial {
                let reply = Message::GetRunningJobsResponse(Default::default());
                send(&mut socket, reply).await.unwrap();
            }
        });
        serve_dev_channel(
            addr,
            "test".into(),
            "ws://localhost".into(),
            String::new(),
            String::new(),
            Arc::new(Server),
        )
        .await
        .unwrap();
        peer.await.unwrap();
    }
}

#[tokio::test]
async fn oversized_frames_are_refused_before_sending() {
    let mut wire = Vec::new();
    let info = livekit_protocol::agent::ServerInfo {
        agent_name: "a".repeat(DEV_FRAME_LIMIT),
        url: String::new(),
        supports_token_mode: false,
    };
    let error = send(&mut wire, Message::ServerInfo(info)).await.unwrap_err();
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    assert!(wire.is_empty());
}

#[test]
fn reloaded_jobs_have_no_dispatch_timestamps() {
    let job = from_proto(livekit_protocol::agent::RunningAgentJobInfo {
        job: livekit_protocol::Job::default().encode_to_vec(),
        ..Default::default()
    })
    .unwrap();
    assert_eq!((job.received_at, job.accepted_at, job.assigned_at), (None, None, None));
}

/// A job kept across a dev reload comes back over the dev channel with every field it left with.
/// https://github.com/livekit/agents/pull/5345
#[tokio::test]
async fn a_kept_job_survives_the_dev_channel_round_trip() {
    let job = livekit_protocol::Job {
        id: "AJ_kept".into(),
        dispatch_id: "AD_1".into(),
        r#type: livekit_protocol::JobType::JtRoom as i32,
        room: Some(livekit_protocol::Room {
            sid: "RM_1".into(),
            name: "call".into(),
            ..Default::default()
        }),
        metadata: "{\"k\":1}".into(),
        agent_name: "front-desk".into(),
        ..Default::default()
    };
    let info = RunningJobInfo {
        accept_arguments: AcceptArguments {
            name: "Agent".into(),
            identity: "agent-AJ_kept".into(),
            metadata: "meta".into(),
            attributes: Default::default(),
        },
        job: job.clone(),
        url: "wss://sfu".into(),
        token: "token".into(),
        worker_id: "AW_1".into(),
        fake_job: true,
        received_at: Some(1.0),
        accepted_at: Some(2.0),
        assigned_at: Some(3.0),
        launched_at: 4.0,
    };
    let mut wire = Vec::new();
    let jobs = vec![to_proto(info)];
    let response = livekit_protocol::agent::GetRunningAgentJobsResponse { jobs };
    send(&mut wire, Message::GetRunningJobsResponse(response)).await.unwrap();
    let received = recv(&mut wire.as_slice()).await.unwrap().unwrap();
    let Some(Message::GetRunningJobsResponse(response)) = received.message else {
        panic!("the frame decoded to another message: {:?}", received.message)
    };
    let [kept] = <[_; 1]>::try_from(response.jobs).unwrap();
    let back = from_proto(kept).unwrap();
    assert_eq!(back.job, job);
    let accept = &back.accept_arguments;
    assert_eq!(
        (accept.name.as_str(), accept.identity.as_str(), accept.metadata.as_str()),
        ("Agent", "agent-AJ_kept", "meta")
    );
    assert_eq!(
        (back.url.as_str(), back.token.as_str(), back.worker_id.as_str(), back.fake_job),
        ("wss://sfu", "token", "AW_1", true)
    );
}

#[tokio::test]
async fn an_invalid_kept_job_does_not_discard_healthy_jobs_or_close_the_dev_channel() {
    struct RecordingServer(
        parking_lot::Mutex<Vec<RunningJobInfo>>,
        parking_lot::Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    );
    impl ServerJobs for RecordingServer {
        fn running_jobs(&self) -> Vec<RunningJobInfo> {
            self.0.lock().clone()
        }
        fn reload_jobs(
            &self,
            jobs: Vec<RunningJobInfo>,
        ) -> lk_agents_abi::BoxFuture<'_, Result<(), lk_agents_abi::worker::ipc_host::WorkerError>>
        {
            self.0.lock().extend(jobs);
            if let Some(reloaded) = self.1.lock().take() {
                reloaded.send(()).unwrap();
            }
            Box::pin(async { Ok(()) })
        }
        fn update_token(&self, _: String) {}
    }
    let token =
        lk_agents_net::token::room_join_token("key", "secret", "agent", "room", Default::default())
            .unwrap();
    let make = |id: &str, token: String| livekit_protocol::agent::RunningAgentJobInfo {
        job: livekit_protocol::Job { id: id.into(), ..Default::default() }.encode_to_vec(),
        token,
        ..Default::default()
    };
    let jobs = vec![
        make("first", token.clone()),
        make("bad-token", "not-a-token".into()),
        livekit_protocol::agent::RunningAgentJobInfo {
            job: vec![0xff],
            token: token.clone(),
            ..Default::default()
        },
        make("last", token),
    ];
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let (reloaded, reloading) = tokio::sync::oneshot::channel();
    let server = Arc::new(RecordingServer(Default::default(), Some(reloaded).into()));
    let peer = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        recv(&mut socket).await.unwrap().unwrap();
        recv(&mut socket).await.unwrap().unwrap();
        send(
            &mut socket,
            Message::GetRunningJobsResponse(livekit_protocol::agent::GetRunningAgentJobsResponse {
                jobs,
            }),
        )
        .await
        .unwrap();
        // Ask for the running jobs only after the reload started.
        reloading.await.unwrap();
        send(&mut socket, Message::GetRunningJobsRequest(Default::default())).await.unwrap();
        let reply = recv(&mut socket).await.unwrap().unwrap();
        let Some(Message::GetRunningJobsResponse(reply)) = reply.message else {
            panic!("expected a running-jobs reply");
        };
        let ids: Vec<_> =
            reply.jobs.into_iter().map(|job| from_proto(job).unwrap().job.id).collect();
        assert_eq!(ids, ["first", "last"]);
    });
    serve_dev_channel(
        addr,
        "test".into(),
        "wss://new-sfu".into(),
        "key".into(),
        "secret".into(),
        server.clone(),
    )
    .await
    .unwrap();
    peer.await.unwrap();
    let jobs = server.0.lock();
    assert_eq!(jobs.iter().map(|job| job.job.id.as_str()).collect::<Vec<_>>(), ["first", "last"]);
    assert!(jobs.iter().all(|job| job.url == "wss://new-sfu"));
}

/// A fake token mode server. Its reload waits for the new worker token, as a registration
/// does.
#[derive(Default)]
struct TokenServer {
    reloaded: parking_lot::Mutex<Vec<RunningJobInfo>>,
    updated: parking_lot::Mutex<Vec<String>>,
    registered: parking_lot::Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
    registration: parking_lot::Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
}

impl ServerJobs for TokenServer {
    fn running_jobs(&self) -> Vec<RunningJobInfo> {
        Vec::new()
    }
    fn reload_jobs(
        &self,
        jobs: Vec<RunningJobInfo>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<(), lk_agents_abi::worker::ipc_host::WorkerError>>
    {
        let registration = self.registration.lock().take().expect("one reload");
        Box::pin(async move {
            registration.await.expect("registered");
            self.reloaded.lock().extend(jobs);
            Ok(())
        })
    }
    fn update_token(&self, token: String) {
        self.updated.lock().push(token);
        if let Some(registered) = self.registered.lock().take() {
            registered.send(()).unwrap();
        }
    }
}

// Token mode reload: the CLI sends the kept job, then a new worker token. Catches these bugs:
// - the reload blocks the channel, so the new worker token that the reload waits for never
//   arrives;
// - the new worker token does not reach the server;
// - the kept job's token is extended with a key that does not exist, which drops the job.
#[tokio::test]
async fn token_mode_reloads_beside_the_channel_and_takes_the_clis_worker_token() {
    let (registered, registration) = tokio::sync::oneshot::channel();
    let server = Arc::new(TokenServer {
        registered: parking_lot::Mutex::new(Some(registered)),
        registration: parking_lot::Mutex::new(Some(registration)),
        ..Default::default()
    });
    let token =
        lk_agents_net::token::room_join_token("key", "secret", "agent", "room", Default::default())
            .unwrap();
    let kept = livekit_protocol::agent::RunningAgentJobInfo {
        job: livekit_protocol::Job { id: "AJ_kept".into(), ..Default::default() }.encode_to_vec(),
        token: token.clone(),
        ..Default::default()
    };
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap().to_string();
    let peer = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        recv(&mut socket).await.unwrap().unwrap();
        recv(&mut socket).await.unwrap().unwrap();
        let jobs = vec![kept];
        let response = livekit_protocol::agent::GetRunningAgentJobsResponse { jobs };
        send(&mut socket, Message::GetRunningJobsResponse(response)).await.unwrap();
        let credentials = livekit_protocol::agent::Credentials {
            token: "refreshed-worker-token".into(),
            ..Default::default()
        };
        send(&mut socket, Message::Credentials(credentials)).await.unwrap();
    });
    let served = serve_dev_channel(
        addr,
        "test".into(),
        "wss://new-sfu".into(),
        String::new(),
        String::new(),
        server.clone(),
    );
    tokio::time::timeout(std::time::Duration::from_secs(5), served).await.unwrap().unwrap();
    peer.await.unwrap();
    assert_eq!(*server.updated.lock(), ["refreshed-worker-token"]);
    let reloaded = server.reloaded.lock();
    let [job] = reloaded.as_slice() else { panic!("{reloaded:?}") };
    assert_eq!(job.job.id, "AJ_kept");
    assert_eq!(job.token, token, "the server replaces it once the worker registers");
    assert_eq!(job.url, "wss://new-sfu");
}
