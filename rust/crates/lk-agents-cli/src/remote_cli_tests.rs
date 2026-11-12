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
                Some(Message::ServerInfo(_))
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
