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
