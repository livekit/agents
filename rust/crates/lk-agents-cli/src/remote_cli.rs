//! Connection to the external `lk` launcher for server discovery and job reloads.

use crate::LOG;
use lk_agents_abi::errors::HostError;
use prost::Message as _;
use std::sync::Arc;

pub use lk_agents_abi::cli::DevServer;
use lk_agents_abi::worker::types::{AcceptArguments, RunningJobInfo};

fn to_proto(info: RunningJobInfo) -> livekit_protocol::agent::RunningAgentJobInfo {
    livekit_protocol::agent::RunningAgentJobInfo {
        job: info.job.proto,
        accept_name: info.accept_arguments.name,
        accept_identity: info.accept_arguments.identity,
        accept_metadata: info.accept_arguments.metadata,
        url: info.url,
        token: info.token,
        worker_id: info.worker_id,
        mock_job: info.fake_job,
    }
}

/// A job kept across the reload; no dispatch offered it to this process.
fn from_proto(
    pb: livekit_protocol::agent::RunningAgentJobInfo,
) -> Result<RunningJobInfo, HostError> {
    let job = livekit_protocol::Job::decode(pb.job.as_slice())
        .map_err(|e| HostError::new(format!("cannot reload a job: {e}")))?;
    Ok(RunningJobInfo {
        accept_arguments: AcceptArguments {
            name: pb.accept_name,
            identity: pb.accept_identity,
            metadata: pb.accept_metadata,
            attributes: Vec::new(),
        },
        job: lk_agents_worker::job_info(&job),
        url: pb.url,
        token: pb.token,
        worker_id: pb.worker_id,
        fake_job: pb.mock_job,
        received_at: 0.0,
        accepted_at: 0.0,
        assigned_at: 0.0,
    })
}

/// The dev channel to `lk agent dev` at `cli_addr` (`host:port`), length-prefixed
/// `AgentDevMessage`s: the server's name and URL first, then the jobs kept from before the reload
/// asked for and resumed (on `url`, each with its token extended by the server's API key and
/// secret), then every running-jobs request answered, until the CLI closes it. A failure is
/// logged: the worker runs on without the channel.
pub async fn dev_channel(
    cli_addr: String,
    agent_name: String,
    url: String,
    api_key: String,
    api_secret: String,
    server: Arc<dyn DevServer>,
) {
    if let Err(err) =
        serve_dev_channel(cli_addr, agent_name, url, (api_key, api_secret), server).await
    {
        log::error!(target: LOG, "dev channel failed: {err}");
    }
}

async fn serve_dev_channel(
    cli_addr: String,
    agent_name: String,
    url: String,
    (api_key, api_secret): (String, String),
    server: Arc<dyn DevServer>,
) -> Result<(), HostError> {
    use livekit_protocol::agent::{
        AgentDevMessage, GetRunningAgentJobsRequest, GetRunningAgentJobsResponse, ServerInfo,
        agent_dev_message::Message,
    };
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let io = |e: std::io::Error| HostError::new(format!("dev channel: {e}"));
    let mut stream = tokio::net::TcpStream::connect(&cli_addr).await.map_err(io)?;
    async fn send(stream: &mut tokio::net::TcpStream, message: Message) -> std::io::Result<()> {
        let bytes = AgentDevMessage { message: Some(message) }.encode_to_vec();
        stream.write_all(&(bytes.len() as u32).to_be_bytes()).await?;
        stream.write_all(&bytes).await?;
        stream.flush().await
    }
    async fn recv(stream: &mut tokio::net::TcpStream) -> std::io::Result<Option<Message>> {
        let mut len = [0u8; 4];
        stream.read_exact(&mut len).await?;
        let mut bytes = vec![0u8; u32::from_be_bytes(len) as usize];
        stream.read_exact(&mut bytes).await?;
        let decoded = AgentDevMessage::decode(bytes.as_slice())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(decoded.message)
    }

    send(&mut stream, Message::ServerInfo(ServerInfo { agent_name, url: url.clone() }))
        .await
        .map_err(io)?;
    send(&mut stream, Message::GetRunningJobsRequest(GetRunningAgentJobsRequest {}))
        .await
        .map_err(io)?;
    let Ok(first) = recv(&mut stream).await else { return Ok(()) };
    if let Some(Message::GetRunningJobsResponse(kept)) = first
        && !kept.jobs.is_empty()
    {
        log::info!(target: LOG, "reloading {} job(s)", kept.jobs.len());
        let mut jobs = Vec::with_capacity(kept.jobs.len());
        for mut job in kept.jobs {
            job.token = lk_agents_net::token::extend_job_token(&job.token, &api_key, &api_secret)
                .map_err(|e| HostError::new(format!("cannot reload a job: {e}")))?;
            job.url = url.clone();
            jobs.push(from_proto(job)?);
        }
        server.reload_jobs(&lk_agents_abi::telemetry::Cx::NONE, jobs).await?;
    }
    // Asked before the CLI restarts this process: what runs now, to resume after the reload.
    while let Ok(message) = recv(&mut stream).await {
        if let Some(Message::GetRunningJobsRequest(_)) = message {
            let jobs = server.running_jobs()?.into_iter().map(to_proto).collect();
            let reply = Message::GetRunningJobsResponse(GetRunningAgentJobsResponse { jobs });
            if send(&mut stream, reply).await.is_err() {
                break;
            }
        }
    }
    Ok(())
}
