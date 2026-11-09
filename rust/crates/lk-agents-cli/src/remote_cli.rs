use lk_agents_abi::errors::HostError;
use prost::Message as _;
use std::sync::Arc;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

use livekit_protocol::agent::{AgentDevMessage, agent_dev_message::Message};
use lk_agents_abi::worker::job::{AcceptArguments, RunningJobInfo};
use lk_agents_abi::worker::server_host::ServerJobs;

const DEV_FRAME_LIMIT: usize = 8 * 1024 * 1024;

fn to_proto(info: RunningJobInfo) -> livekit_protocol::agent::RunningAgentJobInfo {
    livekit_protocol::agent::RunningAgentJobInfo {
        job: info.job.encode_to_vec(),
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
            attributes: Default::default(),
        },
        job,
        url: pb.url,
        token: pb.token,
        worker_id: pb.worker_id,
        fake_job: pb.mock_job,
        received_at: None,
        accepted_at: None,
        assigned_at: None,
        launched_at: 0.0,
    })
}

fn oversize() -> std::io::Error {
    std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("dev frame exceeds {DEV_FRAME_LIMIT} bytes"),
    )
}

/// `None` once the CLI closes the channel between frames.
async fn recv(stream: &mut (impl AsyncRead + Unpin)) -> std::io::Result<Option<AgentDevMessage>> {
    let mut len = [0u8; 4];
    let read = stream.read(&mut len).await?;
    if read == 0 {
        return Ok(None);
    }
    stream.read_exact(&mut len[read..]).await?;
    let len = u32::from_be_bytes(len) as usize;
    if len > DEV_FRAME_LIMIT {
        return Err(oversize());
    }
    let mut bytes = vec![0u8; len];
    stream.read_exact(&mut bytes).await?;
    AgentDevMessage::decode(bytes.as_slice())
        .map(Some)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

async fn send(stream: &mut (impl AsyncWrite + Unpin), message: Message) -> std::io::Result<()> {
    let bytes = AgentDevMessage { message: Some(message) }.encode_to_vec();
    if bytes.len() > DEV_FRAME_LIMIT {
        return Err(oversize());
    }
    stream.write_all(&(bytes.len() as u32).to_be_bytes()).await?;
    stream.write_all(&bytes).await?;
    stream.flush().await
}

/// Serves the dev channel to `lk agent dev` at `cli_addr` (`host:port`): announces the server,
/// resumes the jobs kept across a reload and answers running-jobs requests until the CLI closes
/// it. A failure is logged; the worker runs on without the channel.
pub async fn dev_channel(
    cli_addr: String,
    agent_name: String,
    url: String,
    api_key: String,
    api_secret: String,
    server: Arc<dyn ServerJobs>,
) {
    if let Err(err) =
        serve_dev_channel(cli_addr, agent_name, url, api_key, api_secret, server).await
    {
        log::error!(target: lk_agents_abi::LOG_TARGET, "dev channel failed: {err}");
    }
}

async fn serve_dev_channel(
    cli_addr: String,
    agent_name: String,
    url: String,
    api_key: String,
    api_secret: String,
    server: Arc<dyn ServerJobs>,
) -> Result<(), HostError> {
    use livekit_protocol::agent::{
        GetRunningAgentJobsRequest, GetRunningAgentJobsResponse, ServerInfo,
    };

    let io = |e: std::io::Error| HostError::new(format!("dev channel: {e}"));
    let mut stream = tokio::net::TcpStream::connect(&cli_addr).await.map_err(io)?;

    send(&mut stream, Message::ServerInfo(ServerInfo { agent_name, url: url.clone() }))
        .await
        .map_err(io)?;
    send(&mut stream, Message::GetRunningJobsRequest(GetRunningAgentJobsRequest {}))
        .await
        .map_err(io)?;
    let Some(first) = recv(&mut stream).await.map_err(io)? else { return Ok(()) };
    if let Some(Message::GetRunningJobsResponse(kept)) = first.message
        && !kept.jobs.is_empty()
    {
        log::info!(target: lk_agents_abi::LOG_TARGET, "reloading {} job(s)", kept.jobs.len());
        let mut jobs = Vec::with_capacity(kept.jobs.len());
        for mut job in kept.jobs {
            job.token = lk_agents_net::token::extend_job_token(&job.token, &api_key, &api_secret)
                .map_err(|e| HostError::new(format!("cannot reload a job: {e}")))?;
            job.url = url.clone();
            jobs.push(from_proto(job)?);
        }
        server.reload_jobs(jobs).await.map_err(|e| HostError::new(e.to_string()))?;
    }
    // The CLI asks before restarting this process, to resume these jobs after the reload.
    while let Some(message) = recv(&mut stream).await.map_err(io)? {
        if let Some(Message::GetRunningJobsRequest(_)) = message.message {
            let jobs = server.running_jobs().into_iter().map(to_proto).collect();
            let reply = Message::GetRunningJobsResponse(GetRunningAgentJobsResponse { jobs });
            send(&mut stream, reply).await.map_err(io)?;
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "remote_cli_tests.rs"]
mod tests;
