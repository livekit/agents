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

    // Token mode: the CLI answers with `Credentials`, the worker token, now and on each refresh.
    send(
        &mut stream,
        Message::ServerInfo(ServerInfo { agent_name, url: url.clone(), supports_token_mode: true }),
    )
    .await
    .map_err(io)?;
    send(&mut stream, Message::GetRunningJobsRequest(GetRunningAgentJobsRequest {}))
        .await
        .map_err(io)?;
    while let Some(message) = recv(&mut stream).await.map_err(io)? {
        match message.message {
            Some(Message::GetRunningJobsResponse(kept)) if !kept.jobs.is_empty() => {
                let jobs = kept_jobs(kept.jobs, &url, &api_key, &api_secret);
                if !jobs.is_empty() {
                    server.reload_jobs(jobs).await.map_err(|e| HostError::new(e.to_string()))?;
                }
            }
            // The CLI asks before restarting this process, to resume these jobs after the reload.
            Some(Message::GetRunningJobsRequest(_)) => {
                let jobs = server.running_jobs().into_iter().map(to_proto).collect();
                let reply = Message::GetRunningJobsResponse(GetRunningAgentJobsResponse { jobs });
                send(&mut stream, reply).await.map_err(io)?;
            }
            Some(Message::Credentials(credentials)) => server.update_token(credentials.token),
            _ => {}
        }
    }
    Ok(())
}

/// The jobs kept across a reload, with tokens that outlive it: extended with the key pair, or
/// unchanged without one (token mode), where the server issues new ones.
fn kept_jobs(
    kept: Vec<livekit_protocol::agent::RunningAgentJobInfo>,
    url: &str,
    api_key: &str,
    api_secret: &str,
) -> Vec<RunningJobInfo> {
    log::info!(target: lk_agents_abi::LOG_TARGET, "reloading {} job(s)", kept.len());
    let mut jobs = Vec::with_capacity(kept.len());
    for job in kept {
        let mut job = match from_proto(job) {
            Ok(job) => job,
            Err(error) => {
                log::warn!(target: lk_agents_abi::LOG_TARGET, "cannot reload an invalid kept job: {error}");
                continue;
            }
        };
        if !api_key.is_empty() || !api_secret.is_empty() {
            job.token = match lk_agents_net::token::extend_job_token(
                &job.token, api_key, api_secret,
            ) {
                Ok(token) => token,
                Err(error) => {
                    log::warn!(target: lk_agents_abi::LOG_TARGET, job_id = job.job.id.as_str(); "cannot refresh a kept job's token: {error}");
                    continue;
                }
            };
        }
        job.url = url.to_owned();
        jobs.push(job);
    }
    jobs
}

#[cfg(test)]
#[path = "remote_cli_tests.rs"]
mod tests;
