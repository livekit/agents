//! Who a gateway request is made for. The worker records each job's identity as the job starts and
//! its room session adds the agent's participant once the room connects; a request reads the
//! identity of the job its telemetry context runs for (`cx.job()`) when it is made. A request made
//! outside any job carries none.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use lk_agents_abi::Headers;
use lk_agents_abi::telemetry::record::JobKey;
use parking_lot::Mutex;

/// The job an inference request is made for.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct JobIdentity {
    pub room_sid: Option<String>,
    pub job_id: Option<String>,
    /// The agent participant's sid, once the room is connected.
    pub agent_sid: Option<String>,
    /// `LIVEKIT_WORKER_TOKEN`.
    pub worker_token: Option<String>,
    /// The job's own headers (a text simulation's low priority), which override the ones above.
    pub headers: Headers,
}

fn jobs() -> &'static Mutex<BTreeMap<JobKey, JobIdentity>> {
    static JOBS: OnceLock<Mutex<BTreeMap<JobKey, JobIdentity>>> = OnceLock::new();
    JOBS.get_or_init(Mutex::default)
}

/// Records `job`'s identity as it starts.
pub fn open(job: JobKey, identity: JobIdentity) {
    jobs().lock().insert(job, identity);
}

/// The agent joined `job`'s room as `sid`.
pub fn agent_joined(job: &JobKey, sid: String) {
    if let Some(identity) = jobs().lock().get_mut(job) {
        identity.agent_sid = Some(sid).filter(|sid| !sid.is_empty());
    }
}

/// `job` ended: its requests are done.
pub fn close(job: &JobKey) {
    jobs().lock().remove(job);
}

/// The identity of `job`, when it is a running job of this process.
pub fn of(job: Option<&JobKey>) -> Option<JobIdentity> {
    jobs().lock().get(job?).cloned()
}

/// The `User-Agent` of this process's requests: the host SDK's, set once as it loads, else the
/// core's own.
pub fn user_agent() -> String {
    match HOST_USER_AGENT.get() {
        Some(agent) => agent.clone(),
        None => format!("LiveKit Agents/{} (rust)", env!("CARGO_PKG_VERSION")),
    }
}

static HOST_USER_AGENT: OnceLock<String> = OnceLock::new();

/// The host SDK names itself (`LiveKit Agents/<version> (python <version>)`); the first name
/// given holds for the process.
pub fn set_user_agent(agent: String) {
    let _ = HOST_USER_AGENT.set(agent);
}

#[cfg(test)]
mod tests {
    use super::*;

    // Catches: a request of one job carrying another's identity, or a closed job's identity kept.
    #[test]
    fn each_job_has_its_own_identity_until_it_closes() {
        let (first, second) = (JobKey("AJ_identity_1".into()), JobKey("AJ_identity_2".into()));
        open(
            first.clone(),
            JobIdentity { job_id: Some("AJ_identity_1".into()), ..Default::default() },
        );
        open(
            second.clone(),
            JobIdentity { job_id: Some("AJ_identity_2".into()), ..Default::default() },
        );
        agent_joined(&first, "PA_1".into());

        let identity = of(Some(&first)).expect("open");
        assert_eq!(identity.agent_sid.as_deref(), Some("PA_1"));
        assert_eq!(of(Some(&second)).and_then(|i| i.agent_sid), None);
        assert_eq!(of(None), None);
        close(&first);
        assert_eq!(of(Some(&first)), None);
        close(&second);
    }
}
