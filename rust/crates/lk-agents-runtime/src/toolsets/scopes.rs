use super::group::Group;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::llm::toolsets::{self, Lifetime, ToolsetGroup as _, ToolsetHooks};
use lk_agents_abi::productstats::features::{Feature, SessionFeatures};
use lk_agents_core::toolsets::Scopes as Policy;
use lk_agents_net::env::ProcessEnv;
use parking_lot::Mutex;
use std::{collections::BTreeMap, sync::Arc};

pub struct Scopes {
    env: ProcessEnv,
    state: Arc<Mutex<State>>,
}
#[derive(Default)]
struct State {
    policy: Policy,
    next: u64,
    lifetimes: Vec<Arc<dyn Lifetime>>,
    /// Activations stay owned until their close settles, including after the policy releases
    /// them. Whole-session shutdown must join an earlier close whose waiter was cancelled.
    groups: BTreeMap<u64, Arc<Group>>,
    /// The session already counted its MCP toolsets in its product statistics.
    features: SessionFeatures,
}
// A cancelled or failed admission cannot occupy the older activation's release slot.
struct Admission {
    state: Arc<Mutex<State>>,
    id: u64,
    accepted: bool,
}
impl Drop for Admission {
    fn drop(&mut self) {
        if !self.accepted {
            let closing = {
                let mut state = self.state.lock();
                state.policy.discard(self.id);
                state.groups.get(&self.id).cloned()
            };
            if let Some(group) = closing {
                group.request_close();
            }
        }
    }
}

impl Scopes {
    async fn close_groups(&self, groups: Vec<(u64, Arc<Group>)>) -> Result<(), HostError> {
        // The group owns its close driver. A cancelled waiter leaves it in the registry so a
        // later whole-session release can join the same operation.
        let results =
            futures_util::future::join_all(groups.iter().map(|(_, group)| group.aclose())).await;
        let retired: Vec<_> = {
            let mut state = self.state.lock();
            groups.iter().filter_map(|(id, _)| state.groups.remove(id)).collect()
        };
        drop(retired);
        results.into_iter().find_map(Result::err).map_or(Ok(()), Err)
    }
}

impl toolsets::ToolsetScopes for Scopes {
    type Env = ProcessEnv;
    fn new(env: &ProcessEnv) -> Self {
        Self { env: env.clone(), state: Arc::default() }
    }
    async fn acquire(
        &self,
        scope: u64,
        toolsets: Vec<Arc<dyn ToolsetHooks>>,
    ) -> Result<u64, HostError> {
        // Read host-provided capabilities before the registry lock.
        let roots: Vec<_> = toolsets.iter().map(|hooks| hooks.lifetime().clone()).collect();
        let (id, group) = {
            let mut state = self.state.lock();
            let ids: Vec<_> = roots
                .iter()
                .map(|root| {
                    match state.lifetimes.iter().position(|other| Arc::ptr_eq(other, root)) {
                        Some(index) => index as u64,
                        None => {
                            let id = state.lifetimes.len() as u64;
                            state.lifetimes.push(root.clone());
                            id
                        }
                    }
                })
                .collect();
            state.next += 1;
            let id = state.next;
            let indices = state.policy.admit(scope, id, &ids);
            let entries = indices
                .into_iter()
                .map(|index| (roots[index].clone(), toolsets[index].clone()))
                .collect();
            let group = Arc::new(Group::from_entries(&self.env, entries));
            state.groups.insert(id, group.clone());
            (id, group)
        };
        let mut admission = Admission { state: self.state.clone(), id, accepted: false };
        if let Err(error) = group.start().await {
            // A failed start already settled its rollback. A cancelled start instead leaves
            // its group to the admission guard and whole-session shutdown.
            let retired = self.state.lock().groups.remove(&id);
            drop(retired);
            return Err(error);
        }
        admission.accepted = true;
        // The session's own toolsets and each agent's are activated here, nested ones included.
        if toolsets::LeaseGroup::mcp(&*group) {
            let counted = self.state.lock().features.used(Feature::Mcp);
            counted.into_iter().for_each(lk_agents_net::export::productstats::record);
        }
        Ok(id)
    }
    async fn release(&self, scope: Option<u64>) -> Result<(), HostError> {
        let groups: Vec<_> = {
            let mut state = self.state.lock();
            let released = state.policy.release(scope);
            match scope {
                Some(_) => released
                    .into_iter()
                    .filter_map(|id| state.groups.get(&id).map(|group| (id, group.clone())))
                    .collect(),
                None => state.groups.iter().map(|(&id, group)| (id, group.clone())).collect(),
            }
        };
        self.close_groups(groups).await
    }
}

impl toolsets::ScopesPort for Scopes {
    fn acquire(
        &self,
        scope: u64,
        toolsets: Vec<Arc<dyn ToolsetHooks>>,
    ) -> lk_agents_abi::BoxFuture<'_, Result<u64, HostError>> {
        Box::pin(toolsets::ToolsetScopes::acquire(self, scope, toolsets))
    }
    fn discard(&self, activation: u64) -> lk_agents_abi::BoxFuture<'_, Result<(), HostError>> {
        let group = {
            let mut state = self.state.lock();
            state.policy.discard(activation);
            state.groups.get(&activation).cloned()
        };
        if let Some(group) = &group {
            group.request_close();
        }
        Box::pin(async move {
            self.close_groups(group.into_iter().map(|group| (activation, group)).collect()).await
        })
    }
    fn release(&self, scope: Option<u64>) -> lk_agents_abi::BoxFuture<'_, Result<(), HostError>> {
        Box::pin(toolsets::ToolsetScopes::release(self, scope))
    }
}
