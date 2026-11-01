//! Retains each activation's native lease group; scope selection belongs to core.

use super::group::Group;
use lk_agents_abi::errors::HostError;
use lk_agents_abi::llm::toolsets::{self, Lifetime, ToolsetGroup as _, ToolsetHooks};
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
    groups: BTreeMap<u64, Arc<Group>>,
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
            let retired = {
                let mut state = self.state.lock();
                state.policy.discard(self.id);
                state.groups.remove(&self.id)
            };
            drop(retired);
        }
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
        group.start().await?;
        admission.accepted = true;
        Ok(id)
    }
    async fn release(&self, scope: Option<u64>) -> Result<(), HostError> {
        let groups: Vec<_> = {
            let mut state = self.state.lock();
            state
                .policy
                .release(scope)
                .into_iter()
                .filter_map(|id| state.groups.remove(&id))
                .collect()
        };
        let results =
            futures_util::future::join_all(groups.iter().map(|group| group.aclose())).await;
        results.into_iter().find_map(Result::err).map_or(Ok(()), Err)
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
            state.groups.remove(&activation)
        };
        Box::pin(async move {
            match group {
                Some(group) => group.aclose().await,
                None => Ok(()),
            }
        })
    }
    fn release(&self, scope: Option<u64>) -> lk_agents_abi::BoxFuture<'_, Result<(), HostError>> {
        Box::pin(toolsets::ToolsetScopes::release(self, scope))
    }
}
