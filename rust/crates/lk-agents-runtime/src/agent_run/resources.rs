//! The resources surrounding a direct run or a whole text-session turn. The owner, rather than
//! any language waiter, retains startup, rollback and finalization through settlement.

use lk_agents_abi::agent::resources::{ContextHydration, ResourceError};
use lk_agents_abi::llm::toolsets::{ToolsetGroup as _, ToolsetHooks};
use lk_agents_abi::storage::Storage as _;
use lk_agents_abi::storage::context::ContextActivation as _;
use lk_agents_abi::{BoxFuture, errors::HostError};
use lk_agents_core::resource_lifecycle::{CloseOwner, OwnerLifecycle, StartOwner, StoragePlan};
use lk_agents_net::{env::ProcessEnv, task::TaskSet};
use lk_agents_storage::{context::store::ContextStore, namespace::Namespace};
use parking_lot::Mutex;
use std::sync::{Arc, OnceLock};
use tokio::sync::watch;

type Outcome = Result<(), ResourceError>;
type Reply = watch::Sender<Option<Outcome>>;

#[cfg(test)]
#[path = "resources_tests.rs"]
mod tests;

pub struct RunResources(Arc<Inner>);

struct Inner {
    env: ProcessEnv,
    lifecycle: Mutex<OwnerLifecycle>,
    plan: StoragePlan,
    inputs: Mutex<Option<Inputs>>,
    opened: OnceLock<Opened>,
    toolsets: crate::toolsets::group::Group,
    hydration: Mutex<Hydration>,
    tasks: Arc<TaskSet>,
    closing: watch::Sender<bool>,
    started: Reply,
    closed: Reply,
}

struct Inputs {
    namespace: Option<Arc<Namespace>>,
    thread_id: Option<String>,
    activation: Option<ContextStore>,
    lane: String,
}

#[derive(Default)]
struct Opened {
    // Retain partial acquisition before the next fallible operation.
    namespace: Option<Arc<Namespace>>,
    thread_id: Option<String>,
    activation: Option<ContextStore>,
}

#[derive(Default)]
struct Hydration {
    closed: bool,
    views: Vec<Arc<dyn ContextHydration>>,
}

impl Inner {
    async fn acquire(&self) -> Outcome {
        let inputs = self.inputs.lock().take().ok_or(ResourceError::Closed)?;
        let mut acquired = Opened::default();
        let result = async {
            let namespace = match inputs.namespace {
                Some(namespace) => namespace,
                None => Arc::new(Namespace::new(&self.env, None, None, None, None, None, None)?),
            };
            acquired.namespace = Some(namespace.clone());
            let thread_id = match inputs.thread_id {
                Some(id) => id,
                None => namespace.create_thread(None, None, 0).await?.thread_id,
            };
            acquired.thread_id = Some(thread_id.clone());
            acquired.activation = Some(match inputs.activation {
                Some(activation) => activation,
                None => namespace._core_context_store(thread_id, inputs.lane).await?,
            });
            Ok(())
        }
        .await;
        // Even a failed context open retains the temporary namespace for cleanup.
        let _ = self.opened.set(acquired);
        result
    }

    async fn cleanup(&self) -> Outcome {
        // Toolset callbacks may still use their thread/context while closing.
        let result = self.toolsets.aclose().await.map_err(ResourceError::Host);
        if let Some(opened) = self.opened.get() {
            if self.plan.own_activation
                && let Some(activation) = &opened.activation
            {
                activation.close();
            }
            self.release_hydration();
            if self.plan.own_namespace
                && let Some(namespace) = &opened.namespace
            {
                namespace.aclose().await;
            }
        } else {
            self.release_hydration();
        }
        let retired = self.inputs.lock().take();
        drop(retired);
        result
    }

    fn release_hydration(&self) {
        let views = {
            let mut hydration = self.hydration.lock();
            hydration.closed = true;
            std::mem::take(&mut hydration.views)
        };
        for view in views {
            if let Err(error) = view.release() {
                log::warn!("the run's context hydration could not release: {error}");
            }
        }
    }

    async fn run(self: Arc<Self>) {
        let mut closing = self.closing.subscribe();
        let mut result = if *closing.borrow_and_update() {
            Err(ResourceError::Cancelled)
        } else {
            self.acquire().await
        };
        if result.is_ok() {
            if *closing.borrow_and_update() {
                result = Err(ResourceError::Cancelled);
            } else {
                result = tokio::select! {
                    biased;
                    _ = closing.changed() => Err(ResourceError::Cancelled),
                    result = self.toolsets.start() => result.map_err(ResourceError::Host),
                };
            }
        }
        if result.is_ok() {
            self.started.send_replace(Some(Ok(())));
            while !*closing.borrow_and_update() {
                if closing.changed().await.is_err() {
                    break;
                }
            }
        } else {
            let _ = self.lifecycle.lock().close();
            self.closing.send_replace(true);
        }
        let cleanup = self.cleanup().await;
        if self.started.borrow().is_none() {
            // Startup reports its original failure only after partial acquisition settles.
            self.started.send_replace(Some(result));
        }
        self.closed.send_replace(Some(cleanup));
    }

    fn request_close(self: &Arc<Self>) {
        let action = self.lifecycle.lock().close();
        self.closing.send_replace(true);
        if let CloseOwner::Close { wait_start: false } = action {
            match (self.env.spawner)() {
                Ok(spawner) => {
                    let inner = self.clone();
                    self.tasks.spawn(&*spawner, "unopened run resources close", async move {
                        let result = inner.cleanup().await;
                        inner.closed.send_replace(Some(result));
                    });
                }
                Err(error) => {
                    self.closed.send_replace(Some(Err(HostError::new(error).into())));
                }
            }
        }
    }

    fn ready(&self) -> Result<&Opened, ResourceError> {
        match self.started.borrow().as_ref() {
            Some(Ok(())) => self.opened.get().ok_or(ResourceError::NotReady),
            _ => Err(ResourceError::NotReady),
        }
    }
}

impl Drop for RunResources {
    fn drop(&mut self) {
        self.0.request_close();
    }
}

struct OpeningWait {
    inner: Arc<Inner>,
    armed: bool,
}
impl OpeningWait {
    fn disarm(&mut self) {
        self.armed = false;
    }
}
impl Drop for OpeningWait {
    fn drop(&mut self) {
        if self.armed {
            self.inner.request_close();
        }
    }
}

async fn wait(reply: &Reply) -> Outcome {
    let mut reply = reply.subscribe();
    loop {
        if let Some(result) = reply.borrow_and_update().clone() {
            return result;
        }
        if reply.changed().await.is_err() {
            return Err(ResourceError::Closed);
        }
    }
}

impl lk_agents_abi::agent::resources::RunResources for RunResources {
    type Env = ProcessEnv;
    type Namespace = Namespace;
    type Activation = ContextStore;

    fn new(
        env: &ProcessEnv,
        namespace: Option<Arc<Namespace>>,
        thread_id: Option<String>,
        activation: Option<ContextStore>,
        lane: String,
        toolsets: Vec<Arc<dyn ToolsetHooks>>,
    ) -> Result<Self, ResourceError> {
        let plan = StoragePlan::new(namespace.is_some(), thread_id.is_some(), activation.is_some())
            .ok_or(ResourceError::ThreadStorage)?;
        Ok(Self(Arc::new(Inner {
            env: env.clone(),
            lifecycle: Mutex::default(),
            plan,
            inputs: Mutex::new(Some(Inputs { namespace, thread_id, activation, lane })),
            opened: OnceLock::new(),
            toolsets: crate::toolsets::group::Group::new(env, toolsets),
            hydration: Mutex::default(),
            tasks: Arc::default(),
            closing: watch::channel(false).0,
            started: watch::channel(None).0,
            closed: watch::channel(None).0,
        })))
    }

    fn open(&self) -> Result<BoxFuture<'static, Outcome>, ResourceError> {
        let spawner = (self.0.env.spawner)().map_err(HostError::new)?;
        let action = self.0.lifecycle.lock().start();
        match action {
            StartOwner::Closed => return Err(ResourceError::Closed),
            StartOwner::Join => {}
            StartOwner::Start => {
                let inner = self.0.clone();
                self.0.tasks.spawn(&*spawner, "text run resources", inner.run());
            }
        }
        let mut guard = OpeningWait { inner: self.0.clone(), armed: true };
        Ok(Box::pin(async move {
            let result = wait(&guard.inner.started).await;
            guard.disarm();
            result
        }))
    }

    fn namespace(&self) -> Result<Arc<Namespace>, ResourceError> {
        self.0.ready()?.namespace.clone().ok_or(ResourceError::NotReady)
    }
    fn thread_id(&self) -> Result<String, ResourceError> {
        self.0.ready()?.thread_id.clone().ok_or(ResourceError::NotReady)
    }
    fn activation(&self) -> Result<ContextStore, ResourceError> {
        self.0.ready()?.activation.clone().ok_or(ResourceError::NotReady)
    }
    fn closed(&self) -> bool {
        self.0.closed.borrow().is_some()
    }
    fn retain_hydration(&self, view: Arc<dyn ContextHydration>) {
        if !self.0.plan.own_activation {
            return;
        }
        let retired = {
            let mut hydration = self.0.hydration.lock();
            if hydration.closed {
                Some(view)
            } else {
                hydration.views.push(view);
                None
            }
        };
        if let Some(view) = retired
            && let Err(error) = view.release()
        {
            log::warn!("the ended run's context hydration could not release: {error}");
        }
    }
    fn aclose(&self) -> BoxFuture<'static, Outcome> {
        self.0.request_close();
        let inner = self.0.clone();
        Box::pin(async move { wait(&inner.closed).await })
    }
    fn rollback(&self) -> BoxFuture<'static, ()> {
        let closed = self.aclose();
        Box::pin(async move {
            if let Err(error) = closed.await {
                log::warn!("the failed run's resources could not close: {error}");
            }
        })
    }
}
