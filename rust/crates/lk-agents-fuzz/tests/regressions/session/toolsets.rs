//! A cancelled activation close stays owned until the live session joins it at shutdown.

use std::cell::RefCell;
use std::sync::{Arc, OnceLock};

use lk_agents_abi::errors::HostError;
use lk_agents_abi::llm::toolsets::{self, ToolsetLifetime as _, ToolsetScopes as _};
use lk_agents_abi::telemetry::Cx;
use lk_agents_abi::voice::events::{HookName, UserTurnExceededEvent};
use lk_agents_abi::voice::session::VoiceSession as _;
use lk_agents_abi::voice::session_effects::{CompletedUserTurn, FillerReply, SessionHooks};
use lk_agents_abi::{BoxFuture, HostRef, RefRegistry, Spawner, SpawnerExt};
use lk_agents_fuzz::check::check_all;
use lk_agents_fuzz::drive::{Run, Setup, simulate};
use lk_agents_fuzz::input::Input;
use lk_agents_runtime::net::env::ProcessEnv;
use lk_agents_runtime::net::signal;
use lk_agents_runtime::toolsets::{lifetime::Lifetime, scopes::Scopes};
use lk_agents_runtime::voice::session::VoiceSession;

thread_local! {
    /// Constructors borrow the same tracked spawner as the session whose hooks own them.
    static SPAWNER: RefCell<Option<Arc<dyn Spawner>>> = const { RefCell::new(None) };
}

fn environment(session: &VoiceSession) -> ProcessEnv {
    SPAWNER.with(|spawner| *spawner.borrow_mut() = Some(session.spawner()));
    ProcessEnv {
        spawner: || Ok(SPAWNER.with(|spawner| spawner.borrow().clone().expect("session spawner"))),
        clock: session.time().clock(),
        ..ProcessEnv::current_runtime()
    }
}

struct Refs;
impl RefRegistry for Refs {
    fn release(&self, _: u64) {}
}

struct HeldClose {
    lifetime: Arc<dyn toolsets::Lifetime>,
    entered: signal::Producer<bool>,
    release: signal::Cell<bool>,
    closed: signal::Producer<bool>,
}

impl toolsets::ToolsetHooks for HeldClose {
    fn lifetime(&self) -> &Arc<dyn toolsets::Lifetime> {
        &self.lifetime
    }
    fn setup(
        &self,
        _: &Cx,
        _: Arc<dyn lk_agents_abi::invoke::Cancellation>,
    ) -> BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }
    fn close(&self, _: &Cx) -> BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async move {
            self.entered.set(true);
            self.release
                .wait_map(|release| release.then_some(()))
                .await
                .map_err(|_| HostError::new("the close boundary ended"))?;
            self.closed.set(true);
            Ok(())
        })
    }
    fn children(&self) -> Result<Option<Arc<dyn toolsets::LeaseGroup>>, HostError> {
        Ok(None)
    }
}

struct ObservedScopes {
    inner: Arc<Scopes>,
    shutdown: signal::Producer<bool>,
}

impl toolsets::ScopesPort for ObservedScopes {
    fn acquire(
        &self,
        scope: u64,
        tools: Vec<Arc<dyn toolsets::ToolsetHooks>>,
    ) -> BoxFuture<'_, Result<u64, HostError>> {
        toolsets::ScopesPort::acquire(&*self.inner, scope, tools)
    }
    fn discard(&self, activation: u64) -> BoxFuture<'_, Result<(), HostError>> {
        toolsets::ScopesPort::discard(&*self.inner, activation)
    }
    fn release(&self, scope: Option<u64>) -> BoxFuture<'_, Result<(), HostError>> {
        Box::pin(async move {
            if scope.is_none() {
                self.shutdown.set(true);
            }
            toolsets::ScopesPort::release(&*self.inner, scope).await
        })
    }
}

struct Hooks {
    scopes: OnceLock<Arc<dyn toolsets::ScopesPort>>,
}

impl SessionHooks for Hooks {
    fn on_enter<'a>(&'a self, _: &Cx, _: &'a HostRef) -> BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }
    fn on_exit<'a>(&'a self, _: &Cx, _: &'a HostRef) -> BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }
    fn toolset_scopes(&self) -> &Arc<dyn toolsets::ScopesPort> {
        self.scopes.get().expect("scopes are installed before shutdown")
    }
    fn toolset_resources(
        &self,
        _: &HostRef,
    ) -> Result<Vec<Arc<dyn toolsets::ToolsetHooks>>, HostError> {
        Ok(Vec::new())
    }
    fn tool_infos(
        &self,
        _: &HostRef,
    ) -> Result<Vec<lk_agents_abi::agent::FunctionToolInfo>, HostError> {
        Ok(Vec::new())
    }
    fn on_user_turn_completed<'a>(
        &'a self,
        _: &Cx,
        _: &'a HostRef,
        _: Vec<lk_agents_abi::ChatItem>,
        _: &'a str,
        _: f64,
    ) -> BoxFuture<'a, Result<CompletedUserTurn, HostError>> {
        Box::pin(async {
            Ok(CompletedUserTurn { user_text: None, items: None, stop_response: false })
        })
    }
    fn on_user_turn_exceeded<'a>(
        &'a self,
        _: &Cx,
        _: &'a HostRef,
        _: &'a UserTurnExceededEvent,
    ) -> BoxFuture<'a, Result<(), HostError>> {
        Box::pin(async { Ok(()) })
    }
    fn function_tools_executed(
        &self,
        _: &Cx,
        _: Vec<lk_agents_abi::FunctionCall>,
        outputs: Vec<lk_agents_abi::FunctionCallOutput>,
    ) -> BoxFuture<'_, Result<Vec<bool>, HostError>> {
        Box::pin(
            async move { Ok(outputs.into_iter().map(|output| output.reply_required).collect()) },
        )
    }
    fn filler_source<'a>(
        &'a self,
        _: &Cx,
        _: &'a HostRef,
        _: u32,
    ) -> BoxFuture<'a, Result<FillerReply, HostError>> {
        Box::pin(async { Ok(FillerReply::Declined) })
    }
    fn hook_failed(&self, hook: HookName, _: Option<lk_agents_abi::errors::CapturedError>) {
        panic!("the cleanup fixture hook failed: {hook:?}");
    }
}

// Catches: cancelling the close waiter removes its group, so the real host drain's later
// whole-session release returns while the toolset's authored close is still running.
#[test]
fn session_shutdown_joins_the_toolset_close_after_its_activation_waiter_is_cancelled() {
    simulate(async {
        let refs: Arc<dyn RefRegistry> = Arc::new(Refs);
        let hooks = Arc::new(Hooks { scopes: OnceLock::new() });
        let setup = Setup { tts: false, tools: Vec::new(), ..Default::default() };
        let mut run = Run::start_with(&setup, 0, Default::default(), |args, _| {
            args.hooks = Some(hooks.clone());
            args.agent_host = Some(HostRef::new(1, &refs));
            args.voice_agent = None;
        })
        .await;
        let session = run.registry.session().unwrap().clone();
        let env = environment(&session);
        let scopes = Arc::new(Scopes::new(&env));
        let (shutdown, shutting_down) = signal::signal(false);
        assert!(
            hooks.scopes.set(Arc::new(ObservedScopes { inner: scopes.clone(), shutdown })).is_ok()
        );
        let (entered, closing_entered) = signal::signal(false);
        let (release, released) = signal::signal(false);
        let (closed, is_closed) = signal::signal(false);
        let toolset = Arc::new(HeldClose {
            lifetime: Arc::new(Lifetime::new(&env)),
            entered,
            release: released,
            closed,
        });
        toolsets::ToolsetScopes::acquire(&*scopes, 0, vec![toolset]).await.unwrap();
        let releasing = scopes.clone();
        let activation = session
            .spawner()
            .spawn(async move { toolsets::ToolsetScopes::release(&*releasing, Some(0)).await });
        closing_entered.wait_map(|entered| entered.then_some(())).await.unwrap();
        activation.cancel();
        assert!(activation.await.is_err());

        // The host owns and drives aclose independently of the harness's runtime pump. Its first
        // poll initializes the shared close before another waiter.
        let mut first = session.aclose("user_initiated".into(), false).unwrap();
        assert!(
            std::future::poll_fn(|cx| std::task::Poll::Ready(first.as_mut().poll(cx).is_pending()))
                .await,
            "the close waits on the session"
        );
        let owner = session.spawner().spawn(first);
        let mut closing = session.aclose("user_initiated".into(), false).unwrap();
        let boundary_closed = is_closed.clone();
        let observer = session.spawner().spawn(async move {
            shutting_down.wait_map(|closing| closing.then_some(())).await.unwrap();
            assert!(
                std::future::poll_fn(|cx| std::task::Poll::Ready(
                    closing.as_mut().poll(cx).is_pending()
                ))
                .await,
                "session close returned before the cancelled activation's cleanup settled"
            );
            assert!(!boundary_closed.get());
            release.finish(true);
        });
        run.apply(&Input::Close { drain: false }).await;
        run.apply(&Input::AwaitExit {}).await;
        observer.await.unwrap();
        owner.await.unwrap();
        assert!(is_closed.get());
        let (failures, _) = check_all(&run.record);
        assert!(failures.is_empty(), "{failures:?}");
        SPAWNER.with(|spawner| *spawner.borrow_mut() = None);
    });
}
