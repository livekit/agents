//! The facade forms over a native object handed in: bound arguments, wrapped results, per-call
//! delegates, context managers through a delegate, setters and chained calls, foreign bases.

fn emit(source: &str) -> Result<String, String> {
    let declarations = crate::decl::tests::read(source)?;
    Ok(super::emit(&declarations, &[])?.remove(0).text)
}

const LANE: &str = r#"
    #[crate::host(record(module = "rows", extract, emit))]
    pub struct Row { pub payload: String }
    #[crate::host(class(name = "_core_Transaction"))]
    pub trait Transaction: Sized {
        #[host(await)]
        async fn execute(&self, sql: String);
        #[host(await)]
        async fn commit(&self);
        #[host(call)]
        fn rollback(&self);
    }
    #[crate::host(class(name = "_core_Handle"))]
    pub trait Handle: Sized {
        #[host(class(name = "_core_Transaction"))]
        type Opened: Transaction;
        #[host(await)]
        async fn history(&self, agent: String) -> Vec<Row>;
        #[host(await)]
        async fn tasks(&self, agent: String, #[host(kw, default = 50)] limit: i64) -> Vec<String>;
        #[host(await)]
        async fn transaction(&self, independent: bool) -> Self::Opened;
        #[host(getter)]
        fn closed(&self) -> bool;
        #[host(getter)]
        fn allow(&self) -> bool;
        #[host(setter, name = "allow")]
        fn set_allow(&self, value: bool);
        #[host(call)]
        fn interrupt(&self, #[host(kw, default = false)] force: bool);
    }
    /// One agent's lane.
    #[crate::host(facade(path = "livekit.agents.storage", native = Handle, from_native))]
    pub trait Lane {
        #[host(new)]
        fn new(agent: String) -> Self;
        /// The lane's history.
        #[host(bind(agent = "_agent"), returns = "..llm.ChatContext", each = Row::payload)]
        async fn history(&self) -> crate::HostRef;
        #[host(bind(agent = "_agent"))]
        async fn tasks(&self, #[host(kw, default = 50)] limit: i64) -> Vec<String>;
        #[host(native = closed)]
        fn done(&self) -> bool;
        #[host(getter)]
        fn allow(&self) -> bool;
        #[host(setter = allow)]
        fn set_allow(&self, value: bool);
        #[host(chain)]
        fn interrupt(&self, #[host(kw, default = false)] force: bool);
    }
    #[crate::host(facade(path = "livekit.agents.storage", native = Handle, extended))]
    pub trait Store {
        #[host(delegate)]
        type Activation: Handle;
        /// A transaction.
        #[host(via = Activation, native = transaction, exit = Transaction::commit,
               abort = Transaction::rollback, returns = "..storage.Transaction")]
        async fn _transaction(&self, #[host(kw)] independent: bool) -> crate::HostRef;
    }
"#;

// Catches: a bound argument left as a parameter, a list result handed over unmapped, or an
// awaited native wrapped before it is awaited.
#[test]
fn a_bound_argument_comes_from_self_and_each_row_is_mapped_into_the_host_type() {
    let text = emit(LANE).unwrap();
    assert!(text.contains(
        "    async def history(self) -> ChatContext:\n        \"\"\"The lane's history.\"\"\"\n        return ChatContext([x.payload for x in await self._native.history(self._agent)])\n"
    ));
    assert!(text.contains(
        "    async def tasks(self, *, limit: int = 50) -> list[str]:\n        return await self._native.tasks(self._agent, limit=limit)\n"
    ));
    assert!(text.contains("from ..llm import ChatContext\n"));
}

// Catches: a facade given its native object building another, or its parameters not kept.
#[test]
fn a_facade_from_its_native_object_keeps_it_and_its_parameters() {
    let text = emit(LANE).unwrap();
    assert!(text.contains(
        "    def __init__(self, native: _core._core_Handle, agent: str) -> None:\n        self._native = native\n        self._agent = agent\n"
    ));
}

// Catches: a native getter answered as a property where the Python class has a method, a setter
// not assigning the native property, or a chained call not answering `self`.
#[test]
fn getters_setters_and_chained_calls_read_as_the_python_class() {
    let text = emit(LANE).unwrap();
    assert!(text.contains("    def done(self) -> bool:\n        return self._native.closed\n"));
    assert!(text.contains(
        "    @allow.setter\n    def allow(self, value: bool) -> None:\n        self._native.allow = value\n"
    ));
    assert!(text.contains(
        "    def interrupt(self, *, force: bool = False) -> Self:\n        self._native.interrupt(force=force)\n        return self\n"
    ));
}

// Catches: an exit run after a failed block, an abort swallowing the error, or the entered native
// value handed to the block unwrapped.
#[test]
fn a_context_manager_through_a_delegate_commits_or_rolls_back_the_entered_value() {
    let text = emit(LANE).unwrap();
    assert!(text.contains(
        "    @contextlib.asynccontextmanager\n    async def _transaction(self, *, independent: bool) -> AsyncIterator[storage.Transaction]:\n        \"\"\"A transaction.\"\"\"\n        target = self._activation\n        value = await target.transaction(independent)\n        try:\n            yield storage.Transaction(value)\n        except BaseException:\n            value.rollback()\n            raise\n        await value.commit()\n"
    ));
}

const CALL: &str = r#"
    #[crate::host(class(name = "Session"))]
    pub trait Session: Sized {
        #[host(call)]
        fn filler_start(&self, call_id: String, #[host(kw)] delay: f64) -> u64;
        #[host(call)]
        fn filler_stop(&self, id: u64);
        #[host(await)]
        async fn hold(&self) -> u64;
        #[host(call)]
        fn release(&self, hold: u64);
    }
    #[crate::host(facade(path = "livekit.agents.voice.events", extended, base = "opentelemetry.trace.Span", foreign))]
    pub trait Context {
        /// The call's session, read at each use.
        #[host(delegate = "_call()._core")]
        type Call: Session;
        #[host(via = Call, native = filler_start, exit = filler_stop, bind(call_id = "function_call.call_id"))]
        async fn with_filler(&self, #[host(kw, default = 0.0)] delay: f64);
        #[host(via = Call, native = hold, exit = release, yields = "_call()", annotation(ret = "..voice.agent_session.VoiceSession[Any]"))]
        async fn foreground(&self) -> crate::HostRef;
    }
"#;

// Catches: a per-call delegate read twice (the exit reaching another session than the one that
// entered), or the block handed the native value where it receives a path of self.
#[test]
fn a_per_call_delegate_is_read_once_and_the_block_receives_the_declared_path() {
    let text = emit(CALL).unwrap();
    assert!(text.contains(
        "    async def with_filler(self, *, delay: float = 0.0) -> AsyncIterator[None]:\n        target = self._call()._core\n        value = target.filler_start(cast(Any, self).function_call.call_id, delay=delay)\n        try:\n            yield\n        finally:\n            target.filler_stop(value)\n"
    ));
    assert!(text.contains(
        "    async def foreground(self) -> AsyncIterator[VoiceSession[Any]]:\n        target = self._call()._core\n        value = await target.hold()\n        try:\n            yield self._call()\n        finally:\n            target.release(value)\n"
    ));
    assert!(!text.contains("_native"));
    assert!(
        text.contains(
            "    if TYPE_CHECKING:\n        def _call(self) -> Any: ...\n\n    @contextlib"
        )
    );
}

// Catches: a base outside livekit.agents imported as a framework module.
#[test]
fn a_foreign_base_is_imported_from_its_own_package() {
    let text = emit(CALL).unwrap();
    assert!(text.contains("from opentelemetry.trace import Span\n"));
    assert!(text.contains("class Context(Span):\n"));
}

#[test]
fn a_bound_argument_that_is_also_a_parameter_is_refused() {
    let source = LANE.replace(
        "async fn tasks(&self, #[host(kw, default = 50)] limit: i64) -> Vec<String>;\n        #[host(native",
        "async fn tasks(&self, agent: String, #[host(kw, default = 50)] limit: i64) -> Vec<String>;\n        #[host(native",
    );
    let error = emit(&source).unwrap_err();
    assert!(error.contains("bind(agent) is bound, not a parameter"), "{error}");
}
