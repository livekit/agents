fn emit(source: &str) -> Result<String, String> {
    let declarations = crate::decl::tests::read(source)?;
    Ok(super::emit(&declarations, &[])?.remove(0).text)
}

#[test]
fn optional_generic_context_preserves_typed_and_context_free_call_forms() {
    let source = r#"
        #[crate::host(class(name = "_core_Activation"))]
        pub trait Activation {
            #[host(call)]
            fn bind(&self, context: Option<crate::HostRef>) -> crate::HostRef;
        }
        #[crate::host(facade(path = "livekit.agents.storage", native = Activation))]
        pub trait Store {
            #[host(
                annotation(context = ".storage_session.Context_T | None", ret = ".storage_session.BoundContext[Context_T] | .storage_session.BoundContext[None]"),
                overloads(with_context(context = ".storage_session.Context_T", ret = ".storage_session.BoundContext[Context_T]"),
                without_context(ret = ".storage_session.BoundContext[None]")))]
            fn bind(&self, #[host(default = None)] context: Option<crate::HostRef>) -> crate::HostRef;
        }
    "#;
    let python = emit(source).unwrap();
    assert!(
        python.contains(
            "@overload\n    def bind(self, context: Context_T) -> BoundContext[Context_T]:"
        )
    );
    assert!(python.contains("@overload\n    def bind(self) -> BoundContext[None]:"));
    assert!(python.contains("context: Context_T | None = None"));
    assert!(python.contains("self._native.bind(context)"));
    assert!(emit(&source.replace("without_context(ret =", "without_context(unknown =")).is_err());
}

#[test]
fn static_factories_forward_without_an_instance() {
    let python = emit(
        r#"
        #[crate::host(class(name = "_core_View"))]
        pub trait View: Sized {
            #[host(call)] fn decode(data: String) -> String;
        }
        #[crate::host(facade(path = "livekit.agents.storage", native = View))]
        pub trait Snapshot {
            fn decode(data: String) -> String;
        }
    "#,
    )
    .unwrap();
    assert!(python.contains("@staticmethod\n    def decode(data: str) -> str:"));
    assert!(python.contains("return _core._core_View.decode(data)"));
    assert!(!python.contains("self._native.decode"));
}

const PARTITION_KEY: &str = r#"
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
    /// One agent's partition.
    #[crate::host(facade(path = "livekit.agents.storage", native = Handle, from_native))]
    pub trait Partition {
        #[host(new)]
        fn new(agent: String) -> Self;
        /// The partition's history.
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
    let text = emit(PARTITION_KEY).unwrap();
    assert!(text.contains(
        "    async def history(self) -> ChatContext:\n        \"\"\"The partition's history.\"\"\"\n        return ChatContext([x.payload for x in await self._native.history(self._agent)])\n"
    ));
    assert!(text.contains(
        "    async def tasks(self, *, limit: int = 50) -> list[str]:\n        return await self._native.tasks(self._agent, limit=limit)\n"
    ));
    assert!(text.contains("from ..llm import ChatContext\n"));
}

// Catches: a facade given its native object building another, or its parameters not kept.
#[test]
fn a_facade_from_its_native_object_keeps_it_and_its_parameters() {
    let text = emit(PARTITION_KEY).unwrap();
    assert!(text.contains(
        "    def __init__(self, native: _core._core_Handle, agent: str) -> None:\n        self._native = native\n        self._agent = agent\n"
    ));
}

// Catches: a native getter answered as a property where the Python class has a method, a setter
// not assigning the native property, or a chained call not answering `self`.
#[test]
fn getters_setters_and_chained_calls_read_as_the_python_class() {
    let text = emit(PARTITION_KEY).unwrap();
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
    let text = emit(PARTITION_KEY).unwrap();
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
    let source = PARTITION_KEY.replace(
        "async fn tasks(&self, #[host(kw, default = 50)] limit: i64) -> Vec<String>;\n        #[host(native",
        "async fn tasks(&self, agent: String, #[host(kw, default = 50)] limit: i64) -> Vec<String>;\n        #[host(native",
    );
    let error = emit(&source).unwrap_err();
    assert!(error.contains("bind(agent) is bound, not a parameter"), "{error}");
}

// Catches: a base parameter sent to the native (which lacks it) or dropped instead of passed on
// to the base's method of the same name.
#[test]
fn a_base_parameter_goes_to_the_base_method_not_the_native() {
    let source = r#"
        #[crate::host(class(name = "_core_Client"))]
        pub trait Client {
            #[host(call)]
            fn configure(&self, #[host(kw, default = None)] voice: Option<String>);
        }
        #[crate::host(facade(path = "livekit.agents.tts", native = Client))]
        pub trait Configured {
            #[host(base(note))]
            fn configure(
                &self,
                #[host(kw, default = None)] voice: Option<String>,
                #[host(default = None)] note: Option<String>,
            );
        }
    "#;
    let python = emit(source).unwrap();
    assert!(python.contains(
        "        self._native.configure(voice=voice)\n        super().configure(note=note)\n"
    ));
}

const PACER: &str = r#"
    use std::sync::Arc;
    #[crate::host(proxy(host = "livekit.agents.tokenize.SentenceStream"))]
    pub trait Sentences: Send + Sync {
        #[host(notify, name = "push_text")]
        fn push_text(&self, text: String);
    }
    #[crate::host(record(module = "tokenizer", extract, emit))]
    pub struct Token { pub token: String }
    #[crate::host(class(name = "_core_Paced"))]
    pub trait Paced: Sized {
        #[host(env)]
        type Env;
        #[host(new)]
        fn new(env: &Self::Env, sent_stream: Arc<dyn Sentences>, #[host(kw, default = 300)] limit: u32) -> Self;
        #[host(await)]
        async fn next(&self) -> Result<Option<String>, crate::errors::ApiError>;
        #[host(await)]
        async fn aclose(&self);
        #[host(call)]
        fn begin(&self, text: String) -> u64;
        #[host(call)]
        fn end(&self, token: u64);
    }
    /// A paced stream.
    #[crate::host(facade(path = "livekit.agents.tts", base = "tokenize.SentenceStream", native = Paced))]
    pub trait Wrapper {
        #[host(delegate)]
        type SentStream: Sentences;
        #[host(new)]
        fn new(sent_stream: Self::SentStream, #[host(kw, default = 300)] limit: u32) -> Self;
        #[host(via = SentStream)]
        fn push_text(&self, text: String);
        #[host(none_ends_iteration, wrap = Token::token)]
        async fn __anext__(&self) -> Option<Token>;
        async fn aclose(&self);
        #[host(native = begin, exit = end)]
        fn hold(&self, text: String) -> u64;
    }
"#;

// Catches: the constructor's ABI default retyped, the delegate not kept, or the native built
// from positional arguments past a keyword.
#[test]
fn the_constructor_takes_the_abi_defaults_keeps_the_delegate_and_builds_the_native() {
    let text = emit(PACER).unwrap();
    assert!(text.contains("class Wrapper(SentenceStream):\n    \"\"\"A paced stream.\"\"\""));
    assert!(text.contains("from ..tokenize import SentenceStream\n"));
    assert!(text.contains(
        "    def __init__(self, sent_stream: SentenceStream, *, limit: int = 300) -> None:\n        super().__init__()\n        self._sent_stream = sent_stream\n        self._native = _core._core_Paced(sent_stream, limit=limit)\n"
    ));
}

#[test]
fn a_delegate_method_forwards_and_none_ends_iteration_wraps_the_value() {
    let text = emit(PACER).unwrap();
    assert!(text.contains(
        "    def push_text(self, text: str) -> None:\n        self._sent_stream.push_text(text)\n"
    ));
    assert!(text.contains(
        "    async def __anext__(self) -> Token:\n        if (value := await self._native.next()) is None:\n            raise StopAsyncIteration\n        return Token(token=value)\n"
    ));
    assert!(text.contains("from .tokenizer import Token\n"));
    assert!(
        text.contains("    async def aclose(self) -> None:\n        await self._native.aclose()\n")
    );
}

#[test]
fn a_begin_end_pair_is_an_async_with() {
    let text = emit(PACER).unwrap();
    assert!(text.contains(
        "    @contextlib.asynccontextmanager\n    async def hold(self, text: str) -> AsyncIterator[int]:\n        value = self._native.begin(text)\n        try:\n            yield value\n        finally:\n            self._native.end(value)\n"
    ));
    assert!(text.contains("import contextlib\n"));
}

#[test]
fn a_facade_method_the_native_lacks_is_refused() {
    let source = PACER.replace(
        "async fn aclose(&self);\n        #[host(native",
        "async fn close(&self);\n        #[host(native",
    );
    assert!(emit(&source).unwrap_err().contains("has no method close"));
}

const STREAM: &str = r#"
    #[crate::host(class(name = "Chain"))]
    pub trait Chain: Sized {
        #[host(env)]
        type Env;
        #[host(init)]
        fn new(env: &Self::Env, members: Vec<String>, #[host(kw, default = 1.0)] timeout: f64) -> Self;
        #[host(getter)]
        fn model(&self) -> String;
        #[host(call, name = "_core_conn_options")]
        fn conn_options(&self) -> u32;
        #[host(await, name = "_core_native_synthesize")]
        async fn synthesize(&self, input_text: String, #[host(default = None)] conn_options: Option<u32>) -> Result<u64, crate::errors::ApiError>;
    }
    #[crate::host(facade(path = "livekit.agents.tts", extended, native = Chain, refuse(synthesize), node = "session.tts_node", keep(members)))]
    pub trait Adapter {
        #[host(new)]
        fn new(members: Vec<String>, #[host(kw, default = 1.0)] timeout: f64) -> Self;
        #[host(getter)]
        fn model(&self) -> String;
    }
    #[crate::host(facade(path = "livekit.agents.tts", base = "tts.ChunkedStream", native = Chain, stream(op = synthesize, owner = "_tts", native_retries)))]
    pub trait AdapterStream {}
"#;

// Catches: an engine not built by the constructor, a stream retried on top of a native
// chain, a hook the native runs left overridable, or a native getter shadowed by a forward.
#[test]
fn a_stream_opens_its_owners_native_operation_and_the_base_pumps_it() {
    let text = emit(STREAM).unwrap();
    assert!(text.contains(
        "class AdapterStream(ChunkedStream):\n    _core_native_retries = True\n    _tts: tts.Adapter\n\n    async def _core_open(self) -> int:\n        return await self._tts._core_native_synthesize(self._input_text, self._conn_options)\n\n    _run = ChunkedStream._core_run_native\n"
    ));
    assert!(text.contains("    from .. import tts\n"));
    assert!(text.contains("class Adapter(_core.Chain):\n"));
    assert!(text.contains(
        "    def __init__(self, members: Sequence[str], *, timeout: float = 1.0) -> None:\n        _core.Chain.__init__(self, members, timeout=timeout)\n        self._members = members\n"
    ));
    assert!(text.contains(
        "    def __init_subclass__(cls, **kwargs: Any) -> None:\n        super().__init_subclass__(**kwargs)\n        _bindings.refuse_overrides(cls, (\"synthesize\",), \"session.tts_node\")\n"
    ));
    assert!(!text.contains("def model"), "the native getter is the class's own");
}

// Catches: a per-call argument retyped beside its native operation, or a stream retried on
// top of a native operation that retries itself.
#[test]
fn a_provider_method_takes_the_operations_per_call_arguments_and_the_stream_keeps_them() {
    let source = STREAM.replace(
        "#[host(await, name = \"_core_native_synthesize\")]",
        "#[host(await, name = \"_core_native_listen\")]\n            async fn listen(&self, #[host(kw, default = None)] language: Option<String>, #[host(default = None)] offset: Option<f64>, #[host(default = None)] conn_options: Option<u32>) -> Result<u64, crate::errors::ApiError>;\n            #[host(await, name = \"_core_native_synthesize\")]",
    ) + r#"
    #[crate::host(facade(path = "livekit.agents.tts", base = "tts.ChunkedStream", native = Chain, stream(op = listen, owner = "_tts", native_retries, method = "listen", base_args(conn_options), state(offset), default(conn_options = "..types.DEFAULT"), not_given)))]
    pub trait Listening {}
    "#;
    let text = emit(&source).unwrap();
    assert!(text.contains(
        "    def listen(\n        self,\n        *,\n        language: NotGivenOr[str] = NOT_GIVEN,\n        conn_options: int = DEFAULT,\n    ) -> ChunkedStream:\n        return Listening(\n            tts=cast(\"tts.Adapter\", self),\n            language=language if is_given(language) else None,\n            conn_options=conn_options,\n        )\n"
    ));
    assert!(text.contains(
        "    def __init__(self, *, tts: tts.Adapter, language: str | None, conn_options: int) -> None:\n        super().__init__(tts=tts, conn_options=conn_options)\n        self._language: str | None = language\n"
    ));
    assert!(text.contains(
        "return await self._tts._core_native_listen(\n            language=self._language,\n            offset=self._offset,\n            conn_options=self._conn_options,\n        )\n"
    ));
    assert!(text.contains("from typing import TYPE_CHECKING, Any, cast\n"));
}

// Catches: the provider method taking the operation's name for a renamed parameter, a
// parameter not given defaulting to `None` rather than the owner's policy, or a facade named
// as its base subclassing itself.
#[test]
fn a_provider_method_renames_and_takes_the_owners_default() {
    let source = STREAM
        .replace("extended, native = Chain", "extended, name = \"TTS\", base = \"tts.TTS\", native = Chain")
        .replace(
            "stream(op = synthesize, owner = \"_tts\", native_retries)",
            "stream(op = synthesize, owner = \"_tts\", method = \"synthesize\", base_args(input_text, conn_options), rename(input_text = \"text\"), provider_default(conn_options))",
        );
    let text = emit(&source).unwrap();
    assert!(text.contains("class TTS(_core.Chain, BaseTTS):\n"));
    assert!(text.contains(
        "    def synthesize(self, text: str, conn_options: NotGivenOr[int] = NOT_GIVEN) -> ChunkedStream:\n        return AdapterStream(\n            tts=cast(\"tts.TTS\", self),\n            input_text=text,\n            conn_options=conn_options if is_given(conn_options) else self._core_conn_options(),\n        )\n"
    ));
}
