//! `resolved_declaration_ir`: declarations in, IR or a refusal naming the declaration out.

use super::tree::Crate;
use super::*;

mod callables;
mod models;

const LIB: &str = "pub mod errors; pub mod llm; pub mod utils; pub use llm::chat::Item; pub use utils::host_ref::HostRef;";

const ERRORS: &str = r#"
#[crate::host(error(with = crate::errors::raise))]
pub struct ApiError;
pub fn raise(_: ApiError) {}
#[crate::host(error(kind = "ValueError"))]
pub struct Bad;
#[crate::host(error(kind = "MyError"))]
pub struct Worse;
#[crate::host(error(original))]
pub struct HostError;
#[crate::host(error(raised = "livekit.agents.Refused"))]
pub struct Refused;
#[crate::host(error(with = crate::errors::raise_provider))]
pub struct ProviderError;
pub fn raise_provider(_: ProviderError) {}
"#;

const CHAT: &str = r#"
use serde_json::{Map, Value};
pub type Extra = Map<String, Value>;
pub type Res<T> = Result<T, crate::errors::ApiError>;

#[derive(Host)]
#[python(module = "llm", name = "ChatChunk")]
#[binding(emit, extract)]
pub struct Chunk { pub text: String }

#[derive(Host)]
#[binding(emit, from_items)]
pub struct Report { pub n: u32 }

#[derive(Host)]
#[host(host_type = "livekit.agents.types.APIConnectOptions")]
#[binding(extract)]
pub struct Conn { pub timeout: f64 }

#[derive(Clone)]
pub enum Item { A }

#[crate::host(object(cell = "value", pyclass(eq, from_py_object)))]
impl Item {}

#[crate::host(proxy(host = "livekit.agents.llm.LLM"))]
pub trait Llm {}
"#;

pub(crate) fn read(llm: &str) -> Result<Declarations, String> {
    let krate = Crate::from_files(
        "lk_agents_abi",
        &[
            ("lib.rs", LIB),
            ("errors.rs", "mod api;\npub use api::*;"),
            ("errors/api.rs", ERRORS),
            ("llm.rs", "pub mod chat; pub mod class;"),
            ("llm/chat.rs", CHAT),
            ("llm/class.rs", llm),
            ("utils.rs", "pub mod stream; pub mod maybe; pub mod host_ref;"),
            (
                "utils/host_ref.rs",
                "#[crate::host(handwritten(extract, emit, host = \"object\"))] pub struct HostRef;",
            ),
            ("utils/maybe.rs", "pub enum Maybe<T> { NotGiven, Given(T) }"),
            ("utils/stream.rs", "pub trait StreamSource<T> {}\npub trait StreamSink<T> {}"),
        ],
    )?;
    resolve::declarations(&krate)
}

fn class(body: &str) -> String {
    format!(
        "use super::chat::{{Extra, Res, Chunk, Report, Llm, Conn}};\nuse crate::Item;\nuse std::sync::Arc;\n#[crate::host(class(name = \"Native\", native = Llm))]\npub trait Native: Sized {{ {body} }}"
    )
}

fn refuse(body: &str) -> String {
    match read(&class(body)) {
        Ok(d) => panic!("expected a refusal of `{body}`, got {:?}", d.classes[0].methods),
        Err(e) => e,
    }
}

// Catches: an alias read by its name instead of what it stands for (`Extra` as a record, `Res<T>`
// without its bound argument), a re-exported record not found through `pub use`, a source taken
// from the signature instead of its declaration, the error's conversion not resolved in the
// error's own module, or a path through a private module (`errors::api`) that another crate
// cannot name: each item is written by its shortest public path.
#[test]
fn aliases_generics_reexports_and_declared_kinds_resolve() {
    let decls = read(&class(
        r#"
        #[host(source(item = Chunk, class = "NativeStream"))]
        type Chunks;
        #[host(new)] fn new(model: String, #[host(kw, default = None)] extra: Option<Extra>) -> Res<Self>;
        #[host(call)] fn chat(&self, items: &[Item], reporter: Option<Arc<dyn Llm>>, conn: Conn) -> Res<Self::Chunks>;
        #[host(getter)] fn report(&self) -> Report;
        #[host(await)] async fn ping(&self, #[host(default = 1.5)] timeout: f64);
        "#,
    ))
    .expect("resolves");
    let c = &decls.classes[0];
    assert_eq!((c.name.as_str(), c.family.as_str()), ("Native", "llm"));
    assert_eq!(c.trait_path, "::lk_agents_abi::llm::class::Native");
    assert_eq!(c.native, ["::lk_agents_abi::llm::chat::Llm"]);
    let new = &c.methods[0];
    assert_eq!(new.ret.ok, Ty::This);
    assert!(new.ret.err.as_ref().is_some_and(|e| e.rust == "::lk_agents_abi::errors::ApiError"));
    assert_eq!(new.params[1].ty, Ty::Optional(Box::new(Ty::JsonObject)));
    assert!(new.params[1].kw_only && new.params[1].default == Some(Literal::None));
    let chat = &c.methods[1];
    let Ty::List(item) = &chat.params[0].ty else { panic!("{:?}", chat.params[0].ty) };
    assert!(chat.params[0].by_ref);
    assert!(
        matches!(&**item, Ty::Record(r) if r.rust == "::lk_agents_abi::Item" && r.extract && r.emit && r.host == HostForm::CoreClass("Item".into()))
    );
    assert!(
        matches!(&chat.params[1].ty, Ty::Optional(p) if matches!(&**p, Ty::Proxy { host, .. } if host == "livekit.agents.llm.LLM"))
    );
    let external = HostForm::External {
        module: "livekit.agents.types".into(),
        name: "APIConnectOptions".into(),
    };
    assert!(
        matches!(&chat.params[2].ty, Ty::Record(r) if r.host == external && r.extract && !r.emit)
    );
    let Ty::Source { item, class } = &chat.ret.ok else { panic!("{:?}", chat.ret.ok) };
    assert_eq!(class, "NativeStream");
    assert!(
        matches!(&**item, Ty::Record(r) if r.host == HostForm::Model { module: "llm".into(), name: "ChatChunk".into() })
    );
    assert_eq!(c.methods[3].kind, MethodKind::Await);
}

// Catches: an awaited method declared `-> BoxFuture<'static, T>` started only when the host
// awaits it (a command it sends would then be ordered after later calls), its future's output
// read as the declared result instead of the future itself, or a call refused in a `Result`
// raising only once awaited.
#[test]
fn an_eager_await_runs_at_the_call_and_awaits_its_future() {
    let decls = read(&class(
        "#[host(await)] fn ready(&self, id: String) -> Result<crate::BoxFuture<'static, Res<u32>>, crate::errors::ApiError>;",
    ))
    .expect("resolves");
    let ready = &decls.classes[0].methods[0];
    assert!(ready.eager && ready.kind == MethodKind::Await && ready.refused.is_some());
    assert_eq!(ready.ret.ok, Ty::Int(crate::decl::Int::U32));
    let adapter = crate::adapter::emit(&decls).unwrap();
    let text = adapter.iter().map(|o| o.text.as_str()).collect::<String>();
    let (sent, awaited) = (text.find("let call = py.detach(|| <").unwrap(), text.find("call.await").unwrap());
    assert!(sent < text.find("future_into_py").unwrap() && sent < awaited, "{text}");
    let raised = text[sent..].find(".map_err(").unwrap() + sent;
    assert!(raised < text.find("future_into_py").unwrap(), "{text}");
    let err = refuse("#[host(await)] fn ready(&self) -> crate::BoxFuture<'_, Res<u32>>;");
    assert!(err.contains("BoxFuture<'static"), "{err}");
}

// Catches: an implementation's own object (a handle) taken as an input, or handed out as
// anything but the shell's class built from it.
#[test]
fn a_handle_is_handed_out_as_the_shells_class_and_never_taken_in() {
    let decls = read(&class(
        r#"
        #[host(handle(class = "Thread"))]
        type Thread;
        #[host(await)] async fn open(&self, id: String) -> Res<Self::Thread>;
        "#,
    ))
    .expect("resolves");
    let open = &decls.classes[0].methods[0];
    assert_eq!(open.ret.ok, Ty::Handle { assoc: "Thread".into(), class: "Thread".into() });
    let adapter = crate::adapter::emit(&decls).unwrap();
    let text = adapter.iter().map(|o| o.text.as_str()).collect::<String>();
    assert!(text.contains("crate::handles::Thread::from("), "{text}");
    let err = refuse(
        r#"#[host(handle(class = "Thread"))] type Thread; #[host(call)] fn close(&self, t: Self::Thread);"#,
    );
    assert!(err.contains("is not an input"), "{err}");
}

// Catches: a type accepted for what its last segment spells. `Value` here is the crate's own
// undeclared type, not `serde_json::Value`, and an instant has no declared host form.
#[test]
fn a_type_is_what_it_resolves_to_never_its_name() {
    let krate = Crate::from_files(
        "lk_agents_abi",
        &[
            ("lib.rs", "pub mod llm;"),
            ("llm.rs", "pub struct Value;\n#[crate::host(class)] pub trait C { #[host(call)] fn f(&self, v: Value); }"),
        ],
    )
    .expect("reads");
    let err = resolve::declarations(&krate).expect_err("an undeclared `Value` is refused");
    assert!(
        err.contains("llm.rs: C::f(v)")
            && err.contains("::lk_agents_abi::llm::Value")
            && err.contains("not a declared host type"),
        "{err}"
    );
    let err = refuse("#[host(call)] fn f(&self, d: std::time::Instant);");
    assert!(err.contains("std::time::Instant") && err.contains("no declared conversion"), "{err}");
}

// Catches: a record crossing in a direction it never declared, a proxy handed back to the host,
// or an error raised through anything but its declaration's one conversion (a per-variant kind
// or delegation flattened into one builtin).
#[test]
fn directions_and_errors_are_declared_or_refused() {
    let err = refuse("#[host(call)] fn f(&self, r: Report);");
    assert!(err.contains("Native::f(r)") && err.contains("not declared for input"), "{err}");
    let err = refuse("#[host(call)] fn f(&self) -> Arc<dyn Llm>;");
    assert!(err.contains("not returned to the host"), "{err}");
    let err = refuse("#[host(call)] fn f(&self) -> Result<String, std::fmt::Error>;");
    assert!(err.contains("Native::f") && err.contains("not an ABI declaration"), "{err}");
    let err = refuse("#[host(call)] fn f(&self) -> Result<String, crate::errors::Refused>;");
    assert!(err.contains("Native::f") && err.contains("a refusal the host raises"), "{err}");
    let decls = read(&class("#[host(call)] fn f(&self) -> Result<String, crate::errors::Worse>;"))
        .expect("resolves");
    let with = &decls.classes[0].methods[0].ret.err.as_ref().expect("declared").with;
    assert!(
        with.contains("<::lk_agents_abi::errors::Worse as ::lk_agents_abi::python::HostException>::to_py_err(&e)"),
        "{with}"
    );
}

// Catches: a method's kind guessed from its signature (an `async fn` declared `call`, a `&mut
// self` method, a constructor that does not build `Self`), two kinds at once, an unknown key, an
// associated type with no declared kind, or a default that is not a value of its parameter.
#[test]
fn ambiguous_or_unsupported_declarations_fail_generation() {
    for (body, want) in [
        ("#[host(call)] async fn f(&self);", "does not fit `call`"),
        ("#[host(call)] fn f(&mut self);", "owned by one host object"),
        ("#[host(getter)] fn f(&mut self) -> u32;", "does not fit `getter`"),
        ("#[host(function)] fn f();", "has no `function`"),
        ("#[host(new)] fn new() -> String;", "constructor returns `Self`"),
        ("#[host(call, getter)] fn f(&self) -> u32;", "exactly one of"),
        ("fn f(&self);", "exactly one of"),
        ("#[host(call, soon)] fn f(&self);", "`soon` is not a key here"),
        ("type T;", "declares one kind"),
        ("#[host(call)] fn f(&self, #[host(default = \"x\")] n: u32);", "is not a value of `u32`"),
        (
            "#[host(call)] fn f(&self, #[host(default = 1)] n: u32, m: u32);",
            "required positional parameter follows",
        ),
        ("#[host(call)] fn f(&self) -> Res<String, String>;", "takes 1 type argument"),
        ("#[host(call)] fn f(&self) -> Self::Missing;", "not a declared associated type"),
    ] {
        let err = refuse(body);
        assert!(err.contains("Native") && err.contains(want), "`{body}`: {err}");
    }
}

// Catches: a sink and its source handed out as two objects instead of one stream class, the sink
// trait's own calls dropped from that class, input taken without its record conversion, or a sink
// accepted anywhere but beside its source.
#[test]
fn a_sink_and_its_source_are_one_duplex_stream_class() {
    let control = "use super::chat::Chunk;\nuse crate::utils::stream::StreamSink;\n#[crate::host(sink)]\npub trait Audio: StreamSink<Chunk> { #[host(call)] fn retune(&self, #[host(kw, default = None)] model: Option<String>); }";
    let declare = |body: &str| {
        format!(
            "{}\npub mod audio {{ {control} }}",
            class(&format!(
                "#[host(sink(item = Chunk, class = \"Live\"))] type In: audio::Audio;\n#[host(source(item = Report, class = \"Live\"))] type Out;\n{body}"
            ))
        )
    };
    let decls = read(&declare(
        "#[host(call)] fn stream(&self) -> Result<(Self::In, Self::Out), crate::errors::ProviderError>;",
    ))
    .expect("resolves");
    let c = &decls.classes[0];
    assert!(matches!(&c.methods[0].ret.ok, Ty::Duplex { class, .. } if class == "Live"));
    assert_eq!(c.sinks[0].trait_path, "::lk_agents_abi::llm::class::audio::Audio");
    assert_eq!(c.sinks[0].methods[0].name, "retune");
    let out = crate::adapter::emit(&decls).expect("emits");
    let llm = &out.iter().find(|o| o.path == "llm.rs").expect("llm.rs").text;
    let root = &out.iter().find(|o| o.path == "mod.rs").expect("mod.rs").text;
    assert!(root.contains("m.add_class::<llm::Live>()?;"), "{root}");
    assert!(
        !llm.contains("pub struct Live(::lk_agents_runtime::net::source::NativeSource"),
        "{llm}"
    );
    for expected in [
        "pub struct Live(::std::boxed::Box<LiveImpl>, ::lk_agents_runtime::net::source::NativeSource<::lk_agents_abi::llm::chat::Report>);",
        "py.detach(|| ::lk_agents_abi::StreamSink::push(&*self.0, item)).map_err(|e: ::lk_agents_abi::errors::ProviderError|",
        "fn aclose<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {\n        crate::host::check_pid()?;\n        py.detach(|| ::lk_agents_abi::StreamSink::end_input(&*self.0));",
        "<LiveImpl as ::lk_agents_abi::llm::class::audio::Audio>::retune(&*self.0, model)",
        "{ let (sink, source) = out; Live(::std::boxed::Box::new(sink), ::lk_agents_runtime::net::source::NativeSource::new(::std::boxed::Box::new(source))) }",
    ] {
        assert!(llm.contains(expected), "{expected}\n{llm}");
    }
    let stub = crate::class_stub::emit(&decls).expect("stubs");
    let text = &stub[0].out.text;
    assert!(
        text.contains("class Live:\n    def push(self, item: ChatChunk) -> None: ...")
            && text.contains("    def retune(self, *, model: str | None = None) -> None: ...")
            && text.contains("    async def recv(self) -> dict[str, object] | None: ..."),
        "{text}"
    );
    for (body, refusal) in [
        ("#[host(call)] fn stream(&self) -> Self::In;", "returned beside its source"),
        (
            "#[host(call)] fn stream(&self) -> (Self::Out, Self::In);",
            "returned first beside its source",
        ),
        ("#[host(call)] fn other(&self);", "returned beside its source by a method"),
    ] {
        let error = read(&declare(body)).expect_err(body);
        assert!(error.contains(refusal), "{body}: {error}");
    }
}

// Catches: the adapter choosing a conversion by anything but the declared kind (a JSON object
// extracted by pyo3, a proxy taken as a value), a default dropped from the host signature, the
// error raised through anything but its declared conversion, a source handed out without the
// runtime's single-consumer adapter, or a class the shell need not select an implementation for.
#[test]
fn adapters_and_stubs_follow_the_declared_kinds() {
    let decls = read(&class(
        r#"
        #[host(source(item = Chunk, class = "NativeStream"))]
        type Chunks;
        #[host(new)] fn new(model: String, #[host(kw, default = None)] extra: Option<Extra>) -> Res<Self>;
        #[host(call)] fn chat(&self, items: &[Item], #[host(kw)] conn: Conn) -> Res<Self::Chunks>;
        #[host(getter)] fn report(&self) -> Report;
        #[host(await)] async fn ping(&self, #[host(default = 1.5)] timeout: f64);
        "#,
    ))
    .expect("resolves");
    let out = crate::adapter::emit(&decls).expect("emits");
    let root = &out.iter().find(|o| o.path == "mod.rs").expect("mod.rs").text;
    let llm = &out.iter().find(|o| o.path == "llm.rs").expect("llm.rs").text;
    assert!(root.starts_with("//! GENERATED FILE -- DO NOT EDIT."), "{root}");
    assert!(root.contains("type Native: ::lk_agents_abi::llm::class::Native;"), "{root}");
    assert!(
        root.contains("m.add_class::<llm::Native>()?;")
            && root.contains("m.add_class::<llm::NativeStream>()?;"),
        "{root}"
    );
    assert!(llm.contains("#[pyo3(signature = (model, *, extra = None))]"), "{llm}");
    assert!(
        llm.contains("extra: Option<Bound<'_, PyAny>>")
            && llm.contains("value_from_py(&v, \"extra\")"),
        "{llm}"
    );
    assert!(llm.contains(".map_err(|e: ::lk_agents_abi::errors::ApiError| <::lk_agents_abi::errors::ApiError as ::lk_agents_abi::python::HostException>::to_py_err(&e))?"), "{llm}");
    assert!(llm.contains("(&self.0, &items, conn)"), "{llm}");
    assert!(llm.contains("NativeStream(::lk_agents_runtime::net::source::NativeSource::new(::std::boxed::Box::new(out)))"), "{llm}");
    assert!(
        llm.contains("#[pyo3(signature = (timeout = 1.5))]")
            && llm.contains("future_into_py(py, async move"),
        "{llm}"
    );
    assert!(llm.contains("TypeId::of::<dyn ::lk_agents_abi::llm::chat::Llm>()"), "{llm}");
    let stub = &crate::class_stub::emit(&decls).expect("stubs")[0].out;
    assert_eq!(stub.path, "_core_llm.pyi");
    for want in [
        "def __new__(cls, model: str, *, extra: dict[str, object] | None = None) -> Self: ...",
        "    def chat(self, items: Sequence[Item], *, conn: APIConnectOptions) -> NativeStream: ...",
        "def report(self) -> dict[str, object]: ...",
        "async def ping(self, timeout: float = 1.5) -> None: ...",
        "async def recv(self) -> dict[str, object] | None: ...",
        "from livekit.agents.types import APIConnectOptions",
        "from .._core import Item",
    ] {
        assert!(stub.text.contains(want), "{want}\n{}", stub.text);
    }
}

// Catches: a `&mut self` class shared behind an `Arc` (the borrow would not compile, or a frozen
// class would hand out aliased `&mut`), functions emitted as methods, or a functions trait
// accepting a method.
#[test]
fn mutable_classes_and_functions_are_their_own_forms() {
    let parser = r#"
        #[crate::host(class(name = "Parser"))]
        pub trait Parser: Sized { #[host(new)] fn new(model: String) -> Self; #[host(call)] fn parse(&mut self, id: &str) -> Option<super::chat::Chunk>; }
        #[crate::host(functions)]
        pub trait Wire { #[host(function)] fn llm_usage_chunk(chunk: serde_json::Value) -> Option<super::chat::Chunk>; }
    "#;
    let decls = read(parser).expect("resolves");
    assert_eq!(
        (decls.classes[0].form, decls.classes[1].form),
        (Form::MutableClass, Form::Functions)
    );
    let out = crate::adapter::emit(&decls).expect("emits");
    let llm = &out.iter().find(|o| o.path == "llm.rs").expect("llm.rs").text;
    assert!(
        llm.contains("#[pyclass(name = \"Parser\"")
            && llm.contains("pub struct Parser(pub(crate) ParserImpl);"),
        "{llm}"
    );
    assert!(
        llm.contains("fn parse(&mut self, py: Python<'_>, id: ::std::string::String)")
            && llm.contains("(&mut self.0, &id)"),
        "{llm}"
    );
    assert!(llm.contains("#[pyfunction]\n#[pyo3(signature = (chunk))]\npub fn llm_usage_chunk(py: Python<'_>, chunk: Bound<'_, PyAny>)"), "{llm}");
    let root = &out.iter().find(|o| o.path == "mod.rs").expect("mod.rs").text;
    assert!(
        root.contains("m.add_function(wrap_pyfunction!(llm::llm_usage_chunk, m)?)?;")
            && root.contains("type Wire: ::lk_agents_abi::llm::class::Wire;"),
        "{root}"
    );
    let stub = &crate::class_stub::emit(&decls).expect("stubs")[0].out.text;
    assert!(
        stub.contains("\ndef llm_usage_chunk(chunk: object) -> dict[str, object] | None: ...")
            && !stub.contains("class Wire"),
        "{stub}"
    );
    let err = read("#[crate::host(functions)] pub trait W { #[host(call)] fn f(&self); }")
        .expect_err("a method is refused");
    assert!(err.contains("W: every method of #[host(functions)] is a `function`"), "{err}");
}

pub(crate) fn proxy(body: &str) -> String {
    format!(
        "use super::chat::{{Report, Chunk, Conn}};\nuse crate::errors::{{HostError, Refused}};\n#[crate::host(proxy(host = \"livekit.agents.Callbacks\"))]\n#[async_trait]\npub trait Callbacks: Send + Sync {{ {body} }}"
    )
}

fn refuse_proxy(body: &str) -> String {
    match read(&proxy(body)) {
        Ok(d) => panic!("expected a refusal of `{body}`, got {:?}", d.proxies),
        Err(e) => e,
    }
}

// Catches: a proxy call made outside the context its invocation captured, a host refusal read as
// a failure (or the reverse), a notification awaited, reordered (a task per call) or its failure dropped silently, an argument
// converted by anything but its declared kind, or a class parameter of a declared proxy taken
// through the in-place `HostProxy`.
#[test]
fn declared_proxies_call_the_host_in_their_context() {
    let decls = read(&format!(
        "{}\n#[crate::host(class)] pub trait Runner: Sized {{ #[host(new)] fn new(host: std::sync::Arc<dyn Callbacks>) -> Self; }}",
        proxy(
            r#"
            #[host(await)] async fn render(&self, #[host(context)] at: &Report) -> Result<String, HostError>;
            #[host(await, requires(tool_call))] async fn prepare(&self, #[host(context)] at: &Report, args: serde_json::Map<String, serde_json::Value>) -> Result<Result<Chunk, Refused>, HostError>;
            #[host(notify, name = "abort")] fn cancel(&self, #[host(context)] at: &Report);
            "#
        )
    ))
    .expect("resolves");
    let p = &decls.proxies[0];
    assert_eq!(
        (p.rust_name.as_str(), p.host.as_str(), p.family.as_str()),
        ("Callbacks", "livekit.agents.Callbacks", "llm")
    );
    assert!(p.async_trait);
    let kinds: Vec<_> = p.methods.iter().map(|m| (m.kind, m.name.as_str(), m.context)).collect();
    assert_eq!(
        kinds,
        [
            (ProxyCall::Await, "render", Some(0)),
            (ProxyCall::Await, "prepare", Some(0)),
            (ProxyCall::Notify, "abort", Some(0))
        ]
    );
    let unscoped = refuse_proxy(
        "#[host(await, requires(tool_call))] async fn f(&self) -> Result<(), HostError>;",
    );
    assert!(unscoped.contains("#[host(context)]"), "{unscoped}");
    let refusal = p.methods[1].ret.refusal.as_ref().expect("declared");
    assert_eq!((refusal.module.as_str(), refusal.class.as_str()), ("livekit.agents", "Refused"));
    assert_eq!(p.methods[1].ret.err.as_deref(), Some("::lk_agents_abi::errors::HostError"));
    let out = crate::adapter::emit(&decls).expect("emits");
    let llm = &out.iter().find(|o| o.path == "llm.rs").expect("llm.rs").text;
    for want in [
        "impl ::lk_agents_abi::llm::class::Callbacks for CallbacksProxy {",
        "#[::async_trait::async_trait]",
        "let started = ::lk_agents_runtime::host::call(&crate::host::Entry, \"Callbacks.render\", at, &[], ::std::boxed::Box::new(move |py| {",
        "\"Callbacks.prepare\", at, &[::lk_agents_abi::invoke::Scope::ToolCall], ",
        "let out = started.await;",
        "Ok((host.bind(py).getattr(\"render\")?",
        "::lk_agents_abi::python::extra_to_py(py, &args_arg)?.into_any()",
        "if e.is_instance(py, &py.import(\"livekit.agents\").and_then(|m| m.getattr(\"Refused\"))",
        "return Ok(Err(<::lk_agents_abi::errors::Refused as ::std::convert::From<String>>::from(e.value(py).to_string())));",
        "<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e)",
        "fn cancel(&self, at: &::lk_agents_abi::llm::chat::Report) {",
        "::lk_agents_runtime::host::notify(&self.notifier, \"Callbacks.abort\", at, &[], false, ::std::boxed::Box::new(move |py| {",
        "Ok((host.bind(py).getattr(\"abort\")?",
        "let host = crate::generated::llm::CallbacksProxy::from_host(&host)?;",
        "native::<dyn ::lk_agents_abi::llm::class::Callbacks>(obj)",
    ] {
        assert!(llm.contains(want), "{want}\n{llm}");
    }
}

// Catches: a provider's identity read per call instead of once, a call without its own context
// left with none (rather than the one captured when the proxy was taken), a bound value not
// reaching the awaited calls, or a returned host stream read as a value instead of pulled in the
// call's context.
#[test]
fn provider_proxies_read_once_bind_and_pull_host_streams() {
    let decls = read(&proxy(
        r#"
        #[host(read_once, name = "_core_label")] fn label(&self) -> String;
        #[host(notify)] fn prewarm(&self) {}
        #[host(bind)] fn with_conn(&self, _conn: &Report) -> Option<std::sync::Arc<dyn Callbacks>> { None }
        #[host(await)] async fn chat(&self, #[host(context)] at: &Report) -> Result<Box<dyn crate::utils::stream::StreamSource<Chunk>>, HostError>;
        #[host(await)] async fn count(&self) -> Result<u32, HostError>;
        "#,
    ))
    .expect("resolves");
    let p = &decls.proxies[0];
    assert!(p.captures_context());
    assert_eq!(p.bound().map(|b| b.name.as_str()), Some("conn"));
    let kinds: Vec<_> = p.methods.iter().map(|m| (m.kind, m.context)).collect();
    assert_eq!(
        kinds,
        [
            (ProxyCall::ReadOnce, None),
            (ProxyCall::Notify, None),
            (ProxyCall::Bind, None),
            (ProxyCall::Await, Some(0)),
            (ProxyCall::Await, None)
        ]
    );
    let out = crate::adapter::emit(&decls).expect("emits");
    let llm = &out.iter().find(|o| o.path == "llm.rs").expect("llm.rs").text;
    for want in [
        "let v = crate::host::read(obj, \"_core_label\")?;",
        "None => crate::host::capture_context(obj.py())?,",
        "fn label(&self) -> ::std::string::String {\n        ::std::clone::Clone::clone(&self.label)",
        "\"Callbacks.count\", &self.context, &[], ::std::boxed::Box::new(move |py| {",
        "Some(::std::sync::Arc::new(Self { conn: Some(::std::clone::Clone::clone(conn)), ..::std::clone::Clone::clone(self) }))",
        "if let Some(bound) = bound {\n                    args.push(",
        "crate::host::stream::HostStream::open(py, o, context, |o| Ok(o.extract::<::lk_agents_abi::llm::chat::Chunk>()?)",
    ] {
        assert!(llm.contains(want), "{want}\n{llm}");
    }
}

// Catches: a proxy call with no context to enter (or two), a failure swallowed into a default, an
// error or refusal with no declared conversion, a call kind guessed from the signature, or a
// value handed to the host that it cannot receive.
// Catches: a core stream handed to a host call converted as anything but its generated stream
// class, or that class missing from the proxy's family and registration.
#[test]
fn a_core_stream_reaches_a_host_call_as_its_generated_stream_class() {
    let decls = read(&proxy(
        "#[host(notify)] fn f(&self, #[host(source(class = \"ChunkStream\"))] s: Box<dyn crate::utils::stream::StreamSource<String>>);",
    ))
    .expect("resolves");
    let p = &decls.proxies[0];
    assert!(
        matches!(&p.methods[0].params[0].ty, Ty::Source { class, .. } if class == "ChunkStream")
    );
    assert_eq!(crate::adapter::proxy_sources(&decls, p).len(), 1);
    let text = crate::proxy::proxy(p).expect("emits");
    assert!(
        text.contains("::pyo3::Bound::new(py, ChunkStream(::lk_agents_runtime::net::source::NativeSource::new(s_arg)))"),
        "{text}"
    );
}

#[test]
fn incomplete_proxy_declarations_fail_generation() {
    for (body, want) in [
        (
            "#[host(read_once)] fn f(&self, #[host(context)] a: &Report) -> String;",
            "does not fit `read_once`",
        ),
        (
            "#[host(bind)] fn f(&self, c: &Report) -> Option<std::sync::Arc<dyn super::chat::Llm>>;",
            "returns `Option<Arc<dyn Self>>` of this trait",
        ),
        (
            "#[host(await)] async fn f(&self) -> Result<Box<dyn crate::utils::stream::StreamSource<Report>>, HostError>;",
            "not declared for input",
        ),
        (
            "#[host(notify)] fn f(&self, s: Box<dyn crate::utils::stream::StreamSource<Chunk>>);",
            "a host stream is not handed back to the host",
        ),
        (
            "#[host(await)] async fn f(&self, #[host(context)] a: &Report, #[host(context)] b: &Report) -> Result<String, HostError>;",
            "one parameter is the call's #[host(context)]",
        ),
        (
            "#[host(await)] async fn f(&self, #[host(context)] a: &Report) -> String;",
            "a host failure is reported",
        ),
        (
            "#[host(await)] async fn f(&self, #[host(context)] a: &Report) -> Result<String, std::fmt::Error>;",
            "not an ABI declaration",
        ),
        (
            "#[host(await)] async fn f(&self, #[host(context)] a: &Report) -> Result<Result<String, HostError>, HostError>;",
            "error(raised",
        ),
        (
            "#[host(await)] fn f(&self, #[host(context)] a: &Report) -> Result<String, HostError>;",
            "does not fit `await`",
        ),
        (
            "#[host(notify)] fn f(&self, #[host(context)] a: &Report) -> u32;",
            "does not fit `notify`",
        ),
        (
            "#[host(await, notify)] fn f(&self, #[host(context)] a: &Report);",
            "exactly one of await, call, notify, read_once, bind",
        ),
        (
            "#[host(await)] async fn f(&self, #[host(context)] a: &Report) -> Result<String, HostError>; fn g(&self);",
            "Callbacks::g: a proxy method declares exactly one of",
        ),
        (
            "#[host(await)] async fn f(&self, #[host(context)] a: &Report, c: Conn) -> Result<(), HostError>;",
            "f(c): record `::lk_agents_abi::llm::chat::Conn` is not declared for output",
        ),
        (
            "#[host(await)] async fn f(&self, #[host(context)] a: &Report) -> Result<Report, HostError>;",
            "not declared for input",
        ),
    ] {
        let err = refuse_proxy(body);
        assert!(err.contains("Callbacks") && err.contains(want), "`{body}`: {err}");
    }
}

#[test]
fn a_native_only_tap_is_registered_by_its_declared_class_and_other_traits_are_refused() {
    let declaration = r#"
        #[crate::host(native(refused = "expected the native recorder"))]
        pub trait Tap: Send + Sync {}
        #[crate::host(class(name = "Recorder", native = Tap))]
        pub trait Recorder: Sized {
            #[host(new)] fn new() -> Self;
        }
    "#;
    let declarations = read(declaration).expect("native-only tap resolves");
    assert_eq!(declarations.classes[0].native, ["::lk_agents_abi::llm::class::Tap"]);
    assert!(declarations.proxies.is_empty(), "a native-only tap must not call Python");
    let output = crate::adapter::emit(&declarations).unwrap();
    let class = output.iter().find(|output| output.path == "llm.rs").unwrap();
    assert!(class.text.contains("TypeId::of::<dyn ::lk_agents_abi::llm::class::Tap>()"));
    let plain = declaration
        .replace("#[crate::host(native(refused = \"expected the native recorder\"))]", "");
    assert!(
        read(&plain).unwrap_err().contains("`Tap` is not a #[host(proxy)] trait"),
        "an undeclared trait cannot cross"
    );
}

// Catches: a proxy an awaited call returns capturing its context where it is taken, off the host
// loop (no running loop there), instead of the context the returning call ran in.
#[test]
fn a_returned_proxy_runs_in_the_context_of_the_call_that_returned_it() {
    let decls = read(
        r#"
        use crate::errors::HostError;
        #[crate::host(proxy(host = "livekit.agents.Session"))]
        pub trait Session: Send + Sync {
            #[host(await)] fn next(&self) -> crate::BoxFuture<'_, Result<u32, HostError>>;
        }
        #[crate::host(proxy(host = "livekit.agents.Model"))]
        pub trait Model: Send + Sync {
            #[host(await)] fn open(&self) -> crate::BoxFuture<'_, Result<std::sync::Arc<dyn Session>, HostError>>;
        }
    "#,
    )
    .expect("resolves");
    let out = crate::adapter::emit(&decls).expect("emits");
    let llm = &out.iter().find(|o| o.path == "llm.rs").expect("llm.rs").text;
    assert!(
        llm.contains("let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();")
            && llm.contains("super::llm::SessionProxy::in_context(o, context)")
            && llm.contains("Some(context) => ::std::clone::Clone::clone(context),"),
        "{llm}"
    );
}

// Catches: a native implementation a proxy call passes found through a runtime registry instead
// of a generated downcast to the declared class implementing its trait.
#[test]
fn a_native_argument_crosses_as_the_declared_class_implementing_it() {
    let decls = read(
        r#"
        use crate::errors::HostError;
        #[crate::host(native(refused = "expected the native tap"))]
        pub trait Tap: std::any::Any + Send + Sync {}
        #[crate::host(class(name = "Recorder", native = Tap))]
        pub trait Recorder: Sized {
            #[host(new)] fn new() -> Self;
        }
        #[crate::host(proxy(host = "livekit.agents.Sinks"))]
        pub trait Sinks: Send + Sync {
            #[host(await)] fn attach(&self, tap: std::sync::Arc<dyn Tap>) -> crate::BoxFuture<'_, Result<(), HostError>>;
        }
    "#,
    )
    .expect("resolves");
    let out = crate::adapter::emit(&decls).expect("emits");
    let root = &out.iter().find(|o| o.path == "mod.rs").expect("mod.rs").text;
    let llm = &out.iter().find(|o| o.path == "llm.rs").expect("llm.rs").text;
    assert!(
        root.contains("impl NativeToHost for dyn ::lk_agents_abi::llm::class::Tap {")
            && root.contains("native.downcast::<<crate::Selected as Implementations>::Recorder>()")
            && root.contains("Py::new(py, llm::Recorder(native))"),
        "{root}"
    );
    assert!(llm.contains("pub struct Recorder(pub(crate) "), "{llm}");
    assert!(
        llm.contains(
            "<dyn ::lk_agents_abi::llm::class::Tap as super::NativeToHost>::to_host(py, tap_arg)?"
        ),
        "{llm}"
    );
}

// Catches: an ordered map or set refused as undeclared (TextSession state), a map's keys or a
// set's items crossing by anything but their declared kind, or a host proxy handed back to the
// host from an emitted record.
#[test]
fn records_carry_ordered_maps_sets_and_host_proxies() {
    let decls = read(
        r#"
        use std::collections::{BTreeMap, BTreeSet};
        use super::chat::Llm;
        #[crate::host(record(module = "session", extract, emit))]
        pub struct State { pub frames: BTreeMap<String, u32>, pub started: BTreeSet<String> }
        "#,
    )
    .expect("resolves");
    let shell = crate::records::conversions(&decls).expect("emits").text;
    for want in [
        "pub fn state_from_py(obj: &Bound<'_, PyAny>) -> PyResult<::lk_agents_abi::llm::class::State>",
        "cast::<::pyo3::types::PyDict>()?.iter().map(|(key, item)|",
        "collect::<PyResult<::std::collections::BTreeSet<_>>>()",
        "::pyo3::types::PyFrozenSet::new(py,",
    ] {
        assert!(shell.contains(want), "{want}\n{shell}");
    }
    let models = crate::records::python(&decls).expect("models");
    let session = &models.iter().find(|o| o.path == "session.py").expect("session.py").text;
    assert!(
        session.contains("frames: dict[str, int]") && session.contains("started: frozenset[str]"),
        "{session}"
    );
    let err = read(
        r#"use super::chat::Llm;
        #[crate::host(record(module = "session", emit))]
        pub struct Out { pub llm: std::sync::Arc<dyn Llm> }"#,
    )
    .expect_err("an emitted proxy is refused");
    assert!(err.contains("Out") && err.contains("a host proxy is not returned"), "{err}");
}

// Catches: application-saved state serialized by the host (where empty variants of a union are
// indistinguishable) instead of the record's own codec, or a persisted record that cannot cross
// back the way it was saved.
#[test]
fn a_persisted_record_saves_and_restores_through_its_rust_codec() {
    let decls = read(
        r#"#[crate::host(record(module = "session", extract, emit, persisted))]
        pub struct SessionState { pub active: String }"#,
    )
    .expect("resolves");
    let adapters = crate::adapter::emit(&decls).expect("emits");
    let llm = &adapters.iter().find(|o| o.path == "llm.rs").expect("llm.rs").text;
    let root = &adapters.iter().find(|o| o.path == "mod.rs").expect("mod.rs").text;
    assert!(
        llm.contains("pub fn _core_persist_session_state(record: Bound<'_, PyAny>) -> PyResult<String> {\n    let record = crate::generated::records::session::session_state_from_py(&record)?;\n    ::serde_json::to_string(&record)")
            && llm.contains("let record: ::lk_agents_abi::llm::class::SessionState = ::serde_json::from_str(data)")
            && llm.contains("crate::generated::records::session::session_state_to_py(py, record)")
            && root.contains("wrap_pyfunction!(llm::_core_restore_session_state, m)"),
        "{llm}\n{root}"
    );
    let models = crate::records::python(&decls).expect("models");
    let model = &models.iter().find(|o| o.path == "session.py").expect("session.py").text;
    assert!(
        model.contains("return _core._core_persist_session_state(self)")
            && model.contains("def _from_json(cls, data: str) -> SessionState:"),
        "{model}"
    );
    let err = read(
        r#"#[crate::host(record(module = "session", emit, persisted))]
        pub struct Out { pub active: String }"#,
    )
    .expect_err("an emit-only persisted record is refused");
    assert!(err.contains("Out: a persisted record is a struct crossing both ways"), "{err}");
}

// Catches: a class handing out another declared class's instance without tying the two
// implementations together, or an async-iterable class whose end is not StopAsyncIteration
// (the A2A client's acknowledged `Reply`).
#[test]
fn a_class_hands_out_an_async_iterable_instance_of_another_class() {
    let decls = read(
        r#"
        use super::chat::{Chunk, Res};
        #[crate::host(class(name = "Reply"))]
        pub trait Reply: Sized {
            #[host(getter)] fn task_id(&self) -> String;
            #[host(next)] async fn next(&self) -> Res<Option<Chunk>>;
        }
        #[crate::host(class(name = "Client"))]
        pub trait Client: Sized {
            #[host(class(name = "Reply"))]
            type Reply;
            #[host(await)] async fn send(&self, text: String) -> Res<Self::Reply>;
        }
        "#,
    )
    .expect("resolves");
    let out = crate::adapter::emit(&decls).expect("emits");
    let root = &out.iter().find(|o| o.path == "mod.rs").expect("mod.rs").text;
    let llm = &out.iter().find(|o| o.path == "llm.rs").expect("llm.rs").text;
    for (text, want) in [
        (root, "type Client: ::lk_agents_abi::llm::class::Client<Reply = Self::Reply>;"),
        (llm, "super::llm::Reply(::std::sync::Arc::new(out))"),
        (llm, "fn __aiter__(slf: Py<Self>) -> Py<Self> {"),
        (llm, "None => Err(::pyo3::exceptions::PyStopAsyncIteration::new_err(())),"),
    ] {
        assert!(text.contains(want), "{want}\n{text}");
    }
    let stub = &crate::class_stub::emit(&decls).expect("stubs")[0].out.text;
    assert!(stub.contains("async def __anext__(self) -> dict[str, object]"), "{stub}");
    let err = read(
        r#"use super::chat::{Chunk, Res};
        #[crate::host(class(name = "R"))]
        pub trait R: Sized { #[host(next)] async fn next(&self) -> Res<Chunk>; }"#,
    )
    .expect_err("an iteration with no end is refused");
    assert!(
        err.contains("R::next: next is `async fn next(&self) -> Result<Option<T>, E>`"),
        "{err}"
    );
}

// Catches: the `str | Message` client annotation being emitted without the generated record's
// import, or importing another subject's identically named record.
#[test]
fn handwritten_scalar_unions_import_the_records_in_their_declaring_module() {
    let decls = read(
        r#"
        pub mod other {
            #[crate::host(record(module = "other", extract, emit))]
            pub struct Message { pub text: String }
        }
        #[crate::host(record(module = "client", extract, emit))]
        pub struct Message { pub text: String }
        #[crate::host(handwritten(extract, host = "str | Message"))]
        pub enum Input { Text(String), Message(Message) }
        #[crate::host(class(name = "Client"))]
        pub trait Client: Sized {
            #[host(call)] fn send(&self, message: Input);
        }
        "#,
    )
    .expect("resolves");
    let out = crate::class_stub::emit(&decls).expect("emits");
    let stub = &out[0].out.text;
    assert!(stub.contains("from .client import Message"), "{stub}");
    assert!(stub.contains("message: str | Message"), "{stub}");
    assert!(!stub.contains("from .other import Message"), "{stub}");
}

// Catches: a generated host return refusing a mixed tuple because its JSON map cannot use
// pyo3's native extraction, or accepting a tuple without checking its exact arity.
#[test]
fn proxy_tuple_returns_convert_each_element_by_its_declared_kind() {
    let decls = read(&proxy(
        r#"
        #[host(await)] async fn report(&self) -> Result<(Chunk, super::chat::Extra), HostError>;
        "#,
    ))
    .expect("resolves");
    let out = crate::adapter::emit(&decls).expect("emits");
    let llm = &out.iter().find(|o| o.path == "llm.rs").expect("llm.rs").text;
    assert!(llm.contains("cast::<::pyo3::types::PyTuple>()?"), "{llm}");
    assert!(llm.contains("tuple.len() != 2"), "{llm}");
    assert!(llm.contains("tuple.get_item(0)?"), "{llm}");
    assert!(llm.contains("tuple.get_item(1)?"), "{llm}");
    assert!(llm.contains("extract::<::lk_agents_abi::llm::chat::Chunk>()"), "{llm}");
    assert!(llm.contains("Value::Object(map) => Ok(map)"), "{llm}");
}

// Catches: an awaited call of a trait without async-trait (`-> BoxFuture<'_, T>`) refused or
// implemented as an `async fn` (which would not match the trait), a consuming one losing its
// `'static` future, a method's named future lifetime dropped, a missing context returned as a bare
// `Err` where the future is due, or the two forms mixed in one trait.
#[test]
fn a_proxy_without_async_trait_returns_boxed_futures() {
    let decls = read(
        r#"use super::chat::Chunk;
        use crate::errors::HostError;
        #[crate::host(proxy(host = "livekit.agents.Callbacks"))]
        pub trait Callbacks: Send + Sync {
            #[host(await)] fn next(&self, id: &str) -> crate::BoxFuture<'_, Result<Chunk, HostError>>;
            #[host(await)] fn close(self: Box<Self>) -> crate::BoxFuture<'static, ()>;
        }"#,
    )
    .expect("resolves");
    let out = crate::adapter::emit(&decls).expect("emits");
    let llm = &out.iter().find(|o| o.path == "llm.rs").expect("llm.rs").text;
    assert!(!llm.contains("async_trait") && !llm.contains("async fn"), "{llm}");
    assert!(
        llm.contains("fn next(&self, id: &str) -> ::lk_agents_abi::BoxFuture<'_, ::std::result::Result<::lk_agents_abi::llm::chat::Chunk, ::lk_agents_abi::errors::HostError>> {")
            && llm.contains("fn close(self: ::std::boxed::Box<Self>) -> ::lk_agents_abi::BoxFuture<'static, ()> {")
            && llm.contains("::std::boxed::Box::pin(async move {"),
        "{llm}"
    );
    // TurnHost's form: the future named by the method's lifetime, the call's own context.
    let named = read(
        r#"use super::chat::{Chunk, Report};
        use crate::errors::HostError;
        #[crate::host(proxy(host = "livekit.agents.Turns"))]
        pub trait Turns: Send + Sync {
            #[host(await)] fn handle<'a>(&'a self, #[host(context)] at: &'a Report) -> crate::BoxFuture<'a, Result<Chunk, HostError>>;
        }"#,
    )
    .expect("resolves");
    let out = crate::adapter::emit(&named).expect("emits");
    let llm = &out.iter().find(|o| o.path == "llm.rs").expect("llm.rs").text;
    assert!(
        llm.contains("fn handle<'a>(&'a self, at: &::lk_agents_abi::llm::chat::Report) -> ::lk_agents_abi::BoxFuture<'a, ")
            && llm.contains("let started = ::lk_agents_runtime::host::call(&crate::host::Entry, \"Turns.handle\", at, &[], "),
        "{llm}"
    );
    let mixed = read(
        r#"use crate::errors::HostError;
        #[crate::host(proxy(host = "livekit.agents.Callbacks"))]
        #[async_trait]
        pub trait Callbacks: Send + Sync {
            #[host(await)] fn next(&self) -> crate::BoxFuture<'_, Result<(), HostError>>;
        }"#,
    )
    .expect_err("a boxed call under async-trait is refused");
    assert!(mixed.contains("BoxFuture"), "{mixed}");
}

// Catches: a host lookup answering an optional proxy refused at generation (a room's participant
// by identity), or its absence read as a refused object instead of `None`.
#[test]
fn an_optional_proxy_is_absence_or_the_host_object_behind_its_proxy() {
    let decls = read(&proxy(
        r#"
        #[host(call)] fn lookup(&self, id: &str) -> Result<Option<std::sync::Arc<dyn Callbacks>>, HostError>;
        "#,
    ))
    .expect("resolves");
    let out = crate::adapter::emit(&decls).expect("emits");
    let llm = &out.iter().find(|o| o.path == "llm.rs").expect("llm.rs").text;
    assert!(
        llm.contains("if o.is_none() { Ok(None) } else { (crate::generated::llm::CallbacksProxy::from_host(&o)).map(Some) }"),
        "{llm}"
    );
}

// Catches: a live room read answered from a stale snapshot (or awaited off its caller), a host
// failure of a synchronous call turned into a default, a writer close that cannot consume its
// proxy, a host object the core must own (`Box<dyn ..>`) refused, a failed close that returns
// nothing dropped silently, or a borrowed participant read
// on every call instead of once when the room is taken.
#[test]
fn proxy_calls_consumes_and_lends_by_declaration() {
    let decls = read(&proxy(
        r#"
        #[host(call)] fn state(&self, id: &str) -> Result<Chunk, HostError>;
        #[host(await)] async fn close(self: Box<Self>) -> Result<(), HostError>;
        #[host(read_once)] fn conn(&self) -> &Conn;
        #[host(await)] async fn open_writer(&self) -> Result<Box<dyn Callbacks>, HostError>;
        #[host(await)] async fn release(&self);
        "#,
    ))
    .expect("resolves");
    let out = crate::adapter::emit(&decls).expect("emits");
    let llm = &out.iter().find(|o| o.path == "llm.rs").expect("llm.rs").text;
    assert!(
        llm.contains("fn state(&self, id: &str) -> ::std::result::Result<::lk_agents_abi::llm::chat::Chunk, ::lk_agents_abi::errors::HostError>")
            && llm.contains("let out = ::lk_agents_runtime::host::call_now(&crate::host::Entry, ")
            && llm.contains("getattr(\"state\")")
            && llm.contains("return Err(<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e));"),
        "{llm}"
    );
    assert!(llm.contains("async fn close(self: ::std::boxed::Box<Self>)"), "{llm}");
    // An awaited close returning nothing still reports its failure.
    assert!(
        llm.contains("async fn release(&self) {")
            && llm.contains(
                "::lk_agents_abi::python::log::warn!(\"Callbacks.release failed: {e}\");"
            ),
        "{llm}"
    );
    // A host object handed over to be owned (a writer the core later closes by value).
    assert!(
        llm.contains("-> ::std::result::Result<::std::boxed::Box<dyn ::lk_agents_abi::llm::class::Callbacks>")
            && llm.contains("crate::generated::llm::CallbacksProxy::owned(&")
            && llm.contains("pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::llm::class::Callbacks>>"),
        "{llm}"
    );
    assert!(
        llm.contains(
            "fn conn(&self) -> &::lk_agents_abi::llm::chat::Conn {\n        &self.conn\n    }"
        ) && llm.contains("crate::host::read(obj, \"conn\")"),
        "{llm}"
    );
    let err = refuse_proxy("#[host(call)] fn state(&self) -> Chunk;");
    assert!(err.contains("Result<T, E>"), "{err}");
}

// Catches: a plain value of an ABI type implementing a declared class (a request the core hands a
// host provider) refused, or handed over as anything but a `_core` handle over its own clone.
#[test]
fn a_value_implementing_a_declared_class_crosses_as_a_handle_over_a_clone() {
    let decls = read(
        "#[derive(Clone)]\npub struct Request;\n#[crate::host(class(name = \"Request\"))]\npub trait RequestView: Sized {}\nimpl RequestView for Request {}\n\
         #[crate::host(proxy(host = \"livekit.agents.Model\"))]\n\
         pub trait Model: Send + Sync {\n\
             #[host(notify, name = \"_core_send\")] fn send(&self, req: &Request);\n\
         }",
    )
    .expect("resolves");
    let out = crate::adapter::emit(&decls).expect("emits");
    let llm = &out.iter().find(|o| o.path == "llm.rs").expect("llm.rs").text;
    let want = "::pyo3::Bound::new(py, crate::generated::llm::Request(::std::sync::Arc::new(req_arg)))?";
    assert!(llm.contains(want), "{want}\n{llm}");
}

// Catches: a host class without the `_core_*` surface (an io leg) called directly instead of
// through its declared view, or a shared `_core` object read back as a copy rather than the very
// object the host holds.
#[test]
fn a_viewed_proxy_calls_its_view_and_shares_core_objects() {
    let decls = read(
        "pub struct Tracker;\n#[crate::host(class(name = \"Tracker\"))]\npub trait Tracked: Sized {}\nimpl Tracked for Tracker {}\n\
         #[crate::host(proxy(host = \"livekit.agents.io.Sink\", view = \"livekit.agents.view._Sink\"))]\n\
         pub trait Sink: Send + Sync {\n\
             #[host(read_once, name = \"_core_tracker\")] fn tracker(&self) -> &std::sync::Arc<Tracker>;\n\
             #[host(notify, name = \"_core_flush\")] fn flush(&self);\n\
         }",
    )
    .expect("resolves");
    let out = crate::adapter::emit(&decls).expect("emits");
    let llm = &out.iter().find(|o| o.path == "llm.rs").expect("llm.rs").text;
    for want in [
        "let view = crate::host::construct(&obj.py().import(\"livekit.agents.view\")?.getattr(\"_Sink\")?, (obj,), None)?;\n        let obj = &view;",
        "v.cast::<crate::generated::llm::Tracker>().map(|this| ::std::sync::Arc::clone(&this.get().0))",
        "fn tracker(&self) -> &::std::sync::Arc<::lk_agents_abi::llm::class::Tracker> {",
    ] {
        assert!(llm.contains(want), "{want}\n{llm}");
    }
}

#[test]
fn bound_provider_proxies_use_the_declared_native_wrapper_path() {
    for (wrapper, expected) in [
        ("crate::llm::class::Retrying", "::lk_agents_abi::llm::class::Retrying"),
        ("lk_agents_runtime::stt::RetryingStt", "::lk_agents_runtime::stt::RetryingStt"),
        ("RuntimeRetry", "::lk_agents_runtime::stt::RetryingStt"),
    ] {
        let source = format!(
            "pub struct Retrying; use lk_agents_runtime::stt::RetryingStt as RuntimeRetry; {}",
            proxy(&format!(
                "#[host(bind(wrap = {wrapper}))] fn with_conn(&self, conn: &Report) -> Option<std::sync::Arc<dyn Callbacks>> {{ None }}"
            ))
        );
        let decls = read(&source).expect("native wrapper resolves");
        assert_eq!(decls.proxies[0].methods[0].wrap.as_deref(), Some(expected));
        let emitted = crate::adapter::emit(&decls).expect("wrapper emits");
        let adapter = &emitted.iter().find(|o| o.path == "llm.rs").unwrap().text;
        assert!(adapter.contains(&format!("Some({expected}::bound(")), "{adapter}");
    }
    for invalid in ["lk_agents_runtime::stt::RetryingStt<Report>", "&Report"] {
        let source = proxy(&format!(
            "#[host(bind(wrap = {invalid}))] fn with_conn(&self, conn: &Report) -> Option<std::sync::Arc<dyn Callbacks>> {{ None }}"
        ));
        let error = read(&source).expect_err("invalid wrapper shape is refused");
        assert!(error.contains("Callbacks::with_conn"), "{error}");
    }
}

#[test]
fn declared_classes_consume_string_enums_with_native_error_expressions() {
    let source = format!(
        "{} {}",
        r#"
        #[crate::host(string)]
        #[crate::host(wire_strs(err = UnknownKind, msg = UnknownKind(s.to_owned())))]
        pub enum Kind { Batch, Stream }
        "#,
        class("#[host(new)] fn new(kind: Kind) -> Self;")
    );
    let decls = read(&source).expect("string enum with its native diagnostic resolves");
    let out = crate::adapter::emit(&decls).expect("emits");
    let llm = &out.iter().find(|o| o.path == "llm.rs").unwrap().text;
    assert!(llm.contains("fn new(py: Python<'_>, kind: ::lk_agents_abi::llm::class::Kind)"), "{llm}");
    let stubs = crate::class_stub::emit(&decls).expect("stubs");
    assert!(stubs[0].out.text.contains("kind: str"), "{}", stubs[0].out.text);
}

// Catches: a declared proxy refused as a host call's argument (a typed noise-cancellation choice
// passed back to the SDK), handed back as the view over it rather than the object the host
// passed, a trait no proxy can be found behind accepted, or a host stream a synchronous call
// returns pulled without a context.
#[test]
fn a_declared_proxy_is_handed_back_as_its_host_object() {
    let decls = read(
        r#"use std::sync::Arc;
        use std::any::Any;
        use crate::errors::HostError;
        #[crate::host(proxy(host = "livekit.agents.Noise", view = "livekit.agents._NoiseView"))]
        pub trait Noise: Any + Send + Sync {
            #[host(read_once, name = "_core_kind")] fn kind(&self) -> String;
        }
        #[crate::host(proxy(host = "livekit.agents.Track"))]
        pub trait Track: Send + Sync {
            #[host(call)] fn open(&self, noise: Option<Arc<dyn Noise>>) -> Result<(), HostError>;
            #[host(call)] fn frames(&self) -> Result<Box<dyn crate::utils::stream::StreamSource<String>>, HostError>;
        }"#,
    )
    .expect("resolves");
    let out = crate::adapter::emit(&decls).expect("emits");
    let llm = &out.iter().find(|o| o.path == "llm.rs").expect("llm.rs").text;
    assert!(
        llm.contains(
            "let source = ::std::sync::Arc::new(obj.clone().unbind());\n        let view ="
        ) && llm.contains("Ok(proxy) => Ok(proxy.source.bind(py).clone()),")
            && llm.contains(
                "Some(x) => super::llm::NoiseProxy::to_host(py, ::std::sync::Arc::clone(&x))?"
            )
            && llm.contains("let context = ::lk_agents_abi::agent::HostContext::host_context(&self.context).cloned();"),
        "{llm}"
    );
    let not_any = read(
        r#"use std::sync::Arc;
        use crate::errors::HostError;
        #[crate::host(proxy(host = "livekit.agents.Noise"))]
        pub trait Noise: Send + Sync {
            #[host(read_once, name = "_core_kind")] fn kind(&self) -> String;
        }
        #[crate::host(proxy(host = "livekit.agents.Track"))]
        pub trait Track: Send + Sync {
            #[host(call)] fn open(&self, noise: Arc<dyn Noise>) -> Result<(), HostError>;
        }"#,
    );
    assert!(not_any.err().is_some_and(|e| e.contains("Noise: Any + Send + Sync")));
}
