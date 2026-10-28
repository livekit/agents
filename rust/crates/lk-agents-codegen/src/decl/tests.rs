//! `resolved_declaration_ir`: declarations in, IR or a refusal naming the declaration out.

use super::tree::Crate;
use super::*;

const LIB: &str = "pub mod errors; pub mod llm; pub mod utils; pub use llm::chat::Item;";

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

fn read(llm: &str) -> Result<Declarations, String> {
    let krate = Crate::from_files(
        "lk_agents_abi",
        &[
            ("lib.rs", LIB),
            ("errors.rs", "mod api;\npub use api::*;"),
            ("errors/api.rs", ERRORS),
            ("llm.rs", "pub mod chat; pub mod class;"),
            ("llm/chat.rs", CHAT),
            ("llm/class.rs", llm),
            ("utils.rs", "pub mod stream;"),
            ("utils/stream.rs", "pub trait StreamSource<T> {}"),
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
        #[host(call)] fn chat(&self, items: &[Item], #[host(kw)] conn: Conn, #[host(default = None)] reporter: Option<Arc<dyn Llm>>) -> Res<Self::Chunks>;
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
    assert!(llm.contains("HostProxy>::from_host(&v, None)"), "{llm}");
    assert!(llm.contains("(&self.0, &items, conn, reporter)"), "{llm}");
    assert!(llm.contains("NativeStream(::lk_agents_runtime::source::NativeSource::new(::std::boxed::Box::new(out)))"), "{llm}");
    assert!(
        llm.contains("#[pyo3(signature = (timeout = 1.5))]")
            && llm.contains("future_into_py(py, async move"),
        "{llm}"
    );
    assert!(llm.contains("TypeId::of::<dyn ::lk_agents_abi::llm::chat::Llm>()"), "{llm}");
    let stub = &crate::class_stub::emit(&decls).expect("stubs")[0].out;
    assert_eq!(stub.path, "_core_llm.pyi");
    for want in [
        "def __init__(self, model: str, *, extra: dict[str, object] | None = None) -> None: ...",
        "    def chat(\n        self,\n        items: list[Item],\n        *,\n        conn: APIConnectOptions,\n        reporter: LLM | None = None,\n    ) -> NativeStream: ...",
        "def report(self) -> dict[str, object]: ...",
        "async def ping(self, timeout: float = 1.5) -> None: ...",
        "async def recv(self) -> dict[str, object] | None: ...",
        "from livekit.agents.llm import LLM",
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
            && llm.contains("pub struct Parser(ParserImpl);"),
        "{llm}"
    );
    assert!(
        llm.contains("fn parse(&mut self, id: ::std::string::String)")
            && llm.contains("(&mut self.0, &id)"),
        "{llm}"
    );
    assert!(llm.contains("#[pyfunction]\n#[pyo3(signature = (chunk))]\npub fn llm_usage_chunk(chunk: Bound<'_, PyAny>)"), "{llm}");
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

fn proxy(body: &str) -> String {
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
            #[host(await)] async fn prepare(&self, #[host(context)] at: &Report, args: serde_json::Map<String, serde_json::Value>) -> Result<Result<Chunk, Refused>, HostError>;
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
    let refusal = p.methods[1].ret.refusal.as_ref().expect("declared");
    assert_eq!((refusal.module.as_str(), refusal.class.as_str()), ("livekit.agents", "Refused"));
    assert_eq!(p.methods[1].ret.err.as_deref(), Some("::lk_agents_abi::errors::HostError"));
    let out = crate::adapter::emit(&decls).expect("emits");
    let llm = &out.iter().find(|o| o.path == "llm.rs").expect("llm.rs").text;
    for want in [
        "impl ::lk_agents_abi::llm::class::Callbacks for CallbacksProxy {",
        "#[::async_trait::async_trait]",
        "let Some(context) = ::lk_agents_abi::agent::HostContext::host_context(at).cloned() else {",
        "let out = ::lk_agents_runtime::host::python::call_in(context, move |py| {",
        "Ok((host.bind(py).getattr(\"render\")?",
        "::lk_agents_abi::python::extra_to_py(py, &args_arg)?.into_any()",
        "if e.is_instance(py, &py.import(\"livekit.agents\").and_then(|m| m.getattr(\"Refused\"))",
        "return Ok(Err(<::lk_agents_abi::errors::Refused as ::std::convert::From<String>>::from(e.value(py).to_string())));",
        "<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e)",
        "fn cancel(&self, at: &::lk_agents_abi::llm::chat::Report) {",
        "self.notifier.notify(context, \"Callbacks.abort\", move |py| {",
        "Ok((host.bind(py).getattr(\"abort\")?",
        "let host = super::llm::CallbacksProxy::from_host(&host)?;",
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
        "let v = obj.call_method0(\"_core_label\")?;",
        "context: ::lk_agents_runtime::host::python::capture_context(obj.py())?",
        "fn label(&self) -> ::std::string::String {\n        ::std::clone::Clone::clone(&self.label)",
        "let context = ::std::clone::Clone::clone(&self.context);",
        "Some(::std::sync::Arc::new(Self { conn: Some(::std::clone::Clone::clone(conn)), ..::std::clone::Clone::clone(self) }))",
        "if let Some(bound) = bound {\n                    args.push(",
        "::lk_agents_runtime::host::stream::HostStream::open(py, o, context, |o| Ok(o.extract::<::lk_agents_abi::llm::chat::Chunk>()?)",
        "DeclaredProxy::of::<dyn ::lk_agents_abi::llm::class::Callbacks>",
    ] {
        assert!(llm.contains(want), "{want}\n{llm}");
    }
    let bridges = crate::records::rust(&decls).expect("emits");
    let bridge = &bridges.iter().find(|o| o.path == "llm.rs").expect("llm.rs").text;
    assert!(
        bridge.contains(
            "impl crate::python::HostProxy for dyn ::lk_agents_abi::llm::class::Callbacks {"
        ) && bridge.contains(
            "crate::python::declared_proxy::<dyn ::lk_agents_abi::llm::class::Callbacks>(obj)"
        ),
        "{bridge}"
    );
}

// Catches: a proxy call with no context to enter (or two), a failure swallowed into a default, an
// error or refusal with no declared conversion, a call kind guessed from the signature, or a
// value handed to the host that it cannot receive.
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
        #[crate::host(record(module = "session", extract))]
        pub struct Registration { pub llm: std::sync::Arc<dyn Llm> }
        "#,
    )
    .expect("resolves");
    let rust = crate::records::rust(&decls).expect("emits");
    let llm = &rust.iter().find(|o| o.path == "llm.rs").expect("llm.rs").text;
    for want in [
        "cast::<::pyo3::types::PyDict>()?.iter().map(|(key, item)|",
        "collect::<PyResult<::std::collections::BTreeSet<_>>>()",
        "::pyo3::types::PyFrozenSet::new(py,",
        "<dyn ::lk_agents_abi::llm::chat::Llm as crate::python::HostProxy>::from_host(",
    ] {
        assert!(llm.contains(want), "{want}\n{llm}");
    }
    let models = crate::records::python(&decls).expect("models");
    let session = &models.iter().find(|o| o.path == "session.py").expect("session.py").text;
    assert!(
        session.contains("frames: dict[str, int]")
            && session.contains("started: frozenset[str]")
            && session.contains("llm: LLM"),
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
        llm.contains("pub fn _core_persist_session_state(record: ::lk_agents_abi::llm::class::SessionState) -> PyResult<String> {\n    ::serde_json::to_string(&record)")
            && llm.contains("pub fn _core_restore_session_state(data: &str) -> PyResult<::lk_agents_abi::llm::class::SessionState> {\n    ::serde_json::from_str(data)")
            && root.contains("wrap_pyfunction!(llm::_core_restore_session_state, m)"),
        "{llm}\n{root}"
    );
    let models = crate::records::python(&decls).expect("models");
    let model = &models.iter().find(|o| o.path == "session.py").expect("session.py").text;
    assert!(
        model.contains("return _core._core_persist_session_state(self)")
            && model.contains("def from_json(cls, data: str) -> SessionState:"),
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
            && llm.contains("self.host.bind(py).call_method1(\"state\"")
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
            && llm.contains("super::llm::CallbacksProxy::owned(&")
            && llm.contains("pub fn owned(obj: &Bound<'_, PyAny>) -> PyResult<::std::boxed::Box<dyn ::lk_agents_abi::llm::class::Callbacks>>"),
        "{llm}"
    );
    assert!(
        llm.contains(
            "fn conn(&self) -> &::lk_agents_abi::llm::chat::Conn {\n        &self.conn\n    }"
        ) && llm.contains("obj.call_method0(\"conn\")"),
        "{llm}"
    );
    let err = refuse_proxy("#[host(call)] fn state(&self) -> Chunk;");
    assert!(err.contains("Result<T, E>"), "{err}");
}
