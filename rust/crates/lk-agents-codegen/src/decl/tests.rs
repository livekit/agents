//! `resolved_declaration_ir`: declarations in, IR or a refusal naming the declaration out.

use super::tree::Crate;
use super::*;

const LIB: &str = "pub mod errors; pub mod llm; pub use llm::chat::Item;";

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

// Catches: a type accepted for what its last segment spells. `Value` here is the crate's own
// undeclared type, not `serde_json::Value`, and `Duration` has no declared unit.
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
    let err = refuse("#[host(call)] fn f(&self, d: std::time::Duration);");
    assert!(err.contains("std::time::Duration") && err.contains("no declared conversion"), "{err}");
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
        ("type T;", "declares its kind"),
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
    let stub = &crate::class_stub::emit(&decls).expect("stubs")[0];
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
    let stub = &crate::class_stub::emit(&decls).expect("stubs")[0].text;
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
// a failure (or the reverse), a notification awaited or its failure dropped silently, an argument
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
    assert_eq!((p.rust_name.as_str(), p.host.as_str(), p.family.as_str()), ("Callbacks", "livekit.agents.Callbacks", "llm"));
    assert!(p.async_trait);
    let kinds: Vec<_> = p.methods.iter().map(|m| (m.kind, m.name.as_str(), m.context)).collect();
    assert_eq!(kinds, [(ProxyCall::Await, "render", 0), (ProxyCall::Await, "prepare", 0), (ProxyCall::Notify, "abort", 0)]);
    let refusal = p.methods[1].ret.refusal.as_ref().expect("declared");
    assert_eq!((refusal.module.as_str(), refusal.class.as_str()), ("livekit.agents", "Refused"));
    assert_eq!(p.methods[1].ret.err.as_deref(), Some("::lk_agents_abi::errors::HostError"));
    let out = crate::adapter::emit(&decls).expect("emits");
    let llm = &out.iter().find(|o| o.path == "llm.rs").expect("llm.rs").text;
    for want in [
        "impl ::lk_agents_abi::llm::class::Callbacks for CallbacksProxy {",
        "#[::async_trait::async_trait]",
        "let context = ::lk_agents_abi::agent::HostContext::host_context(at);",
        "Some(context) => ::lk_agents_runtime::host::python::call_in(context, move |py| {",
        "Ok((host.bind(py).getattr(\"render\")?",
        "::lk_agents_abi::python::extra_to_py(py, &args_arg)?.into_any()",
        "if e.is_instance(py, &py.import(\"livekit.agents\").and_then(|m| m.getattr(\"Refused\"))",
        "return Ok(Err(<::lk_agents_abi::errors::Refused as ::std::convert::From<String>>::from(e.value(py).to_string())));",
        "<::lk_agents_abi::errors::HostError as ::lk_agents_abi::python::FromHostErr>::from_host(py, &e)",
        "fn cancel(&self, at: &::lk_agents_abi::llm::chat::Report) {",
        "runtime.spawn(async move {",
        "Ok((host.bind(py).getattr(\"abort\")?",
        "::lk_agents_abi::python::log::warn!(\"Callbacks.abort failed: {e}\");",
        "let host = super::llm::CallbacksProxy::from_host(&host)?;",
        "native::<dyn ::lk_agents_abi::llm::class::Callbacks>(obj)",
    ] {
        assert!(llm.contains(want), "{want}\n{llm}");
    }
}

// Catches: a proxy call with no context to enter (or two), a failure swallowed into a default, an
// error or refusal with no declared conversion, a call kind guessed from the signature, or a
// value handed to the host that it cannot receive.
#[test]
fn incomplete_proxy_declarations_fail_generation() {
    for (body, want) in [
        ("#[host(await)] async fn f(&self, at: &Report) -> Result<String, HostError>;", "one #[host(context)] parameter"),
        (
            "#[host(await)] async fn f(&self, #[host(context)] a: &Report, #[host(context)] b: &Report) -> Result<String, HostError>;",
            "one parameter is the call's #[host(context)]",
        ),
        ("#[host(await)] async fn f(&self, #[host(context)] a: &Report) -> String;", "a host failure is reported"),
        ("#[host(await)] async fn f(&self, #[host(context)] a: &Report) -> Result<String, std::fmt::Error>;", "not an ABI declaration"),
        ("#[host(await)] async fn f(&self, #[host(context)] a: &Report) -> Result<Result<String, HostError>, HostError>;", "error(raised"),
        ("#[host(await)] fn f(&self, #[host(context)] a: &Report) -> Result<String, HostError>;", "does not fit `await`"),
        ("#[host(notify)] fn f(&self, #[host(context)] a: &Report) -> u32;", "does not fit `notify`"),
        ("#[host(await, notify)] fn f(&self, #[host(context)] a: &Report);", "exactly one of await, notify"),
        (
            "#[host(await)] async fn f(&self, #[host(context)] a: &Report) -> Result<String, HostError>; fn g(&self);",
            "Callbacks::g: a proxy method declares exactly one of",
        ),
        (
            "#[host(await)] async fn f(&self, #[host(context)] a: &Report, c: Conn) -> Result<(), HostError>;",
            "f(c): record `::lk_agents_abi::llm::chat::Conn` is not declared for output",
        ),
        ("#[host(await)] async fn f(&self, #[host(context)] a: &Report) -> Result<Report, HostError>;", "not declared for input"),
    ] {
        let err = refuse_proxy(body);
        assert!(err.contains("Callbacks") && err.contains(want), "`{body}`: {err}");
    }
}
