//! The napi emitter's tests: one source in, the generated Node binding out.
//!
//! Same rule as `tests.rs`: every test names the bug it catches, and
//! asserts the property that would have shipped broken rather than the whole output. The core
//! module is `lk_agents_abi`, because an item from a crate `lk-agents-node` does not link is a gap.

use crate::{napi, node, parse};

fn emit(src: &str) -> Vec<crate::python::Out> {
    let decls = parse::file("test.rs", "lk_agents_abi", src).expect("the declarations parse");
    let x = parse::exposed("test.rs", "lk_agents_abi", src).expect("the items parse");
    napi::emit(&decls, &x).expect("the binding generates")
}

/// Every generated file, concatenated.
fn generate(src: &str) -> String {
    emit(src).into_iter().map(|o| o.text).collect::<Vec<_>>().join("\n")
}

/// `mod.rs`: the gap list.
fn gaps(src: &str) -> String {
    emit(src).into_iter().find(|o| o.path == "mod.rs").expect("always written").text
}

// --- functions -----------------------------------------------------------------------------------

// Catches: a raw-identifier parameter (`r#ref`) written into the binding as `r#ref`. napi-derive
// names the TS parameter from the pattern's text, so `index.d.ts` said `r#ref: string` and
// `napi build` failed on it (2026-09-24).
#[test]
fn a_keyword_parameter_is_a_plain_ts_name() {
    let out = generate("#[host(export)]\npub fn resolve_ref(root: &str, r#ref: &str) -> bool { true }");
    assert!(out.contains("pub fn resolve_ref(root: String, ref_: String) -> bool {\n    lk_agents_abi::resolve_ref(&root, &ref_)\n}"), "{out}");
    assert!(!out.contains("r#ref"), "{out}");
}

// Catches: a Node name in snake_case (napi's convention and agents-js's are camelCase), or a
// borrowed `&str` taken as something the core cannot borrow.
#[test]
fn an_export_is_a_camel_case_napi_function_that_lends_its_strings() {
    let out = generate("#[host(export(name = \"is_cloud\"))]\npub fn is_cloud_url(url: &str) -> bool { true }");
    assert!(out.contains("#[napi(js_name = \"isCloud\")]\npub fn is_cloud(url: String) -> bool {\n    lk_agents_abi::is_cloud_url(&url)\n}"), "{out}");
}

// Catches: a declared default lost in Node -- an omitted argument must take the declaration's
// value, as pyo3's signature gives it, not the type's `Default`.
#[test]
fn an_omitted_argument_takes_the_declared_default() {
    let out = generate("#[host(export(defaults(n = 20, flag = true)))]\npub fn f(n: usize, flag: bool) {}");
    assert!(out.contains("pub fn f(env: Env, n: Option<i64>, flag: Option<bool>) -> napi::Result<()>"), "{out}");
    assert!(out.contains("let n = match n { Some(v) => <usize>::try_from(v)"), "{out}");
    assert!(out.contains("None => 20 };"), "{out}");
    assert!(out.contains("let flag = flag.unwrap_or(true);"), "{out}");
}

// Catches: a wide integer wrapping silently in either direction. pyo3 raises `OverflowError` for a
// value that does not fit; a count past `i64::MAX` saturates rather than turning negative.
#[test]
fn a_wide_integer_is_range_checked_in_and_saturated_out() {
    let out = generate("#[host(export)]\npub fn f(n: u64) -> usize { 0 }");
    assert!(out.contains("<u64>::try_from(n).map_err(|e| crate::host_error::JsError::new(\"OverflowError\", format!(\"n: {e}\")))?"), "{out}");
    assert!(out.contains("i64::try_from(lk_agents_abi::f(n)).unwrap_or(i64::MAX)"), "{out}");
}

// Catches: `Maybe<T>` required in Node, or an omitted argument read as a value -- it is `NotGiven`,
// as pyo3's `Maybe` folds `None`.
#[test]
fn an_omitted_maybe_is_not_given() {
    let out = generate("#[host(export)]\npub fn f(a: Maybe<f64>) {}");
    assert!(out.contains("pub fn f(a: Option<f64>)"), "{out}");
    assert!(out.contains("let a = match a { Some(v) => lk_agents_abi::Maybe::Given(v), None => lk_agents_abi::Maybe::NotGiven };"), "{out}");
}

// Catches: a wire enum crossing as anything but its string, or an unknown string accepted -- pyo3
// raises `ValueError` from `parse_str`, and TypeScript names the closed set.
#[test]
fn a_wire_enum_parses_in_and_names_its_strings_in_typescript() {
    let src = "#[host(wire_enum)]\npub enum Mode { #[host(rename = \"on-air\")] OnAir, Off }\n\
               #[host(export)]\npub fn f(m: Mode) -> Mode { m }";
    let out = generate(src);
    assert!(out.contains("#[napi(js_name = \"f\", ts_return_type = \"'on-air' | 'off'\")]"), "{out}");
    assert!(out.contains("#[napi(ts_arg_type = \"'on-air' | 'off'\")] m: String"), "{out}");
    assert!(out.contains("m.parse().map_err(|e| crate::host_error::JsError::new(\"ValueError\""), "{out}");
    assert!(out.contains(".as_str().to_owned()"), "{out}");
    // `wire_strs` (and a wire_enum's `err`/`msg`) are Rust-only: an enum under it binds as nothing.
    let src = "#[host(wire_strs(err = E, msg = E(s.to_owned())))]\npub enum Mode { OnAir, Off }\n\
               #[host(export)]\npub fn f(m: Mode) -> Mode { m }";
    let out = generate(src);
    assert!(out.contains("`Mode` is not a #[derive(Host)] type or a #[host(wire_enum)]"), "{out}");
}

// --- errors --------------------------------------------------------------------------------------

// Catches: a Node error whose kind disagrees with Python's for the same core failure -- the
// `#[host(error)]` table, an explicit `err = ".."`, and the `ValueError` default all carry over.
#[test]
fn errors_carry_the_python_kind_through_the_same_tables() {
    let src = "#[host(error(kind = \"RuntimeError\"))]\npub enum ParseError { #[host(kind = \"TypeError\")] WrongType(String), Other }\n\
               #[host(export)]\npub fn a() -> Result<u8, ParseError> { todo!() }\n\
               #[host(export(err = \"AssertionError\"))]\npub fn b() -> Result<u8, Other> { todo!() }\n\
               #[host(export)]\npub fn c() -> Result<u8, Unmapped> { todo!() }";
    let out = generate(src);
    assert!(out.contains("pub fn parse_error_to_node(e: lk_agents_abi::ParseError) -> JsError {"), "{out}");
    assert!(out.contains("lk_agents_abi::ParseError::WrongType(..) => \"TypeError\","), "{out}");
    assert!(out.contains("_ => \"RuntimeError\","), "{out}");
    assert!(out.contains("lk_agents_abi::a().map_err(crate::generated::errors::parse_error_to_node)"), "{out}");
    assert!(out.contains("JsError::new(\"AssertionError\", e.to_string())"), "{out}");
    assert!(out.contains("JsError::new(\"ValueError\", e.to_string())"), "{out}");
    assert!(out.contains("pub fn a(env: Env) -> napi::Result<u8> {\n    crate::host_error::run(&env, || {"), "{out}");
}

// Catches: an item Node cannot bind failing the whole run (and with it the pyo3 side of every
// declaration), or being dropped with no trace. A pyo3 converter is a gap, named in `mod.rs`.
#[test]
fn an_item_node_cannot_bind_is_a_listed_gap_not_a_failed_run() {
    let src = "#[host(export)]\npub fn b(o: std::rc::Rc<u8>) {}\n\
               #[host(export)]\npub fn ok() -> u8 { 0 }";
    let m = gaps(src);
    assert!(m.contains("- `lk_agents_abi::b`: parameter `o`:"), "{m}");
    assert!(m.contains("1 core items bound (1 functions and consts, 0 classes, 0 records, 0 proxies)"), "{m}");
    assert!(!generate(src).contains("pub fn b("), "a refused item must not be half-emitted");
}

// Catches: an item from a crate `lk-agents-node` does not link emitted anyway -- an unresolved path
// in a file nobody may edit.
#[test]
fn an_item_from_an_unlinked_crate_is_a_gap() {
    let decls = parse::file("t.rs", "lk_agents_abi", "").unwrap();
    let x = parse::exposed("t.rs", "lk_agents_rtc", "#[host(export)]\npub fn f() -> u8 { 0 }").unwrap();
    let out = napi::emit(&decls, &x).unwrap();
    let m = &out.iter().find(|o| o.path == "mod.rs").unwrap().text;
    assert!(m.contains("`lk_agents_rtc` is not linked by lk-agents-node"), "{m}");
    assert!(!out.iter().any(|o| o.path == "functions.rs"), "nothing to write");
}

// --- classes -------------------------------------------------------------------------------------

// Catches: a cell read without its lock, a `&mut self` method on a shared borrow, or a setter whose
// JS name keeps its `set_` (napi strips it; the property is `x`).
#[test]
fn a_class_holds_its_cell_and_each_method_reaches_it_the_cell_s_way() {
    let src = "#[host(object(name = \"Gate\", cell = \"mutex\"))]\nimpl Gate {\n\
               #[host(new)] pub fn new() -> Self { todo!() }\n\
               #[host(method)] pub fn fail(&mut self, reason: Option<String>) {}\n\
               #[host(getter)] pub fn disabled(&self) -> bool { false }\n}\n\
               #[host(object(name = \"Plain\"))]\nimpl Plain {\n\
               #[host(setter)] pub fn set_rate(&mut self, rate: u32) {}\n\
               #[host(method)] pub fn is_on(status_code: u16) -> bool { true }\n}";
    let out = generate(src);
    assert!(out.contains("#[napi(js_name = \"Gate\")]\npub struct Gate {\n    pub(crate) inner: std::sync::Mutex<lk_agents_abi::Gate>,\n}"), "{out}");
    assert!(out.contains("Gate { inner: std::sync::Mutex::new(lk_agents_abi::Gate::new()) }") || out.contains("std::sync::Mutex::new(out)"), "{out}");
    assert!(out.contains("pub fn fail(&self, reason: Option<String>) {\n        let mut inner = self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner);"), "{out}");
    assert!(out.contains("#[napi(getter, js_name = \"disabled\")]"), "{out}");
    assert!(out.contains("#[napi(setter, js_name = \"rate\")]\n    pub fn set_rate(&mut self, rate: u32) {\n        self.inner.set_rate(rate);"), "{out}");
    assert!(out.contains("#[napi(js_name = \"isOn\")]\n    pub fn is_on(status_code: u16) -> bool {"), "{out}");
}

// Catches: an exposed class argument converted instead of lent off its JS object -- the core takes
// a reference to the value the class holds, not a copy of it.
#[test]
fn an_exposed_class_argument_is_lent_off_its_js_object() {
    let src = "#[host(object(name = \"Opts\"))]\nimpl Opts {\n#[host(new)] pub fn new() -> Self { todo!() }\n}\n\
               #[host(export)]\npub fn enabled(options: &Opts) -> bool { true }";
    let out = generate(src);
    assert!(out.contains("pub fn enabled(options: &crate::generated::objects::Opts) -> bool {\n    lk_agents_abi::enabled(&options.inner)"), "{out}");
}

// Catches: an async method that blocks the JS thread, converts its result off it, or holds a JS
// borrow across the await. The future runs on napi's runtime and the result converts in the
// callback, where an `Env` exists for the error.
#[test]
fn an_async_method_runs_on_the_runtime_and_converts_back_on_the_js_thread() {
    let src = "#[host(object(name = \"C\", cell = \"arc\"))]\nimpl C {\n\
               #[host(method)] pub async fn fetch(&self, id: String) -> Result<u32, E> { todo!() }\n}";
    let out = generate(src);
    assert!(out.contains("#[napi(js_name = \"fetch\", ts_return_type = \"Promise<number>\")]"), "{out}");
    assert!(out.contains("pub fn fetch<'env>(&self, env: &'env Env, id: String) -> napi::Result<napi::bindgen_prelude::PromiseRaw<'env, u32>>"), "{out}");
    assert!(out.contains("let inner = std::sync::Arc::clone(&self.inner);"), "{out}");
    assert!(out.contains("env.spawn_future_with_callback(async move { Ok(inner.fetch(id).await.map_err("), "{out}");
    assert!(out.contains("Err(e) => Err(e.into_napi(env)),"), "{out}");
    let m = gaps("#[host(object(name = \"C\", cell = \"arc\"))]\nimpl C {\n#[host(method)] pub async fn f(&self, id: &str) -> u32 { 0 }\n}");
    assert!(m.contains("cannot hold across an await"), "{m}");
}

// Catches: a Python protocol method emitted under a mangled JS name (`__len__` -> `len`) that no
// Node reader would look for.
#[test]
fn a_python_protocol_method_is_a_gap() {
    let m = gaps("#[host(object(name = \"B\"))]\nimpl B {\n#[host(method)] pub fn __len__(&self) -> usize { 0 }\n}");
    assert!(m.contains("`__len__` is a Python protocol method"), "{m}");
}

// --- records -------------------------------------------------------------------------------------

const EVENT: &str = r#"
    #[derive(Host)]
    #[node(enum, module = "events", name = "UserState")]
    pub enum UserState { #[host(rename = "speaking")] Speaking, #[host(rename = "away")] Away }

    #[derive(Host)]
    #[node(interface, module = "events", name = "UserStateChangedEvent")]
    #[binding(emit, from = lk_agents_abi::voice::events::UserStateChanged)]
    pub struct UserStateChanged {
        pub new_state: UserState,
        #[python(default = "0.0")]
        pub start_time: f64,
        pub speaker_id: Option<String>,
        #[python(default_factory = "time.time")]
        #[binding(skip)]
        pub created_at: f64,
    }
"#;

// Catches: the napi CLI's `index.d.ts` and `node.rs`'s interface disagreeing about a field -- its
// name, its optionality, or its type. Both come from `node::field_ts`, so this pins that each napi
// field carries exactly that answer (with the enum spelled out, since `index.d.ts` has no alias).
#[test]
fn a_node_record_s_typescript_is_node_rs_s_field_for_field() {
    let decls = parse::file("test.rs", "lk_agents_abi", EVENT).unwrap();
    let out = generate(EVENT);
    let d = decls.iter().find(|d| d.rust_name == "UserStateChanged").unwrap();
    for f in d.fields() {
        let ts = node::field_ts(&decls, d, f, true).unwrap().unwrap();
        assert!(out.contains(&format!("ts_type = {:?}", ts.ty)), "{} -> {}\n{out}", f.name, ts.ty);
    }
    assert!(out.contains("#[napi(ts_type = \"'speaking' | 'away'\")]\n    pub new_state: String,"), "{out}");
    // Optional in TypeScript is `Option` in napi, whatever the core type.
    assert!(out.contains("pub start_time: Option<f64>,"), "{out}");
    let ts = crate::node::emit(&decls).unwrap().into_iter().map(|o| o.text).collect::<String>();
    assert!(ts.contains("  newState: UserState;\n  startTime?: number;\n  speakerId?: string | null;\n  createdAt?: number;\n"), "{ts}");
}

// Catches: a key JS may omit read as the type's default instead of the declaration's, and a key
// the host stamps (`#[binding(skip)]`, `created_at = time.time()`) sent as the core's placeholder
// -- pyo3 omits it so pydantic stamps it; Node stamps it the same way.
#[test]
fn record_defaults_are_the_declaration_s_both_ways() {
    let out = generate(EVENT.replace("pub new_state: UserState,\n", "").as_str());
    assert!(out.contains("start_time: match v.start_time { Some(v) => v, None => 0.0_f64 },"), "{out}");
    assert!(out.contains("created_at: Some(lk_agents_abi::epoch_secs(std::time::SystemTime::now())),"), "{out}");
    assert!(out.contains("impl From<UserStateChangedEvent> for lk_agents_abi::voice::events::UserStateChanged {"), "an infallible read is a `From`\n{out}");
}

// Catches: a declared unit enum read from JS through a `FromStr` the derive does not promise -- a
// compile error in a generated file. It is written out (`as_str`), and the record is not readable.
#[test]
fn a_declared_enum_field_is_written_and_not_read() {
    let out = generate(EVENT);
    assert!(out.contains("new_state: v.new_state.as_str().to_owned(),"), "{out}");
    assert!(!out.contains("for lk_agents_abi::voice::events::UserStateChanged {\n    type Error"), "{out}");
    assert!(!out.contains("impl From<UserStateChangedEvent> for"), "{out}");
}

// Catches: a union crossing as a tagged wrapper JS never sees (the member object is the value), or
// read by trying each arm -- napi objects validate only `typeof`, so the first arm would take all.
#[test]
fn a_union_is_its_member_object_and_is_read_by_its_discriminant() {
    let src = r#"
        #[derive(Host)]
        #[binding(emit, from = lk_agents_abi::k::A, tag = "type", tag_value = "a")]
        pub struct A { pub n: u32 }
        #[derive(Host)]
        #[binding(emit, from = lk_agents_abi::k::B, tag = "type", tag_value = "b")]
        pub struct B { pub s: String }
        #[derive(Host)]
        #[binding(emit, from = lk_agents_abi::k::U)]
        pub enum U { Ay(A), Be(B) }
        #[host(export)]
        pub fn f(u: U) -> U { u }
    "#;
    let out = generate(src);
    assert!(out.contains("pub enum U {\n    Ay(A),\n    Be(B),\n}"), "{out}");
    assert!(out.contains("U::Ay(v) => unsafe { napi::bindgen_prelude::ToNapiValue::to_napi_value(env, v) },"), "{out}");
    assert!(out.contains("let tag: String = obj.get_named_property(\"type\")?;"), "{out}");
    assert!(out.contains("\"a\" => Ok(U::Ay("), "{out}");
    assert!(out.contains("#[napi(js_name = \"type\", ts_type = \"'a'\")]\n    pub r#type: String,"), "{out}");
    assert!(out.contains("ts_return_type = \"A | B\""), "{out}");
}

// Catches: a record reached only through a signature left ungenerated -- the function would name a
// type that does not exist.
#[test]
fn a_record_a_signature_reaches_is_generated() {
    let src = r#"
        #[derive(Host)]
        #[binding(emit, from = lk_agents_abi::k::Prompts)]
        pub struct Prompts { pub AUDIO_SPECIFIC: String, pub count: u64 }
        #[host(export)]
        pub fn prompts() -> Prompts { todo!() }
    "#;
    let out = generate(src);
    assert!(out.contains("pub fn prompts() -> crate::generated::records::Prompts {"), "{out}");
    assert!(out.contains("#[napi(object, js_name = \"Prompts\")]"), "{out}");
    // A constant's SCREAMING name is kept, not squashed to `AUDIOSPECIFIC`.
    assert!(out.contains("#[napi(js_name = \"AUDIO_SPECIFIC\")]"), "{out}");
}

// --- naming ---------------------------------------------------------------------------------------

// Catches: two spellings of one Node name -- `node.rs`'s interfaces and the napi fields must come
// from one function, and it must be napi's camelCase.
#[test]
fn camel_is_napi_s_camel_case() {
    assert_eq!(node::camel("start_time"), "startTime");
    assert_eq!(node::camel("vad_speech_started_at"), "vadSpeechStartedAt");
    assert_eq!(node::camel("type"), "type");
    assert_eq!(node::camel("_provider_key"), "_providerKey");
    assert_eq!(node::camel("AUDIO_SPECIFIC"), "AUDIO_SPECIFIC");
}

// Catches: a stream or proxy Node cannot bind skipped with no trace -- each refusal is named.
#[test]
fn streams_and_proxies_node_cannot_bind_are_named_gaps() {
    let src = "#[host(stream(name = \"TtsStream\", sink = dyn StreamSink<String>, input = String, output = u32, error = crate::e))]\n\
               pub type TtsSource = Box<dyn StreamSource<u32>>;";
    let m = gaps(src);
    assert!(m.contains("- `TtsStream`: error = \"crate::e\" is not a #[host(error)] type"), "{m}");
    let m = gaps("#[host(proxy(host = \"h.C\"))]\npub trait T: Send + Sync {\nfn label(&self) -> String;\n}");
    assert!(m.contains("not yet: a bare proxy"), "{m}");
}

// --- group module paths ----------------------------------------------------------------------------

/// A `group` file as the tree has it: two `CartesiaOptions`, one per nested module.
const GROUPS: &str = r#"
    #[crate::host(group(kind = "typed_dict", total = "False", node))]
    pub mod gateway {
        #[crate::host(group(path = "livekit.agents.inference.stt"))]
        pub mod stt {
            pub struct CartesiaOptions { pub min_volume: Option<f64> }
        }
        #[crate::host(group(path = "livekit.agents.inference.tts"))]
        pub mod tts {
            pub struct CartesiaOptions { pub emotion: Option<String> }
        }
    }
"#;

// Catches: a `group` module's record bound against a guessed path (the file's), or not at all --
// the IR carries its inline modules, so the core path is exact with no `#[node(from)]` -- and two
// same-named group records collapsing into one napi item (`index.d.ts` is one namespace).
#[test]
fn a_group_record_is_bound_and_named_by_its_inline_modules() {
    let (label, module) = ("rust/crates/lk-agents-abi/src/inference/mod.rs", "lk_agents_abi::inference");
    let decls = parse::file(label, module, GROUPS).unwrap();
    let x = parse::exposed(label, "lk_agents_abi", GROUPS).unwrap();
    let out = napi::emit(&decls, &x).unwrap().into_iter().map(|o| o.text).collect::<String>();
    for m in ["stt", "tts"] {
        let js = format!("{}{}CartesiaOptions", m[..1].to_uppercase(), &m[1..]);
        assert!(out.contains(&format!("impl From<&lk_agents_abi::inference::gateway::{m}::CartesiaOptions> for {js} {{")), "{out}");
        assert!(out.contains(&format!("#[napi(object, js_name = \"{js}\")]")), "{out}");
    }
}

// --- streams -----------------------------------------------------------------------------------------

// Catches: a stream whose push is async (the sink is synchronous by contract), whose `recv` blocks
// the JS thread, or whose optional sink is pushed without its check (Python raises
// `RuntimeError(message)`; so does Node).
#[test]
fn a_stream_pushes_synchronously_and_receives_as_a_promise() {
    let src = "#[host(error(kind = \"RuntimeError\"))]\npub enum StreamError { A }\n\
               #[host(stream(name = \"TtsStream\", sink = dyn StreamSink<String>, input = String, output = u32, \
               error = StreamError, push = \"push_text\", arg = \"token\", end_input, close_ends_input, optional_sink = \"one-shot\"))]\n\
               pub type TtsSource = Box<dyn StreamSource<u32>>;";
    let out = generate(src);
    assert!(out.contains("#[napi(js_name = \"pushText\")]\n    pub fn push_text(&self, env: Env, token: String) -> napi::Result<()> {"), "{out}");
    assert!(out.contains("self.sink()?.push(token).map_err(crate::generated::errors::stream_error_to_node)"), "{out}");
    assert!(out.contains("JsError::new(\"RuntimeError\", \"one-shot\")"), "{out}");
    assert!(out.contains("#[napi(js_name = \"recv\", ts_return_type = \"Promise<number | null>\")]"), "{out}");
    assert!(out.contains("env.spawn_future_with_callback("), "{out}");
    assert!(out.contains("if let Some(sink) = &self.sink {\n            sink.end_input();\n        }"), "aclose ends the input first\n{out}");
}

// --- the small gaps ----------------------------------------------------------------------------------

// Catches: `Maybe<Option<T>>` folding `null` into "not given" -- `Agent(llm=None)` is "no LLM",
// not "unchanged". JS says the three answers as `undefined`, `null` and a value.
#[test]
fn maybe_option_keeps_undefined_and_null_apart() {
    let out = generate("#[host(export)]\npub fn f(a: Maybe<Option<String>>) {}");
    assert!(out.contains("#[napi(ts_arg_type = \"string | null | undefined\")] a: crate::host_error::MaybeNull<String>"), "{out}");
    assert!(out.contains("let a = a.into_core();"), "{out}");
}

// Catches: a lenient parameter raising on a JS value of the wrong type (it reads the default), or
// accepted on a type that converts (the default would swallow a real error).
#[test]
fn a_lenient_parameter_reads_the_default() {
    let out = generate("#[host(export)]\npub fn f(#[host(lenient)] n: u32) {}");
    assert!(out.contains("pub fn f(n: napi::bindgen_prelude::Unknown<'_>)"), "{out}");
    assert!(out.contains("let n: u32 = crate::host_error::lenient(&n);"), "{out}");
    assert!(gaps("#[host(export)]\npub fn f(#[host(lenient)] d: Duration) {}").contains("reads a plain value"));
}

// Catches: a defaulted `&str` refused (it was `encode_image(format = "png")`), or the omitted
// argument read as `""` instead of the declaration's literal.
#[test]
fn a_defaulted_borrowed_string_borrows_the_declared_literal() {
    let out = generate("#[host(export(defaults(format = \"png\")))]\npub fn f(format: &str) {}");
    assert!(out.contains("pub fn f(format: Option<String>)"), "{out}");
    assert!(out.contains("let format = format.as_deref().unwrap_or(\"png\");"), "{out}");
}
