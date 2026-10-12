//! The expansions as text. What Python observes of them (names, signatures, docs, exceptions) is
//! pinned end to end by `lk-agents-py`'s export-table test and the `_core` surface diff; these pin
//! the choices that diff cannot localise.

use quote::quote;

fn text(t: proc_macro2::TokenStream) -> String {
    t.to_string()
}

fn export(attr: proc_macro2::TokenStream, item: proc_macro2::TokenStream) -> syn::Result<String> {
    crate::export_fn(attr, &syn::parse2(item).unwrap()).map(text)
}

fn object(attr: proc_macro2::TokenStream, item: proc_macro2::TokenStream) -> syn::Result<String> {
    crate::object(attr, &syn::parse2(item).unwrap()).map(text)
}

fn class(attr: proc_macro2::TokenStream, item: proc_macro2::TokenStream) -> syn::Result<String> {
    crate::class(attr, &syn::parse2(item).unwrap()).map(text)
}

// Catches: the fork guard dropped from a default export or kept on a `pid = false` one; the `*`
// landing anywhere but before the `kw_only` parameter; a default's Rust expression mangled; the
// registration not naming the wrapper; the whole thing escaping the `python` feature.
#[test]
fn an_export_is_a_guarded_registered_pyfunction_with_the_declared_signature() {
    let out = export(
        quote!(export(kw_only = "retain_format", defaults(min_sentence_len = 20, retain_format = false))),
        quote! { pub fn split_sentences(text: &str, min_sentence_len: usize, retain_format: bool) -> Result<Vec<Sentence>, Error> { todo!() } },
    )
    .unwrap();
    assert!(out.starts_with("# [cfg (feature = \"python\")] const _ : () ="), "{out}");
    assert!(out.contains("# [pyo3 (name = \"split_sentences\" , signature = (text , min_sentence_len = 20 , * , retain_format = false))]"), "{out}");
    assert!(out.contains("fn __host_py_split_sentences (text : & str , min_sentence_len : usize , retain_format : bool) -> :: pyo3 :: PyResult < Vec < Sentence > >"), "{out}");
    assert!(out.contains(":: lk_agents_abi :: python :: check_pid () ?"), "{out}");
    assert!(out.contains("let out = split_sentences (text , min_sentence_len , retain_format) ;"), "{out}");
    assert!(out.contains("wrap_pyfunction ! (__host_py_split_sentences , m)"), "{out}");
    let out = export(quote!(export(pid = false)), quote! { pub fn f(x: u8) -> u8 { x } }).unwrap();
    assert!(!out.contains("check_pid"), "{out}");
    assert!(!out.contains("signature"), "no defaults, no kw_only: pyo3 infers the same text: {out}");
}

// Catches: an explicit `err` kind not reaching the exception, or an undeclared error bypassing the
// autoref dispatch (then a `#[host(error)]` type elsewhere could never pick its own class).
#[test]
fn a_failure_raises_the_named_kind_or_goes_through_the_error_dispatch() {
    let out = export(quote!(export), quote! { pub fn f(x: &[u8]) -> Result<Vec<u8>, String> { todo!() } }).unwrap();
    assert!(out.contains("(& :: lk_agents_abi :: python :: Fail (& e)) . py_err ()"), "{out}");
    let out = object(
        quote!(object(name = "AudioDecoder", cell = "mutex")),
        quote! { impl Decoder { #[host(method(name = "push", err = "RuntimeError"))] pub fn push_bytes(&mut self, data: &[u8]) -> Result<Vec<u8>> { todo!() } } },
    )
    .unwrap();
    assert!(out.contains(":: pyo3 :: exceptions :: PyRuntimeError :: new_err (e . to_string ())"), "{out}");
    assert!(!out.contains("Fail"), "{out}");
}

// Catches: the free function's `__doc__` growing past the first paragraph, or an intra-doc link
// reaching Python verbatim.
#[test]
fn a_function_doc_is_its_first_paragraph_unlinked() {
    let out = export(
        quote!(export),
        quote! {
            /// Encode [`RgbaFrame`] bytes,
            /// as the reference does.
            ///
            /// Not for Python.
            pub fn f() {}
        },
    )
    .unwrap();
    assert!(out.contains("# [doc = \" Encode `RgbaFrame` bytes,\"] # [doc = \" as the reference does.\"]"), "{out}");
    assert!(!out.contains("Not for Python"), "{out}");
}

// Catches: a plain object's wrappers not calling the core method through `Self::` (a wrapper with
// the core's own name would recurse or collide), the magic/renamed method or getter losing its
// Python name, `detach` not releasing the GIL around the call, or the class-name check missing.
#[test]
fn a_plain_object_is_pymethods_on_the_type_itself() {
    let out = object(
        quote!(object(name = "StreamBuffer")),
        quote! {
            impl StreamBuffer {
                #[host(new)] pub fn new() -> Self { todo!() }
                #[host(method(name = "__len__"))] pub fn len(&self) -> usize { 0 }
                #[host(method(name = "read", detach, defaults(size = -1)))] pub fn read_up_to(&self, size: isize) -> Vec<u8> { todo!() }
                pub fn read(&self, buf: &mut [u8]) -> usize { 0 }
            }
        },
    )
    .unwrap();
    assert!(out.contains("# [:: pyo3 :: pymethods] impl StreamBuffer"), "{out}");
    assert!(out.contains("same_name (< StreamBuffer as :: pyo3 :: PyClass > :: NAME , \"StreamBuffer\")"), "{out}");
    assert!(out.contains("# [new] fn __host_py_new () -> :: pyo3 :: PyResult < Self >"), "{out}");
    assert!(out.contains("# [pyo3 (name = \"__len__\")] fn __host_py_len (& self ,)"), "{out}");
    assert!(out.contains("# [pyo3 (name = \"read\" , signature = (size = - 1))]"), "{out}");
    assert!(out.contains("fn __host_py_read_up_to < 'py > (& self , py : :: pyo3 :: Python < 'py > , size : isize)"), "{out}");
    assert!(out.contains("let out = py . detach (|| Self :: read_up_to (self , size)) ;"), "{out}");
    assert!(!out.contains("__host_py_read ("), "an unmarked method is not bound: {out}");
}

// Catches: a `mutex` object emitting a non-frozen class (two threads would race the borrow flag),
// a `&mut self` core method bound as `&mut self` on the frozen class (does not compile), the lock
// not being taken for the call, or the constructor not wrapping the core value.
#[test]
fn a_mutex_object_is_a_frozen_class_over_the_core_type_beside_it() {
    let out = object(
        quote!(object(name = "AudioDecoder", cell = "mutex")),
        quote! {
            /// `AudioStreamDecoder`'s engine.
            impl Decoder {
                #[host(new(kw_only = "sample_rate", defaults(sample_rate = 48000)))] pub fn new(mime: &str, sample_rate: u32) -> Result<Self> { todo!() }
                #[host(getter(name = "output_format"))] pub fn output_format_pair(&self) -> Option<(u32, u16)> { None }
                #[host(method(name = "flush"))] pub fn flush_bytes(&mut self) -> Result<Vec<u8>> { todo!() }
            }
        },
    )
    .unwrap();
    assert!(out.contains("# [doc = \" `AudioStreamDecoder`'s engine.\"] # [:: pyo3 :: pyclass (frozen , name = \"AudioDecoder\" , module = \"livekit.agents._core\")] pub struct PyAudioDecoder { pub inner : :: std :: sync :: Mutex < Decoder > , }"), "{out}");
    assert!(out.contains("# [pyo3 (signature = (mime , * , sample_rate = 48000))] fn __host_py_new (mime : & str , sample_rate : u32) -> :: pyo3 :: PyResult < PyAudioDecoder >"), "{out}");
    assert!(out.contains("let out = < Decoder > :: new (mime , sample_rate) ;"), "{out}");
    assert!(out.contains("Ok (PyAudioDecoder { inner : :: std :: sync :: Mutex :: new (out) })"), "{out}");
    assert!(out.contains("# [getter (output_format)] fn __host_py_output_format_pair (& self ,)"), "{out}");
    assert!(out.contains("fn __host_py_flush_bytes (& self ,)"), "{out}");
    assert!(out.contains("self . inner . lock () . unwrap_or_else (:: std :: sync :: PoisonError :: into_inner) . flush_bytes ()"), "{out}");
    assert!(out.contains("m . add_class :: < PyAudioDecoder > ()"), "{out}");
}

// Catches: `subclass` not reaching pyo3 (Python then refuses the class as a base, which the
// reference's pure-Python `ExpFilter`/`MovingAverage` never did), or it being on by default.
#[test]
fn subclass_is_an_opt_in_pyclass_option() {
    let out = class(quote!(name = "ExpFilter", mutable, subclass), quote! { pub struct F; }).unwrap();
    assert!(out.contains(":: pyo3 :: pyclass (subclass , skip_from_py_object , name = \"ExpFilter\" ,"), "{out}");
    let out = class(quote!(), quote! { pub struct F; }).unwrap();
    assert!(!out.contains("subclass"), "{out}");
}

// Catches: the class's `__doc__` becoming the whole rustdoc, rustdoc losing its later paragraphs,
// or the summary `cfg_attr`s landing after the `pyclass` -- rustc expands attributes in order, so
// pyo3 would then see an unexpanded `cfg_attr` and no doc at all.
#[test]
fn a_class_is_frozen_and_documented_by_its_first_paragraph() {
    let out = class(
        quote!(),
        quote! {
            /// A thread-safe buffer.
            ///
            /// Rust readers only.
            #[derive(Default)]
            pub struct StreamBuffer { x: u8 }
        },
    )
    .unwrap();
    let summary = out.find("# [cfg_attr (feature = \"python\" , doc = \" A thread-safe buffer.\")]").expect(&out);
    let pyclass = out
        .find("# [cfg_attr (feature = \"python\" , :: pyo3 :: pyclass (frozen , skip_from_py_object , name = \"StreamBuffer\" , module = \"livekit.agents._core\"))]")
        .expect(&out);
    assert!(summary < pyclass, "{out}");
    // `///` lines reach the macro as raw strings.
    assert!(out.contains("# [cfg_attr (not (feature = \"python\") , doc = r\" Rust readers only.\")]"), "{out}");
    assert!(!out.contains("feature = \"python\" , doc = \" Rust readers only.\""), "{out}");
    assert!(out.contains("m . add_class :: < StreamBuffer > ()"), "{out}");
    let out = class(quote!(name = "Buf", mutable), quote! { pub struct B; }).unwrap();
    assert!(out.contains(":: pyo3 :: pyclass (skip_from_py_object , name = \"Buf\" ,"), "{out}");
    let out = class(quote!(from_py_object), quote! { #[derive(Clone)] pub struct B; }).unwrap();
    assert!(out.contains(":: pyo3 :: pyclass (frozen , from_py_object , name = \"B\""), "{out}");
}

// Catches: a key or marker the expansion does not implement being accepted and silently dropped --
// in place, a dropped `extend` or `convert` is a binding that compiles and behaves differently.
#[test]
fn what_does_not_expand_in_place_yet_is_refused_by_name() {
    let refused = |r: syn::Result<String>, want: &str| {
        let e = r.expect_err("must be refused").to_string();
        assert!(e.contains(want), "{e}");
    };
    refused(object(quote!(object(extend = "X")), quote! { impl T {} }), "object.extend");
    refused(object(quote!(object), quote! { impl T { #[host(method(clock = "c"))] pub fn c(&self) {} } }), "clock");
    refused(object(quote!(object), quote! { impl T { #[host(method)] pub fn m(&self, #[host(convert = "f")] x: u8) {} } }), "parameter marker");
    refused(object(quote!(object(rust_name = PyT)), quote! { impl T {} }), "rust_name");
    refused(object(quote!(object), quote! { impl T { #[host(getter)] pub fn m() -> u8 { 0 } } }), "needs a receiver");
    refused(object(quote!(object), quote! { impl T { #[host(new)] pub fn m(&self) -> Self { todo!() } } }), "takes no receiver");
    refused(object(quote!(object), quote! { impl T { #[host(method(name = n))] pub fn m(&self) {} } }), "host text");
    refused(export(quote!(export(defaults(x = None))), quote! { pub fn f(#[host(required)] x: Option<u8>) {} }), "`required` and has a default");
    refused(export(quote!(export(err = "RuntimeError")), quote! { pub fn f() -> u8 { 0 } }), "does not return a Result");
    refused(export(quote!(export(defaults(y = 1))), quote! { pub fn f(x: u8) {} }), "names no parameter");
    refused(class(quote!(frozen), quote! { pub struct S; }), "frozen");
}

// Catches: a const export named by anything but its lower-cased identifier (or `name`), the fork
// guard kept on a `pid = false` const, a `Duration` reaching Python as pyo3's `timedelta` instead
// of the codegen's float seconds, or an associated const reached without its type.
#[test]
fn a_const_is_a_zero_argument_function_and_a_duration_is_float_seconds() {
    let c: syn::ItemConst = syn::parse2(quote! {
        /// Default ping interval.
        pub const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(30);
    })
    .unwrap();
    let out = text(crate::export_const(quote!(export(pid = false)), &c).unwrap());
    assert!(out.contains("# [pyo3 (name = \"heartbeat_interval\")] fn __host_py_heartbeat_interval () -> :: pyo3 :: PyResult < f64 >"), "{out}");
    assert!(out.contains("let out = HEARTBEAT_INTERVAL ; let out = out . as_secs_f64 () ;"), "{out}");
    assert!(!out.contains("check_pid"), "{out}");
    let imp: syn::ItemImpl = syn::parse2(quote! {
        impl DefaultLoadCalc {
            #[host(export(name = "default_load_window"))] pub const WINDOW: usize = 5;
            pub const OTHER: usize = 1;
        }
    })
    .unwrap();
    let out = text(crate::export_impl(quote!(export), &imp).unwrap());
    assert!(out.contains("let out = < DefaultLoadCalc > :: WINDOW ;"), "{out}");
    assert!(out.contains(":: lk_agents_abi :: python :: check_pid () ?"), "{out}");
    assert!(!out.contains("OTHER"), "an unmarked const is not exported: {out}");
    // A parameter: seconds in, back to a `Duration` with the codegen's `ValueError` label.
    let out = export(quote!(export), quote! { pub fn f(wait: Option<Duration>) -> Duration { todo!() } }).unwrap();
    assert!(out.contains("wait : :: std :: option :: Option < f64 >"), "{out}");
    assert!(out.contains("let wait = { let v = wait ; v . map ("), "{out}");
    assert!(out.contains("-> :: pyo3 :: PyResult < f64 >"), "{out}");
}

// Catches: a `wire_enum` crossing as anything but its wire string, or an unknown string raising
// something other than `ValueError`.
#[test]
fn a_wire_enum_crosses_as_its_string_both_ways() {
    let out = text(crate::wire_enum(&syn::parse_str("CpuMonitorKind").unwrap()));
    assert!(out.contains("PyString :: new (py , self . as_str ())"), "{out}");
    assert!(out.contains("< Self as :: core :: str :: FromStr > :: from_str (& s) . map_err (:: pyo3 :: exceptions :: PyValueError :: new_err)"), "{out}");
    assert!(out.starts_with("# [cfg (feature = \"python\")]"), "{out}");
}

// Catches: the switch matching a crate by a prefix of its name (`lk-agents-mediax` turning in
// place with `lk-agents-media`), a non-core crate (Node's) expanding, or missing rustc's
// workspace-relative and absolute spellings of a file.
#[test]
fn a_file_is_in_place_by_its_crate_whatever_spelling_rustc_uses() {
    use std::path::Path;
    assert!(crate::in_place(Some(Path::new("crates/lk-agents-media/src/decoder.rs"))));
    assert!(crate::in_place(Some(Path::new("/Users/x/repo/rust/crates/lk-agents-media/src/images/encode.rs"))));
    assert!(!crate::in_place(Some(Path::new("crates/lk-agents-node/src/decoder.rs"))));
    assert!(!crate::in_place(Some(Path::new("crates/lk-agents-mediax/src/decoder.rs"))));
    assert!(!crate::in_place(None));
}

// Catches: a JSON argument extracted by anything but the checked reader with the argument's name
// (or its `label`) in the error, a borrowed core parameter not lent the converted value, or a
// JSON result nested in a container escaping the conversion.
#[test]
fn json_crosses_through_the_checked_readers_and_a_borrow_is_lent() {
    let out = export(
        quote!(export),
        quote! { pub fn f(schema: &Value, root: Option<&Value>, #[host(label = "tool arguments")] v: Value, tags: &[&str], ids: &[String], m: Extra) -> Vec<(&'static str, Value, Extra)> { todo!() } },
    )
    .unwrap();
    assert!(out.contains("fn __host_py_f < 'py > (py : :: pyo3 :: Python < 'py > , schema : & :: pyo3 :: Bound < 'py , :: pyo3 :: PyAny >"), "{out}");
    assert!(out.contains("value_from_py (v , \"schema\")"), "{out}");
    assert!(out.contains("value_from_py (v , \"tool arguments\")"), "{out}");
    assert!(out.contains("root : :: std :: option :: Option < & :: pyo3 :: Bound < 'py , :: pyo3 :: PyAny > >"), "{out}");
    assert!(out.contains("tags : :: std :: vec :: Vec < :: std :: string :: String >"), "{out}");
    assert!(out.contains("ids : :: std :: vec :: Vec < String >"), "{out}");
    assert!(out.contains("map_from_py (v , \"m\")"), "{out}");
    assert!(out.contains("let out = f (& schema , root . as_ref () , v , & tags , & ids , m) ;"), "{out}");
    assert!(out.contains("-> :: pyo3 :: PyResult < :: std :: vec :: Vec < (& 'static str , :: pyo3 :: Bound < 'py , :: pyo3 :: PyAny > , :: pyo3 :: Bound < 'py , :: pyo3 :: types :: PyDict > ,) > >"), "{out}");
    assert!(out.contains("value_to_py (py , & v1) ?"), "{out}");
}

// Catches: a variant's kind not reaching its arm, a dotted kind not importing the host class, a
// `delegate` not asking the wrapped error's own table, the default kind lost for unmarked
// variants (in `to_py_err` or `host_kind`), or a binding-crate converter path (`with`) accepted in
// a core attribute.
#[test]
fn an_error_is_a_host_exception_by_its_kind_table() {
    let e: syn::Item = syn::parse2(quote! {
        pub enum SchemaError {
            #[host(delegate)] Strict(StrictError),
            #[host(kind = "TypeError")] NotAnObject { path: String },
            #[host(kind = "asyncio.CancelledError")] Cancelled,
            Other(String),
        }
    })
    .unwrap();
    let out = text(crate::error(quote!(error(kind = "RuntimeError")), &e).unwrap());
    assert!(out.contains("impl :: lk_agents_abi :: python :: HostException for SchemaError"), "{out}");
    assert!(out.contains("Self :: Strict (inner) => :: lk_agents_abi :: python :: HostException :: to_py_err (inner)"), "{out}");
    assert!(out.contains("Self :: NotAnObject { .. } => :: pyo3 :: exceptions :: PyTypeError :: new_err"), "{out}");
    assert!(out.contains("Self :: Cancelled => :: lk_agents_abi :: python :: host_exception (\"asyncio\" , \"CancelledError\""), "{out}");
    assert!(out.contains("_ => :: pyo3 :: exceptions :: PyRuntimeError :: new_err"), "{out}");
    // The same table, as the class name a non-Python binding raises.
    assert!(out.contains("Self :: Strict (inner) => inner . host_kind ()"), "{out}");
    assert!(out.contains("Self :: NotAnObject { .. } => \"TypeError\""), "{out}");
    assert!(out.contains("Self :: Cancelled => \"asyncio.CancelledError\""), "{out}");
    assert!(out.contains("_ => \"RuntimeError\""), "{out}");
    let e: syn::Item = syn::parse2(quote! { pub struct E; }).unwrap();
    let out = text(crate::error(quote!(error(with = crate::x::to_py)), &e).unwrap());
    assert!(out.contains("impl :: lk_agents_abi :: python :: HostException for E"), "{out}");
    assert!(out.contains("crate :: x :: to_py (:: std :: clone :: Clone :: clone (self))"), "{out}");
    let out = export(quote!(export(err = "livekit.agents.llm.ToolError")), quote! { pub fn f() -> Result<(), E> { todo!() } }).unwrap();
    assert!(out.contains("host_exception (\"livekit.agents.llm\" , \"ToolError\" , e . to_string ())"), "{out}");
}

// Catches: a keyword record's fields not reaching the wrapper's signature (the struct and the
// method expand separately, so the spread goes through the struct's macro), the `*` of `kw_only`
// landing inside the fields instead of before them, or the record rebuilt after the call.
#[test]
fn a_keyword_record_parameter_is_spread_through_the_records_macro() {
    let st: syn::ItemStruct = syn::parse2(quote! {
        pub struct Frame {
            pub agent: String,
            #[host(kwarg(default = None))]
            pub tools: Option<Vec<String>>,
        }
    })
    .unwrap();
    let mac = text(crate::kwargs(quote!(kwargs), &st).unwrap());
    assert!(mac.contains("macro_rules ! __host_kwargs_Frame"), "{mac}");
    assert!(mac.contains("# [pyo3 (signature = ($ ($ sig_pre) * agent , tools = None , $ ($ sig_post) *))]"), "{mac}");
    assert!(mac.contains("let $ var = Frame { agent , tools } ;"), "{mac}");
    let out = object(
        quote!(object),
        quote! { impl T { #[host(method(kw_only = "frame"))] pub fn update(&self, id: u64, #[host(kwargs)] frame: Frame) {} } },
    )
    .unwrap();
    assert!(out.contains("__host_kwargs_Frame ! {"), "{out}");
    assert!(out.contains("{ id , * , } { }"), "{out}");
    assert!(out.contains("{ & self , py : :: pyo3 :: Python < 'py > , id : u64 , } { }"), "{out}");
    assert!(out.contains("let out = Self :: update (self , id , frame) ;"), "{out}");
}

// Catches: a method returning `Reply<T>` awaited on the calling thread (it would block the loop),
// its send refusal raised from the awaitable instead of at the call, the reply converted without
// the GIL, or a method returning anything else handed back as an awaitable.
#[test]
fn a_reply_method_raises_its_refusal_now_and_awaits_the_reply_on_the_loop() {
    let out = object(
        quote!(object),
        quote! { impl T { #[host(method)] pub fn ask(&self) -> Result<Reply<bool>, Refused> { todo!() } } },
    )
    .unwrap();
    let send = out.find("let out = Self :: ask (self ,) ;").expect(&out);
    let refusal = out.find("(& :: lk_agents_abi :: python :: Fail (& e)) . py_err ()").expect(&out);
    let future = out.find("future_into_py (py , async move { let out = reply . await ;").expect(&out);
    assert!(send < refusal && refusal < future, "{out}");
    assert!(out.contains("-> :: pyo3 :: PyResult < :: pyo3 :: Bound < 'py , :: pyo3 :: PyAny > >"), "{out}");
    let plain = object(quote!(object), quote! { impl T { #[host(method)] pub fn ask(&self) -> bool { todo!() } } }).unwrap();
    assert!(!plain.contains("future_into_py") && plain.contains("-> :: pyo3 :: PyResult < bool >"), "{plain}");
}

// Catches: a method's kind read from anything but its signature -- a receiver-less `method` not
// becoming a staticmethod, an `async fn` not handed back as an awaitable.
#[test]
fn a_methods_kind_is_its_signatures() {
    let out = object(
        quote!(object(cell = "arc")),
        quote! { impl T {
            #[host(method)] pub fn make() -> u8 { 0 }
            #[host(method)] pub async fn wait(&self) {}
        } },
    )
    .unwrap();
    assert!(out.contains("# [staticmethod] # [pyo3 (name = \"make\")]"), "{out}");
    assert!(out.contains("future_into_py (py , async move { let out = inner . wait () . await ;"), "{out}");
}

// Catches: the session clock not scoped over a proxied argument's conversion (a provider proxy
// would time out on the wall clock under a virtual one), or a probe on every wrapper that builds
// no proxy at all.
#[test]
fn a_receiver_scopes_its_clock_only_over_calls_that_build_a_proxy() {
    let out = object(
        quote!(object),
        quote! { impl T {
            #[host(method)] pub fn set_llm(&self, llm: Option<Arc<dyn Llm>>) {}
            #[host(method)] pub fn plain(&self, x: u8) {}
        } },
    )
    .unwrap();
    let set_llm = out.find("fn __host_py_set_llm").expect(&out);
    let plain = out.find("fn __host_py_plain").expect(&out);
    let probe = out.find("ClockOf (self)").expect(&out);
    let proxy = out.find("HostProxy > :: from_host").expect(&out);
    assert!(set_llm < probe && probe < proxy && proxy < plain, "{out}");
    assert_eq!(out.matches("ClockOf (").count(), 1, "{out}");
}

// Catches: a constant exported under a Python constant's name becoming a function (the host reads
// `_core.DTMF_PUBLISH_DELAY`, not calls it), or a lowercase one becoming an attribute.
#[test]
fn an_upper_case_constant_is_a_module_attribute_and_any_other_a_function() {
    let c: syn::ItemConst = syn::parse2(quote! { pub const DELAY: f64 = 0.3; }).unwrap();
    let attr = text(crate::export_const(quote!(export(name = "DTMF_DELAY")), &c).unwrap());
    assert!(attr.contains("m . add (\"DTMF_DELAY\" , DELAY)"), "{attr}");
    assert!(!attr.contains("pyfunction"), "{attr}");
    let func = text(crate::export_const(quote!(export), &c).unwrap());
    assert!(func.contains("# [pyo3 (name = \"delay\")]"), "{func}");
}

// Catches: an `Option<T>` parameter left required when every parameter after it has a default (the
// 207 `defaults(x = None)` it replaces), given a default a required parameter follows (a Python
// syntax error at import), or defaulted despite `#[host(required)]`.
#[test]
fn an_option_defaults_to_none_unless_required_or_a_required_parameter_follows() {
    let out = export(quote!(export), quote! { pub fn f(a: Option<u8>, b: u8, c: Option<u8>, d: Option<u8>) {} }).unwrap();
    assert!(out.contains("signature = (a , b , c = None , d = None)"), "{out}");
    let out = export(quote!(export), quote! { pub fn f(c: Option<u8>, #[host(required)] d: Option<u8>) {} }).unwrap();
    assert!(!out.contains("signature"), "nothing defaults, so pyo3 infers the signature: {out}");
    let out = export(quote!(export(defaults(b = 2))), quote! { pub fn f(a: Option<u8>, b: u8) {} }).unwrap();
    assert!(out.contains("signature = (a = None , b = 2)"), "{out}");
}

// Catches: a Rust value read as host text or the other way -- a quoted default taken for Rust
// source (the old grammar), a string literal default not reaching pyo3 as a string, a quoted path
// accepted where Rust is written unquoted, an unquoted host name accepted.
#[test]
fn rust_is_written_unquoted_and_host_text_as_a_string() {
    let out = export(quote!(export(defaults(n = -1, s = "auto"))), quote! { pub fn f(n: i64, s: &str) {} }).unwrap();
    assert!(out.contains("signature = (n = - 1 , s = \"auto\")"), "{out}");
    let e = object(quote!(object(cell = "mutex", rust_name = "PyT")), quote! { impl T {} }).expect_err("a quoted ident");
    assert!(e.to_string().contains("write it unquoted"), "{e}");
    let e = export(quote!(export(name = f2)), quote! { pub fn f() {} }).expect_err("an unquoted name");
    assert!(e.to_string().contains("host text"), "{e}");
    let out = object(quote!(object(cell = "mutex", rust_name = Cell)), quote! { impl T { #[host(new)] pub fn new() -> Self { todo!() } } }).unwrap();
    assert!(out.contains("pub struct Cell {"), "{out}");
}

// Catches: a `clock` proxy not keeping the clock it is built with (a view it returns would run on
// the wall clock), or a proxy without it growing the field.
#[test]
fn a_clock_proxy_keeps_the_callers_clock() {
    let tr: syn::ItemTrait = syn::parse2(quote! { pub trait Stt: Send + Sync { fn label(&self) -> String; } }).unwrap();
    let out = crate::proxy(quote!(proxy(host = "livekit.agents.stt.STT", clock)), &tr).unwrap().to_string();
    assert!(out.contains("pub clock : :: std :: sync :: Arc < dyn :: lk_agents_abi :: Clock >"), "{out}");
    assert!(out.contains("from_bridge (bridge , clock . unwrap_or_else"), "{out}");
    let out = crate::proxy(quote!(proxy(host = "livekit.agents.stt.STT")), &tr).unwrap().to_string();
    assert!(!out.contains("pub clock"), "{out}");
}

// Catches: a record's extract error naming anything but the record (the twin names the old
// grammar required), or a key of the deleted twin grammar (`module`, `name`, `extract_name`, a
// `from` naming the record itself) accepted and silently ignored.
#[test]
fn a_record_is_named_by_itself_and_the_twin_keys_are_refused() {
    let rec = |attr: proc_macro2::TokenStream| -> syn::Result<String> {
        let input: syn::DeriveInput = syn::parse2(quote! { #[derive(Host)] #attr pub struct Usage { pub tokens: u32, pub model: String } }).unwrap();
        crate::record(&input).map(text)
    };
    let out = rec(quote!(#[binding(extract)])).unwrap();
    assert!(out.contains("extract_struct_field :: < u32 > (& value , \"Usage\" , \"tokens\")"), "{out}");
    for bad in [quote!(#[binding(emit, module = "m")]), quote!(#[binding(extract, extract_name = "UsageFromPy")]), quote!(#[binding(emit, name = "UsagePy")])] {
        let e = rec(bad).expect_err("a twin key");
        assert!(e.to_string().contains("is not a key here"), "{e}");
    }
    let e = rec(quote!(#[binding(emit, from = crate::Usage)])).expect_err("from the record itself");
    assert!(e.to_string().contains("leave it out"), "{e}");
}

// Catches: a record-level `default` / `skip_if_none` (or a `total = "False"` typed dict's implied
// `default`) expanding differently from the same words written on each field, reaching a field
// that says its own, or `skip_if_none` reaching a field that is not an `Option`.
#[test]
fn record_level_field_words_expand_as_the_same_words_on_each_field() {
    let expand = |t: proc_macro2::TokenStream| text(crate::record(&syn::parse2(t).unwrap()).unwrap());
    let per_field = expand(quote! {
        #[binding(emit, extract, from_items)]
        pub struct R {
            #[binding(skip_if_none, default)] pub a: Option<f64>,
            #[binding(default)] pub b: u32,
            #[binding(default = 7)] pub c: u32,
        }
    });
    let hoisted = expand(quote! {
        #[binding(emit, extract, from_items, default, skip_if_none)]
        pub struct R {
            pub a: Option<f64>,
            pub b: u32,
            #[binding(default = 7)] pub c: u32,
        }
    });
    assert_eq!(hoisted, per_field);
    let open = expand(quote! {
        #[python(typed_dict, total = "False", module = "m")]
        #[binding(emit, extract, from_items, skip_if_none)]
        pub struct R {
            pub a: Option<f64>,
            pub b: u32,
            #[binding(default = 7)] pub c: u32,
        }
    });
    assert_eq!(open, per_field);
    let closed = expand(quote! {
        #[python(typed_dict, module = "m")]
        #[binding(emit, extract, from_items, skip_if_none)]
        pub struct R { pub a: Option<f64>, pub b: u32, #[binding(default = 7)] pub c: u32 }
    });
    assert_ne!(closed, per_field, "a total typed dict implies nothing");
}

fn filled(item: proc_macro2::TokenStream) -> syn::Result<syn::ItemImpl> {
    let mut imp: syn::ItemImpl = syn::parse2(item).unwrap();
    crate::fill_sends(&mut imp).map(|()| imp)
}

// Catches: a `send` method's body not sending its expression, its refusal swallowed where the
// signature returns it (or returned where it returns nothing), the declaration's docs, markers or
// visibility lost on the way, or the expansion reading it as anything but an ordinary method.
#[test]
fn a_send_method_is_written_as_a_send_of_its_expression() {
    let imp = filled(quote! { impl S {
        /// Drop the turn.
        #[host(method(send = TurnCommand::Clear))]
        pub fn clear(&self) -> Result<(), Refused>;
        #[host(method(send = Hold { id: HoldId(hold) }))]
        pub fn release(&self, hold: u64);
        #[host(method)]
        pub fn plain(&self) -> u8 { 0 }
    } })
    .unwrap();
    let text = quote!(#imp).to_string();
    assert!(text.contains("# [doc = r\" Drop the turn.\"] # [host (method (send = TurnCommand :: Clear))] # [allow (clippy :: needless_question_mark)] pub fn clear (& self) -> Result < () , Refused > { :: core :: result :: Result :: Ok (self . send (TurnCommand :: Clear) ?) }"), "{text}");
    assert!(text.contains("pub fn release (& self , hold : u64) { let _ = self . send (Hold { id : HoldId (hold) }) ; }"), "{text}");
    assert!(text.contains("pub fn plain (& self) -> u8 { 0 }"), "{text}");
    let out = crate::object(quote!(object), &imp).unwrap().to_string();
    assert!(out.contains("let out = Self :: clear (self ,) ;") && out.contains("let out = Self :: release (self , hold) ;"), "{out}");
}

// Catches: a body-less method compiled with no body at all, `send` beside a body it would silently
// replace, or a `send` method whose return cannot carry the refusal.
#[test]
fn a_body_is_written_by_send_or_by_hand_never_both_nor_neither() {
    let err = |item| filled(item).err().map(|e| e.to_string()).unwrap_or_default();
    assert!(err(quote! { impl S { #[host(method)] pub fn f(&self); } }).contains("needs #[host(method(send = <expr>))]"));
    assert!(err(quote! { impl S { #[host(method(send = C::X))] pub fn f(&self) -> Result<(), E> { todo!() } } }).contains("`send` writes the body"));
    assert!(err(quote! { impl S { #[host(method(send = C::X))] pub fn f(&self) -> bool; } }).contains("returns nothing"));
    assert!(err(quote! { impl S { #[host(method(send = C::X))] pub fn f() -> Result<(), E>; } }).contains("takes `&self`"));
}
