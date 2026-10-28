//! Generated callable arguments, async functions and native class contracts.

use super::*;

#[test]
fn async_module_functions_own_arguments_and_await_the_implementation() {
    let declarations = read(r#"
        #[crate::host(functions)]
        pub trait Formatting {
            #[host(function, name = "prepare")]
            async fn run(text: String, #[host(kw, default = false)] strict: bool) -> Result<Vec<String>, crate::errors::Bad>;
        }
    "#).expect("async module function resolves");
    assert_eq!(declarations.classes[0].methods[0].kind, MethodKind::AsyncFunction);
    let out = crate::adapter::emit(&declarations).expect("async function adapter emits");
    let llm = &out.iter().find(|o| o.path == "llm.rs").unwrap().text;
    assert!(
        llm.contains(
            "pub fn prepare<'py>(py: Python<'py>, text: ::std::string::String, strict: bool)"
        ),
        "{llm}"
    );
    assert!(llm.contains("future_into_py(py, async move"), "{llm}");
    assert!(llm.contains("::run(text, strict).await.map_err("), "{llm}");
    assert!(!llm.contains("&self") && !llm.contains("let this ="), "{llm}");
    let stubs = crate::class_stub::emit(&declarations).expect("async function stub emits");
    let text = stubs.iter().map(|stub| stub.out.text.as_str()).collect::<String>();
    assert!(
        text.contains("async def prepare(text: str, *, strict: bool = False) -> list[str]"),
        "{text}"
    );
}

#[test]
fn raw_rust_parameter_names_keep_their_python_keyword_spelling() {
    let declarations = read(
        r#"
        #[crate::host(functions)]
        pub trait Schema {
            #[host(function)] fn resolve(r#ref: &str) -> String;
        }
    "#,
    )
    .unwrap();
    let adapters = crate::adapter::emit(&declarations).unwrap();
    let adapter = &adapters.iter().find(|out| out.path == "llm.rs").unwrap().text;
    assert!(
        adapter.contains("r#ref: ::std::string::String") && adapter.contains("::resolve(&r#ref)"),
        "{adapter}"
    );
    let stubs = crate::class_stub::emit(&declarations).unwrap();
    let text = stubs.iter().map(|stub| stub.out.text.as_str()).collect::<String>();
    assert!(text.contains("def resolve(ref: str) -> str") && !text.contains("r#ref"), "{text}");
}

#[test]
fn wire_enum_parameters_keep_their_existing_native_conversion() {
    let declarations = read(
        r#"
        #[crate::host(wire_enum)]
        pub enum Role { User, Assistant }
        #[crate::host(functions)]
        pub trait Roles {
            #[host(function)] fn echo(role: Role) -> Role;
        }
    "#,
    )
    .expect("legacy wire enum is a declared string boundary");
    let adapters = crate::adapter::emit(&declarations).unwrap();
    let adapter = &adapters.iter().find(|out| out.path == "llm.rs").unwrap().text;
    assert!(adapter.contains("role: ::lk_agents_abi::llm::class::Role"), "{adapter}");
    assert!(adapter.contains("::echo(role)") && !adapter.contains("from_py"), "{adapter}");
    let stubs = crate::class_stub::emit(&declarations).unwrap();
    let text = stubs.iter().map(|stub| stub.out.text.as_str()).collect::<String>();
    assert!(text.contains("def echo(role: str) -> str"), "{text}");
}

#[test]
fn maybe_parameters_preserve_omission_null_and_given_values() {
    let declarations = read(r#"
        use crate::utils::maybe::Maybe;
        use serde_json::Value;
        #[crate::host(functions)]
        pub trait Settings {
            #[host(function)] fn update(#[host(kw)] scalar: Maybe<Option<f64>>, data: Maybe<Option<Value>>);
        }
    "#).expect("nullable omission is declared with the shared Maybe type");
    let adapters = crate::adapter::emit(&declarations).unwrap();
    let adapter = &adapters.iter().find(|out| out.path == "llm.rs").unwrap().text;
    assert!(adapter.contains("scalar = ::lk_agents_abi::Maybe::NotGiven"), "{adapter}");
    assert!(
        adapter.contains("scalar: ::lk_agents_abi::Maybe<::std::option::Option<f64>>"),
        "{adapter}"
    );
    assert!(adapter.contains("data: ::lk_agents_abi::Maybe<Bound<'_, PyAny>>"), "{adapter}");
    assert!(adapter.contains("Maybe::NotGiven => ::lk_agents_abi::Maybe::NotGiven"), "{adapter}");
    assert!(adapter.contains("if value.is_none() { Ok(None) }"), "{adapter}");
    assert!(adapter.contains("value_from_py(&value"), "{adapter}");
    let stubs = crate::class_stub::emit(&declarations).unwrap();
    let text = stubs.iter().map(|stub| stub.out.text.as_str()).collect::<String>();
    assert!(text.contains("from ..types import NotGivenOr"), "{text}");
    assert!(text.contains("scalar: NotGivenOr[float | None] = ..."), "{text}");
    assert!(text.contains("data: NotGivenOr[object | None] = ..."), "{text}");
}

#[test]
fn declared_mutable_classes_preserve_python_identity_and_subclassing() {
    let declarations = read(
        r#"
        #[crate::host(class(name = "Filter", module = "package.filters", subclass))]
        pub trait Filter: Sized {
            #[host(new)] fn new() -> Self;
            #[host(call)] fn update(&mut self, value: f64);
        }
    "#,
    )
    .expect("mutable subclass contract resolves");
    let adapters = crate::adapter::emit(&declarations).unwrap();
    let adapter = &adapters.iter().find(|out| out.path == "llm.rs").unwrap().text;
    assert!(
        adapter.contains("#[pyclass(subclass, name = \"Filter\", module = \"package.filters\")]"),
        "{adapter}"
    );
    assert!(adapter.contains("fn update(&mut self"), "{adapter}");
}

// Pure exports retain their pre-declaration fork behavior; resource entries keep the guard.
#[test]
fn fork_guards_follow_group_defaults_and_entry_overrides() {
    let declarations = read(
        r#"
        #[crate::host(class(pid = false))]
        pub trait Filter: Sized {
            #[host(new)] fn new() -> Self;
            #[host(call)] fn apply(&self, value: f64) -> f64;
        }
        #[crate::host(functions(pid = false))]
        pub trait Algorithms {
            #[host(function)] fn pure() -> u32;
            #[host(function, pid = true)] async fn admitted() -> u32;
        }
        #[crate::host(functions)]
        pub trait Runtime {
            #[host(function)] fn guarded() -> u32;
            #[host(function, pid = false)] fn pure_override() -> u32;
        }
    "#,
    )
    .expect("typed fork policy is declared for classes, function groups and entries");
    let adapters = crate::adapter::emit(&declarations).unwrap();
    let llm = &adapters.iter().find(|output| output.path == "llm.rs").unwrap().text;
    assert_eq!(llm.matches("::lk_agents_abi::python::check_pid()?").count(), 2, "{llm}");
    for invalid in ["false", "0"] {
        let value = if invalid == "false" { "\"false\"" } else { invalid };
        let declaration = format!(
            "#[crate::host(functions(pid = {value}))] pub trait Wrong {{ #[host(function)] fn call(); }}"
        );
        assert!(
            read(&declaration).err().is_some_and(|error| error.contains("`pid` takes a boolean"))
        );
    }
}

#[test]
fn input_maps_accept_covariant_mapping_values_while_results_remain_dicts() {
    let declarations = read(r#"
        use std::collections::BTreeMap;
        #[crate::host(functions)]
        pub trait Bytes {
            #[host(function)] fn copy(values: BTreeMap<String, Vec<u8>>) -> BTreeMap<String, Vec<u8>>;
            #[host(function)] fn objects(values: Vec<serde_json::Value>) -> Vec<serde_json::Value>;
        }
    "#).unwrap();
    let stubs = crate::class_stub::emit(&declarations).unwrap();
    let text = stubs.iter().map(|stub| stub.out.text.as_str()).collect::<String>();
    assert!(text.contains("from collections.abc import Mapping"), "{text}");
    assert!(text.contains("values: Mapping[str, bytes | bytearray | list[int]]"), "{text}");
    assert!(text.contains("-> dict[str, list[int]]"), "{text}");
    assert!(text.contains("values: Sequence[object]"), "{text}");
    assert!(text.contains("-> list[object]"), "{text}");
}
