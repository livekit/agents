// Catches: a host class generic over several type parameters annotated with one (or none), which
// strict type checking refuses in the generated stub.
#[test]
fn a_proxy_is_annotated_with_each_type_parameter() {
    let source = |generic: &str| {
        format!(
            r#"use crate::errors::HostError;
            #[crate::host(proxy(host = "livekit.agents.Mux"{generic}))]
            pub trait Mux: Send + Sync {{
                #[host(call)] fn ping(&self) -> Result<(), HostError>;
            }}
            #[crate::host(class(name = "Pool"))]
            pub trait Pool: Sized {{ #[host(new)] fn new(host: std::sync::Arc<dyn Mux>) -> Self; }}"#
        )
    };
    let stub = |generic: &str| {
        let decls = crate::decl::tests::read(&source(generic)).expect("resolves");
        super::emit(&decls).expect("stubs")[0].out.text.clone()
    };
    for (generic, want) in [
        ("", "host: Mux)"),
        (", generic", "host: Mux[Any])"),
        (", generic = 2", "host: Mux[Any, Any])"),
    ] {
        let text = stub(generic);
        assert!(text.contains(want), "{want}\n{text}");
    }
    let err = crate::decl::tests::read(&source(", generic = 0"))
        .expect_err("no type parameter is refused");
    assert!(err.contains("positive type parameter count"), "{err}");
}
