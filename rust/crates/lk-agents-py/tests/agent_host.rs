//! The generated `AgentCallbacksProxy` against a live interpreter: what the host object receives,
//! the context each call runs in, refusals versus failures, and notifications.

use _core::generated::agent::AgentCallbacksProxy;
use lk_agents_abi::agent::{Invocation, OutputCandidate, ValidatedOutput};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// The package itself, beside the crate: the generated models import the SDK and the rest of the
/// package. Its dependencies come from the dev venv's site-packages on `PYTHONPATH`, whose `.pth`
/// installs (the protocol and SDK packages) only a site directory processes.
const MODELS: &str = r#"
import site, sys
for path in [p for p in sys.path if p.endswith("site-packages")]:
    site.addsitedir(path)
sys.path.insert(0, PACKAGE)
import livekit.agents._generated.agent_run
import livekit.agents._generated.tool_report
import livekit.agents.voice.agent_host
"#;

const HOST: &str = r#"
import asyncio, contextvars, sys, threading

models = sys.modules["livekit.agents._generated.agent_run"]
ToolArgumentError = sys.modules["livekit.agents.voice.agent_host"].ToolArgumentError
reports = sys.modules["livekit.agents._generated.tool_report"]

var = contextvars.ContextVar("var", default="unset")
loop = asyncio.new_event_loop()
ready = threading.Event()
def run():
    asyncio.set_event_loop(loop); loop.call_soon(ready.set); loop.run_forever()
threading.Thread(target=run, daemon=True).start()
ready.wait()

def capture(value):
    var.set(value)
    return (loop, contextvars.copy_context())

class Host:
    def __init__(self):
        self.seen = {}
        self.cancelled = threading.Event()

    async def render_instructions(self, invocation):
        self.seen["render"] = (var.get(), invocation)
        return "be brief"

    async def prepare_tool(self, invocation, arguments):
        if arguments.get("bad"):
            raise ToolArgumentError("city is required")
        if arguments.get("crash"):
            raise KeyError("crash")
        return models.PreparedArguments(
            prepared=arguments, identity=models.ArgumentIdentityJson(value=arguments)
        )

    async def run_tool(self, invocation, prepared):
        report = reports.ToolReport(outcome=reports.ToolOutcomeValue(text="answered"))
        if prepared.get("bad_arity"):
            return (report,)
        if prepared.get("bad_metadata"):
            return report, ["not a map"]
        return report, {"lk.delegation.task_id": "original-task"}

    async def validate_output(self, invocation, candidate):
        self.seen["candidate"] = candidate
        return models.ValidatedOutputRetry(reason="not a number")

    def cancel(self, invocation):
        self.seen["cancel"] = (var.get(), invocation.operation)
        self.cancelled.set()

host = Host()
"#;

/// Imports `MODELS` once per process, outside the interpreter lock, so a test waiting for the
/// import never holds it.
fn install_models() {
    static INSTALLED: std::sync::Once = std::sync::Once::new();
    INSTALLED.call_once(|| {
        Python::initialize();
        Python::attach(|py| {
            let globals = PyDict::new(py);
            let package = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../livekit-agents-next");
            globals.set_item("PACKAGE", package).unwrap();
            py.run(&std::ffi::CString::new(MODELS).unwrap(), Some(&globals), None).unwrap();
        });
    });
}

/// A fresh `HOST` in its own globals.
fn host_globals(py: Python<'_>) -> Bound<'_, PyDict> {
    let globals = PyDict::new(py);
    py.run(&std::ffi::CString::new(HOST).unwrap(), Some(&globals), None).unwrap();
    globals
}

fn invocation(context: lk_agents_abi::HostRef) -> Invocation {
    Invocation {
        operation: 7,
        invocation_id: "AI_1".into(),
        agent_name: "front".into(),
        execution_id: None,
        frame_id: None,
        context_scope: None,
        application_context: None,
        history: Vec::new(),
        function_call: None,
        approval: None,
        speech: None,
        host_context: Some(context),
    }
}

/// `invocation`'s copy for one tool call, as the HostInvoker derives it.
fn tool_call(invocation: &Invocation) -> Invocation {
    let call = lk_agents_abi::FunctionCall {
        id: "item_1".into(),
        call_id: "call_1".into(),
        arguments: "{}".into(),
        name: "lookup".into(),
        created_at: std::time::SystemTime::UNIX_EPOCH,
        extra: Default::default(),
        group_id: None,
    };
    Invocation { function_call: Some(call), ..invocation.clone() }
}

// Catches: a callback run outside the context its invocation captured, the captured context
// handed to the host, a refusal read as a failure (or a failure as a refusal), a host shape read
// by anything but its declaration, or a notification that never reaches the host.
#[test]
fn the_generated_agent_host_proxy_calls_the_host_in_the_captured_context() {
    install_models();
    let (globals, proxy, inv) = Python::attach(|py| {
        let globals = host_globals(py);
        let pair = globals.get_item("capture").unwrap().unwrap().call1(("captured",)).unwrap();
        let context = lk_agents_abi::python::intern_host(pair.unbind());
        let host = globals.get_item("host").unwrap().unwrap();
        (globals.unbind(), AgentCallbacksProxy::from_host(&host).unwrap(), invocation(context))
    });
    let runtime = lk_agents_runtime::process::handle().unwrap();
    runtime.block_on(async {
        assert_eq!(proxy.render_instructions(&inv).await.unwrap(), "be brief");
        let agent = inv.clone();
        let inv = tool_call(&agent);
        let args = |key: &str| {
            let mut m = serde_json::Map::new();
            m.insert(key.into(), true.into());
            m
        };
        let prepared = proxy.prepare_tool(&inv, args("city")).await.unwrap().unwrap();
        assert_eq!(
            prepared.identity,
            lk_agents_abi::agent::ArgumentIdentity::Json(serde_json::json!({"city": true}))
        );
        let (report, extra) = proxy.run_tool(&inv, &prepared.prepared).await.unwrap();
        assert!(matches!(
            report.outcome,
            lk_agents_abi::llm::tool_executor::ToolOutcome::Value { text, .. } if text == "answered"
        ));
        assert_eq!(extra["lk.delegation.task_id"], "original-task");
        for key in ["bad_arity", "bad_metadata"] {
            let prepared = proxy.prepare_tool(&inv, args(key)).await.unwrap().unwrap();
            let error = proxy.run_tool(&inv, &prepared.prepared).await.unwrap_err();
            assert_eq!(error.kind, "TypeError");
            let expected = if key == "bad_arity" { "tuple of 2 items" } else { "expected a dict" };
            assert!(error.message.contains(expected), "{error}");
        }
        let refused = proxy.prepare_tool(&inv, args("bad")).await.unwrap().unwrap_err();
        assert_eq!(refused.message, "city is required");
        let failed = proxy.prepare_tool(&inv, args("crash")).await.unwrap_err();
        assert_eq!(failed.kind, "KeyError");
        let verdict =
            proxy.validate_output(&agent, OutputCandidate::Value(serde_json::json!(3))).await;
        assert!(matches!(verdict, Ok(ValidatedOutput::Retry(r)) if r == "not a number"));
    });
    proxy.cancel(&inv);
    let cancelled = || {
        Python::attach(|py| {
            let host = globals.bind(py).get_item("host").unwrap().unwrap();
            host.getattr("cancelled").unwrap().call_method0("is_set").unwrap().is_truthy().unwrap()
        })
    };
    for _ in 0..500 {
        if cancelled() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(cancelled(), "the notification never reached the host");
    Python::attach(|py| {
        let host = globals.bind(py).get_item("host").unwrap().unwrap();
        let seen = host.getattr("seen").unwrap();
        let (var, invocation): (String, Bound<'_, PyAny>) =
            seen.get_item("render").unwrap().extract().unwrap();
        assert_eq!(var, "captured");
        assert_eq!(invocation.getattr("agent_name").unwrap().extract::<String>().unwrap(), "front");
        assert!(invocation.getattr("host_context").is_err(), "the context stays native");
        let candidate = seen.get_item("candidate").unwrap();
        let class = candidate.get_type().name().unwrap().to_string();
        assert_eq!(class, "OutputCandidateValue");
        assert_eq!(candidate.getattr("value").unwrap().extract::<i64>().unwrap(), 3);
        let (var, operation): (String, u64) = seen.get_item("cancel").unwrap().extract().unwrap();
        assert_eq!((var.as_str(), operation), ("captured", 7));
    });
}

// Catches: a tool callback dispatched without the tool call it belongs to (the host would run a
// body with no call scope), or a refusal that enters the host anyway.
#[test]
fn a_tool_callback_without_a_tool_call_scope_is_refused_before_the_host() {
    install_models();
    let (globals, proxy, inv) = Python::attach(|py| {
        let globals = host_globals(py);
        py.run(c"async def entered(invocation, arguments):\n    host.seen['entered'] = True\nhost.prepare_tool = entered\n", Some(&globals), None).unwrap();
        let pair = globals.get_item("capture").unwrap().unwrap().call1(("captured",)).unwrap();
        let context = lk_agents_abi::python::intern_host(pair.unbind());
        let host = globals.get_item("host").unwrap().unwrap();
        (globals.unbind(), AgentCallbacksProxy::from_host(&host).unwrap(), invocation(context))
    });
    let runtime = lk_agents_runtime::process::handle().unwrap();
    let error = runtime.block_on(proxy.prepare_tool(&inv, serde_json::Map::new())).unwrap_err();
    assert_eq!(error.kind, "RuntimeError");
    assert!(
        error.message.contains("AgentCallbacks.prepare_tool: the call requires a tool call scope"),
        "{error}"
    );
    Python::attach(|py| {
        let seen = globals.bind(py).get_item("host").unwrap().unwrap().getattr("seen").unwrap();
        assert!(seen.get_item("entered").is_err(), "the host was entered");
    });
}

// Catches: a record read from anything but its model (a kind-dict, a look-alike), an explicit
// None accepted for a field that is not optional, or a default applied to anything but omission.
#[test]
fn records_cross_as_their_models_with_declared_presence() {
    use lk_agents_abi::agent::Retries;
    install_models();
    Python::attach(|py| {
        let globals = host_globals(py);
        let eval = |code: &str| {
            py.eval(&std::ffi::CString::new(code).unwrap(), Some(&globals), None).unwrap()
        };

        let accepted = eval("models.ValidatedOutputAccepted(text=None, value=None)");
        assert!(matches!(
            _core::generated::records::agent_run::validated_output_from_py(&accepted).unwrap(),
            ValidatedOutput::Accepted { text: None, value: None, serialized: None }
        ));
        let dict = _core::generated::records::agent_run::validated_output_from_py(&eval(
            "{'kind': 'retry', 'reason': 'no'}",
        ))
        .unwrap_err();
        assert!(dict.is_instance_of::<pyo3::exceptions::PyTypeError>(py), "{dict}");
        assert!(
            dict.to_string().contains("ValidatedOutputAccepted | ValidatedOutputRetry"),
            "{dict}"
        );

        let null = _core::generated::records::agent_run::read_back_answer_from_py(&eval(
            "models.ReadBackAnswerInputRequired(question=None, verbatim=False)",
        ))
        .unwrap_err();
        assert!(null.to_string().contains("ReadBackAnswerInputRequired.question"), "{null}");

        assert_eq!(
            _core::generated::records::agent_run::retries_from_py(&eval("models.Retries()"))
                .unwrap(),
            Retries { tools: 1, output: 1 }
        );
        assert_eq!(
            _core::generated::records::agent_run::retries_from_py(&eval(
                "models.Retries(output=0)"
            ))
            .unwrap(),
            Retries { tools: 1, output: 0 }
        );
        assert!(
            _core::generated::records::agent_run::retries_from_py(&eval(
                "models.Retries(tools=None)"
            ))
            .is_err(),
            "None is not omission"
        );
    });
}
