//! The generated `AgentHostProxy` against a live interpreter: what the host object receives,
//! the context each call runs in, refusals versus failures, and notifications.

use lk_agents_abi::agent::{Invocation, OutputCandidate, ValidatedOutput};
use _core::generated::agent::AgentHostProxy;
use pyo3::prelude::*;
use pyo3::types::PyDict;

const HOST: &str = r#"
import asyncio, contextvars, sys, threading, types

for parent in ("livekit", "livekit.agents", "livekit.agents.voice"):
    sys.modules.setdefault(parent, types.ModuleType(parent))
module = types.ModuleType("livekit.agents.voice.agent_host")
class ToolArgumentError(Exception):
    pass
module.ToolArgumentError = ToolArgumentError
sys.modules["livekit.agents.voice.agent_host"] = module

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
        return {"prepared": arguments, "identity": {"kind": "json", "value": arguments}}

    async def validate_output(self, invocation, candidate):
        self.seen["candidate"] = candidate
        return {"kind": "retry", "reason": "not a number"}

    def cancel(self, invocation):
        self.seen["cancel"] = (var.get(), invocation["operation"])
        self.cancelled.set()

host = Host()
"#;

fn invocation(context: lk_agents_abi::HostRef) -> Invocation {
    Invocation {
        operation: 7,
        invocation_id: "AI_1".into(),
        agent_name: "front".into(),
        function_call: None,
        approval: None,
        host_context: Some(context),
    }
}

// Catches: a callback run outside the context its invocation captured, the captured context
// handed to the host, a refusal read as a failure (or a failure as a refusal), a host shape read
// by anything but its declaration, or a notification that never reaches the host.
#[test]
fn the_generated_agent_host_proxy_calls_the_host_in_the_captured_context() {
    Python::initialize();
    let (globals, proxy, inv) = Python::attach(|py| {
        let globals = PyDict::new(py);
        py.run(&std::ffi::CString::new(HOST).unwrap(), Some(&globals), None).unwrap();
        let pair = globals.get_item("capture").unwrap().unwrap().call1(("captured",)).unwrap();
        let context = lk_agents_abi::python::intern_host(pair.unbind());
        let host = globals.get_item("host").unwrap().unwrap();
        (globals.unbind(), AgentHostProxy::from_host(&host).unwrap(), invocation(context))
    });
    let runtime = lk_agents_runtime::process::handle().unwrap();
    runtime.block_on(async {
        assert_eq!(proxy.render_instructions(&inv).await.unwrap(), "be brief");
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
        let refused = proxy.prepare_tool(&inv, args("bad")).await.unwrap().unwrap_err();
        assert_eq!(refused.message, "city is required");
        let failed = proxy.prepare_tool(&inv, args("crash")).await.unwrap_err();
        assert_eq!(failed.kind, "KeyError");
        let verdict =
            proxy.validate_output(&inv, OutputCandidate::Value(serde_json::json!(3))).await;
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
        let (var, invocation): (String, Bound<'_, PyDict>) =
            seen.get_item("render").unwrap().extract().unwrap();
        assert_eq!(var, "captured");
        assert_eq!(invocation.get_item("agent_name").unwrap().unwrap().extract::<String>().unwrap(), "front");
        assert!(invocation.get_item("host_context").unwrap().is_none(), "the context stays native");
        let candidate = seen.get_item("candidate").unwrap();
        assert_eq!(candidate.get_item("kind").unwrap().extract::<String>().unwrap(), "value");
        let (var, operation): (String, u64) = seen.get_item("cancel").unwrap().extract().unwrap();
        assert_eq!((var.as_str(), operation), ("captured", 7));
    });
}
